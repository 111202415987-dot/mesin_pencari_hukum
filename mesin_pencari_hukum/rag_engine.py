"""
rag_engine.py
=============
Lapisan RAG anti-halusinasi, disesuaikan dengan notebook
STKI_Chatbot_RAG_Hukum_Indonesia_REVISI.ipynb (bagian 5).

Sama dengan notebook:
  * konteks tiap sumber memuat judul | kategori | tahun | rentang halaman | berkas,
  * LLM default Gemini (`gemini-2.5-flash`); Groq tersedia sebagai alternatif,
  * tanpa API key -> mode retrieval-only (tidak berpura-pura menghasilkan jawaban LLM),
  * bila retrieval kosong, LLM tidak dipanggil.

Tambahan di atas notebook: jawaban di-stream dan setiap kalimat diverifikasi
(apakah mengutip [S#] yang valid dan apakah isinya didukung teks sumber).
Verifikasi ini heuristik tumpang-tindih term, BUKAN jaminan bebas halusinasi.
"""
from __future__ import annotations

import re
from typing import Dict, Generator, List, Sequence

from preprocessing import preprocess_text

PROVIDERS = {
    "Gemini": {"env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
    "Groq": {"env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
}

NOT_FOUND = "Informasi tidak ditemukan dalam dokumen yang tersedia."

REFUSAL = ("Saya belum menemukan dokumen hukum yang cukup relevan dalam dataset untuk menjawab pertanyaan ini. "
           "Coba istilah atau nomor peraturan yang lebih spesifik, atau rujuk sumber resmi "
           "(mis. peraturan.bpk.go.id).")

SYSTEM_PROMPT = """Anda adalah asisten informasi hukum Indonesia berbasis dokumen. Jawab dalam Bahasa Indonesia yang jelas dan ringkas.
ATURAN KETAT:
1. Jawab HANYA berdasarkan KONTEKS yang diberikan. Dilarang memakai pengetahuan di luar konteks.
2. Setiap pernyataan penting harus diberi sitasi dengan format [S1], [S2], dst. sesuai nomor sumber, dan sebutkan nama peraturan serta nomor pasalnya bila tertulis di konteks.
3. Jika konteks tidak memuat jawaban, jawab persis: "Informasi tidak ditemukan dalam dokumen yang tersedia." lalu sarankan verifikasi ke sumber resmi.
4. Jangan mengarang nomor pasal, ayat, sanksi, angka, tahun, atau nama peraturan yang tidak tertulis di konteks.
5. Bedakan ringkasan isi dokumen dari nasihat hukum. Akhiri dengan: "Ini bukan nasihat hukum; verifikasi ke sumber resmi." """

_PASAL_RE = re.compile(r"\bPasal\s+(\d+[A-Z]?)\b")
_CITE_RE = re.compile(r"\[S(\d+)\]")


def pasal_label(text: str) -> str:
    """Label pasal yang muncul di chunk (mis. 'Pasal 5–6'); '-' jika tidak ada."""
    found = list(dict.fromkeys(_PASAL_RE.findall(text)))
    if not found:
        return "-"
    return f"Pasal {found[0]}" if len(found) == 1 else f"Pasal {found[0]}–{found[-1]}"


def page_range(chunk: dict) -> str:
    a, b = chunk.get("page_start"), chunk.get("page_end")
    return f"{a}" if a == b or b is None else f"{a}-{b}"


def source_record(engine, result: dict) -> dict:
    """Satu sumber lengkap (dipakai untuk konteks LLM dan tampilan sitasi)."""
    ch = engine.chunks[result["idx"]]
    doc = engine.corpus.doc_by_id[ch["doc_id"]]
    return {
        "doc_id": doc["doc_id"], "chunk_id": ch.get("chunk_id", ""), "title": doc["title"],
        "category": doc.get("category"), "year": doc.get("year") or "-",
        "pasal": pasal_label(ch["text"]), "pages": page_range(ch),
        "file": doc.get("source_path", ""), "score": float(result["score"]), "text": ch["text"],
    }


def build_context(sources: Sequence[dict]) -> str:
    """Format konteks sama dengan notebook: [S#] judul | kategori | Tahun | Halaman | File."""
    return "\n\n".join(
        f"[S{i}] {s['title']} | {s['category']} | Tahun: {s['year']} | {s['pasal']} | "
        f"Halaman: {s['pages']} | File: {s['file']}\n{s['text']}"
        for i, s in enumerate(sources, 1))


def user_prompt(query: str, context: str) -> str:
    return f"KONTEKS DOKUMEN:\n{context}\n\nPERTANYAAN: {query}\n\nJAWABAN:"


def stream_llm(provider: str, api_key: str, model: str, query: str, context: str) -> Generator[str, None, None]:
    """Stream potongan teks jawaban. Melempar exception bila API gagal (ditangani pemanggil)."""
    prompt = user_prompt(query, context)
    if provider == "Gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1, max_output_tokens=2048)
        for part in client.models.generate_content_stream(model=model, contents=prompt, config=cfg):
            if getattr(part, "text", None):
                yield part.text
    elif provider == "Groq":
        from groq import Groq

        stream = Groq(api_key=api_key).chat.completions.create(
            model=model, temperature=0.1, max_tokens=900, stream=True,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        for part in stream:
            delta = part.choices[0].delta.content
            if delta:
                yield delta
    else:
        raise ValueError(f"Provider tidak dikenal: {provider}")


def evidence_only_text(n: int, reason: str = "") -> str:
    """Pesan mode retrieval-only; jujur bahwa tidak ada jawaban generatif."""
    base = ("**Mode retrieval-only:** jawaban generatif belum dijalankan"
            + (f" ({reason})" if reason else "")
            + f". Berikut {n} sumber paling relevan dari dataset untuk diperiksa langsung.")
    return base


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def validate_answer(text: str, sources: Sequence[str], min_overlap: float = 0.4) -> Dict:
    """
    Verifikasi per kalimat. Status: supported | weak | no_cite | bad_cite.
    grounding = proporsi kalimat berstatus 'supported' (None jika tidak ada klaim).
    """
    n = len(sources)
    src_tokens = [set(preprocess_text(s, stem=False)) for s in sources]
    claims = []
    for s in _sentences(text):
        low = s.lower()
        if "bukan nasihat hukum" in low or "tidak ditemukan" in low:
            continue
        content = [t for t in preprocess_text(_CITE_RE.sub(" ", s), stem=False) if len(t) > 2]
        if len(content) < 3:
            continue
        refs = [int(x) for x in _CITE_RE.findall(s)]
        valid = [r for r in refs if 1 <= r <= n]
        if not refs:
            status, overlap = "no_cite", 0.0
        elif not valid:
            status, overlap = "bad_cite", 0.0
        else:
            pool = set().union(*(src_tokens[r - 1] for r in valid))
            overlap = sum(t in pool for t in content) / len(content)
            status = "supported" if overlap >= min_overlap else "weak"
        claims.append({"sentence": s, "status": status, "refs": refs, "overlap": round(overlap, 2)})
    total = len(claims)
    ok = sum(c["status"] == "supported" for c in claims)
    return {
        "claims": claims,
        "grounding": (ok / total) if total else None,
        "n_supported": ok,
        "n_total": total,
        "bad_refs": sorted({r for c in claims for r in c["refs"] if not 1 <= r <= n}),
    }
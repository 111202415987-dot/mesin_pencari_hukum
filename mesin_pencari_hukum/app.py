"""
app.py
======
Antarmuka Streamlit terpadu: Cari · Tanya Hukum (RAG) · Sitasi · Lab · Evaluasi · Korpus
Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mesin Pencari Hukum Indonesia", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="expanded")

import ui  # noqa: E402
from preprocessing import (build_corpus, case_fold, dtm_dataframe, get_sample_corpus,  # noqa: E402
                           incidence_dataframe, make_terms, remove_stopwords, save_metadata_json,
                           sparsity, stem_tokens, stemmer_name, tokenize)
from rag_engine import (PROVIDERS, REFUSAL, build_context, evidence_only_text, pasal_label,  # noqa: E402
                        source_record, stream_llm, validate_answer)
from retrieval_engine import DEFAULT_EMBED_MODEL, SearchEngine, ast_to_string, make_snippet  # noqa: E402

ui.inject_css()

METHODS = ["Hybrid", "BM25", "TF-IDF", "LSA", "Embedding", "Boolean", "Frasa"]
EXAMPLES = ["gaji pekerja yang di-PHK", "perlindungan konsumen barang", "sanksi pidana korupsi",
            "upah minimum", "(pekerja OR buruh) AND upah"]


KNOWN_CATEGORIES = {"UU", "PP", "UUD", "UUDRT"}


def _secret(name: str) -> str:
    try:
        return os.environ.get(name) or st.secrets.get(name, "") or ""
    except Exception:
        return os.environ.get(name, "")


def find_law_dataset_root(base: str):
    """Cari folder yang berisi sub-folder UU/PP/UUD/UUDRT (boleh berada di dalam satu folder pembungkus)."""
    if not os.path.isdir(base):
        return None
    for dp, dirs, _ in os.walk(base):
        if KNOWN_CATEGORIES & {d.upper() for d in dirs}:
            return dp
        if dp[len(base):].count(os.sep) >= 3:
            dirs[:] = []
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">⚖️ Hukum<span>Search</span></div>', unsafe_allow_html=True)
    with st.expander("📁 Basis pengetahuan", expanded=True):
        data_dir = st.text_input("Folder dataset Kaggle", value=os.getenv("LAW_DATA_DIR", "data/law-indonesia"),
                                 help="Dataset hermansugiharto/law-indonesia dengan sub-folder UU, PP, UUD, UUDRT.")
        dataset_root = find_law_dataset_root(data_dir)
        if dataset_root:
            st.success("Dataset terdeteksi", icon="✅")
        else:
            st.warning("Folder UU/PP/UUD/UUDRT belum ditemukan di lokasi ini.")
        demo_mode = st.toggle("Mode demo (korpus contoh)", value=False, disabled=bool(dataset_root),
                              help="Hanya untuk mencoba tampilan. Isinya BUKAN peraturan resmi dan tidak boleh dipakai untuk hasil eksperimen.")
        max_docs = st.number_input("Batas dokumen (0 = semua)", 0, 10000, 200, step=50,
                                   disabled=not dataset_root, help="Notebook memakai 200 sebagai batas awal.")
        chunk_size = st.slider("Ukuran chunk (kata)", 40, 400, 100, step=10)
        overlap = st.slider("Overlap (kata)", 0, 100, 20, step=5)
        use_bigram = st.checkbox("Sertakan bigram", True)
        stem = st.checkbox("Stemming Sastrawi", True)
        st.caption(f"Stemmer: **{stemmer_name()}**")
        build_btn = st.button("🔄 Bangun / muat ulang basis pengetahuan", type="primary", width="stretch")
    top_k = st.slider("Jumlah hasil (Top-K)", 3, 30, 10)
    with st.expander("🧠 Mesin semantik"):
        lsa_pref = st.slider("Komponen LSA (k)", 2, 300, 50)
        model_name = st.text_input("Model embedding", DEFAULT_EMBED_MODEL)
        backend = st.selectbox("Vector store", ["faiss", "chroma", "numpy"])
        sem_btn = st.button("🚀 Bangun vector store", width="stretch")
    with st.expander("🤖 LLM untuk RAG"):
        provider = st.radio("Penyedia", list(PROVIDERS), horizontal=True,
                            help="Notebook memakai Gemini. Groq tersedia sebagai alternatif.")
        api_key = st.text_input(PROVIDERS[provider]["env"], value=_secret(PROVIDERS[provider]["env"]), type="password",
                                key=f"key_{provider}",
                                help="Tidak disimpan. Bisa juga lewat env var atau .streamlit/secrets.toml. Kosong = retrieval-only.")
        llm_model = st.text_input("Model", PROVIDERS[provider]["model"], key=f"model_{provider}")


@st.cache_resource(show_spinner=False)
def load_engine(root, demo, max_docs, chunk_size, overlap, use_bigram, stem):
    if demo:
        corpus = get_sample_corpus(chunk_size, overlap, stem)
    else:
        corpus = build_corpus(root, chunk_size, overlap, max_docs or None, stem)
        if not corpus.docs or not corpus.chunks:
            raise RuntimeError("Tidak ada PDF/TXT yang berhasil dibaca. Periksa struktur UU/PP/UUD/UUDRT "
                               "dan pastikan PDF memiliki lapisan teks (bukan hasil pindai).")
        if corpus.config.get("source") == "sample":
            raise RuntimeError("Korpus contoh tidak diizinkan sebagai basis pengetahuan utama.")
    return SearchEngine(corpus, use_bigram=use_bigram)


if not dataset_root and not demo_mode:
    st.markdown(ui.hero([("Dokumen", "0"), ("Chunk", "0"), ("Vocabulary", "0"), ("Sitasi tersimpan", "0")]),
                unsafe_allow_html=True)
    st.markdown(ui.empty("📂", "Dataset belum ditemukan",
                         f"Unduh hermansugiharto/law-indonesia dari Kaggle dan ekstrak ke “{data_dir}” "
                         "(berisi UU, PP, UUD, UUDRT), lalu klik Bangun. Aplikasi tidak memakai korpus contoh "
                         "secara diam-diam. Untuk sekadar mencoba tampilan, aktifkan Mode demo di sidebar."),
                unsafe_allow_html=True)
    st.stop()

params = (dataset_root or "", bool(demo_mode and not dataset_root), int(max_docs), chunk_size, overlap, use_bigram, stem)
if "engine" not in st.session_state or build_btn:
    try:
        with st.status("Membaca peraturan, preprocessing, membangun inverted index…", expanded=False) as s:
            st.session_state["engine"] = load_engine(*params)
            st.session_state["params"] = params
            s.update(label="Basis pengetahuan siap", state="complete")
        if build_btn:
            st.toast("Basis pengetahuan dibangun ulang", icon="✅")
    except Exception as e:
        st.error(f"Gagal membangun basis pengetahuan: {e}")
        st.stop()
engine: SearchEngine = st.session_state["engine"]

if sem_btn:
    with st.status("Menghitung embedding seluruh chunk (pertama kali mengunduh model)…", expanded=True) as s:
        try:
            store = engine.build_semantic(model_name, backend)
            s.update(label=f"Vector store siap · {store.backend} · {len(engine.chunks)} vektor", state="complete")
            for note in store.notes:
                st.info(note)
        except Exception as e:
            s.update(label="Gagal membangun embedding", state="error")
            st.error(str(e))

doc_by_id = engine.corpus.doc_by_id
n_chunks = len(engine.chunks)
LSA_K = max(2, min(lsa_pref, n_chunks - 1, engine.index.vocab_size - 1)) if n_chunks > 2 else 2
st.session_state.setdefault("saved", {})
st.session_state.setdefault("chat", [])

# ---------------------------------------------------------------------------
# Hero + pipeline status
# ---------------------------------------------------------------------------
st.markdown(ui.hero([("Dokumen", f"{len(engine.corpus.docs):,}"), ("Chunk", f"{n_chunks:,}"),
                     ("Vocabulary", f"{engine.index.vocab_size:,}"),
                     ("Sitasi tersimpan", str(len(st.session_state['saved'])))]), unsafe_allow_html=True)
st.markdown(ui.pipeline([
    ("Korpus", "done", f"{len(engine.corpus.docs)} dok · {n_chunks} chunk"),
    ("Inverted index", "done", f"{engine.index.vocab_size:,} term"),
    ("LSA", "done" if engine._lsa_cache else "idle", "siap" if engine._lsa_cache else "dibangun saat dipakai"),
    ("Embedding", "done" if engine.semantic else "idle",
     engine.semantic.backend if engine.semantic else "belum dibangun"),
    (f"LLM · {provider}", "done" if api_key else "warn", llm_model if api_key else "retrieval-only"),
]), unsafe_allow_html=True)
if st.session_state.get("params") != params:
    st.info("Konfigurasi korpus berubah — klik **Bangun / muat ulang basis pengetahuan** untuk menerapkan.")
if engine.corpus.config.get("source") == "sample":
    st.warning("**MODE DEMO** — korpus contoh, bukan peraturan resmi. Jangan dipakai untuk hasil eksperimen atau laporan.", icon="🧪")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def citation(chunk: dict, doc: dict) -> str:
    year = doc.get("year") or "t.t."
    pages = (f"hlm. {chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
             else f"hlm. {chunk['page_start']}–{chunk['page_end']}")
    return (f"Pemerintah Republik Indonesia. ({year}). {doc['title']}. "
            f"[{doc['category']}] {chunk['chunk_id']}, {pages}. Berkas: {doc.get('source_path', '-')}. "
            f"Sumber data: Kaggle — hermansugiharto/law-indonesia.")


def toggle_save(chunk_id: str, cite: str, text: str) -> None:
    saved = st.session_state["saved"]
    if chunk_id in saved:
        del saved[chunk_id]
        st.toast("Dihapus dari sitasi", icon="🗑️")
    else:
        saved[chunk_id] = {"cite": cite, "text": text}
        st.toast("Disimpan ke sitasi", icon="📌")


def render_results(results, query, key, show_parts=False):
    if not results:
        st.markdown(ui.empty("🔍", "Tidak ada hasil", "Coba kata kunci lain atau metode berbasis makna (LSA / Embedding)."),
                    unsafe_allow_html=True)
        return
    top = max(r["score"] for r in results) or 1.0
    for rank, r in enumerate(results, start=1):
        ch = engine.chunks[r["idx"]]
        doc = doc_by_id[ch["doc_id"]]
        chips = (ui.chip(doc["category"], "ac") + ui.chip(str(doc.get("year") or "-"))
                 + ui.chip(f"hlm. {ch['page_start']}–{ch['page_end']}") + ui.chip(pasal_label(ch["text"]))
                 + ui.chip(ch["chunk_id"]))
        if show_parts:
            chips += ui.chip(f"leksikal {r.get('lex', 0):.3f}", "warn") + ui.chip(f"semantik {r.get('sem', 0):.3f}", "ok")
        for m in (r.get("matched") or [])[:6]:
            chips += ui.chip(m.replace("_", " "), "ok")
        st.markdown(ui.result_card(rank, doc["title"], chips,
                                   make_snippet(ch["text"], query, stem=engine.stem),
                                   r["score"], r["score"] / top), unsafe_allow_html=True)
        cite = citation(ch, doc)
        c1, c2 = st.columns([6, 1.2])
        with c1:
            with st.expander("📎 Sitasi, metadata & teks lengkap"):
                st.code(cite, language=None)
                st.json(doc, expanded=False)
                st.write(ch["text"])
        with c2:
            saved = ch["chunk_id"] in st.session_state["saved"]
            st.button("✅ Tersimpan" if saved else "📌 Simpan", key=f"{key}_{rank}_{ch['chunk_id']}",
                      on_click=toggle_save, args=(ch["chunk_id"], cite, ch["text"]), width="stretch")


def run_method(method, q, k, p):
    """Return (results, info_text, show_parts)."""
    if method == "BM25":
        return engine.search_bm25(q, p["k1"], p["b"], k), "", False
    if method == "TF-IDF":
        return engine.search_tfidf(q, p["tf_mode"], p["norm"], True, k), "", False
    if method == "LSA":
        return engine.search_lsa(q, LSA_K, k), f"LSA dengan {LSA_K} komponen laten", False
    if method == "Embedding":
        if engine.semantic is None:
            return [], "Vector store belum dibangun — gunakan panel **Mesin semantik** di sidebar.", False
        return engine.search_semantic(q, k), f"Backend: {engine.semantic.backend}", False
    if method == "Boolean":
        res, ast, n = engine.search_boolean(q, k)
        return res, f"Parse tree: `{ast_to_string(ast)}` · {n} dari {n_chunks} chunk cocok", False
    if method == "Frasa":
        return engine.search_phrase(q, k), "Pencarian frasa berurutan (positional index)", False
    src = "embedding" if (p["hy_src"] == "embedding" and engine.semantic) else "lsa"
    res, used = engine.search_hybrid(q, p["alpha"], p["fusion"], src, 50, p["k1"], p["b"], LSA_K, k)
    return res, f"Fusi **{p['fusion']}** · komponen semantik: **{used.upper()}**", True


def _pick_example():
    v = st.session_state.get("ex_pick")
    if v:
        st.session_state["q"] = v
        st.session_state["ex_pick"] = None


# Navigasi berbasis session_state (bukan st.tabs): pilihan halaman tetap setelah rerun/chat,
# dan hanya halaman aktif yang dieksekusi (LSA, DTM, dsb. tidak dihitung ulang di halaman tersembunyi).
NAV = ["Cari", "RAG", "Sitasi", "Lab", "Evaluasi", "Korpus"]
NAV_LABEL = {"Cari": "🔍 Cari", "RAG": "💬 Tanya Hukum (RAG)", "Sitasi": "📌 Sitasi", "Lab": "🧪 Lab",
             "Evaluasi": "📈 Evaluasi", "Korpus": "📚 Korpus"}
n_saved = len(st.session_state["saved"])
nav = st.segmented_control("Halaman", NAV, default="Cari", key="nav", label_visibility="collapsed",
                           format_func=lambda x: f"📌 Sitasi ({n_saved})" if x == "Sitasi" else NAV_LABEL[x]) or "Cari"

# ===========================================================================
# TAB 1 — Cari
# ===========================================================================
if nav == "Cari":
    st.session_state.setdefault("q", "gaji pekerja yang di-PHK")
    q = st.text_input("Kueri", key="q", label_visibility="collapsed",
                      placeholder="Ketik pertanyaan atau kata kunci hukum… (Boolean: pakai AND / OR / NOT)")
    st.pills("Contoh", EXAMPLES, key="ex_pick", on_change=_pick_example, label_visibility="collapsed")
    mc, pc = st.columns([5, 1.3])
    method = mc.segmented_control("Metode", METHODS, default="Hybrid", key="method",
                                  label_visibility="collapsed") or "Hybrid"
    P = {"k1": 1.5, "b": 0.75, "tf_mode": "log", "norm": True, "alpha": 0.5, "fusion": "rrf", "hy_src": "embedding"}
    with pc.popover("⚙️ Parameter", width="stretch"):
        P["k1"] = st.slider("BM25 k1", 0.0, 3.0, 1.5, 0.1)
        P["b"] = st.slider("BM25 b", 0.0, 1.0, 0.75, 0.05)
        P["tf_mode"] = st.radio("TF (TF-IDF)", ["log", "raw"], horizontal=True)
        P["norm"] = st.checkbox("Normalisasi panjang (cosine)", True)
        P["fusion"] = st.radio("Fusi Hybrid", ["rrf", "weighted"], horizontal=True)
        P["alpha"] = st.slider("α bobot leksikal (weighted)", 0.0, 1.0, 0.5, 0.05)
        P["hy_src"] = st.radio("Sumber semantik Hybrid", ["embedding", "lsa"], horizontal=True)

    if not q.strip():
        st.markdown(ui.empty("⌨️", "Mulai dengan kueri", "Contoh: “gaji pekerja” akan menemukan pasal tentang “upah” "
                                                          "lewat metode LSA / Hybrid (vocabulary mismatch)."),
                    unsafe_allow_html=True)
    elif n_chunks < 3 and method in ("LSA", "Hybrid"):
        st.warning("Korpus terlalu kecil untuk LSA.")
    else:
        t0 = time.perf_counter()
        results, info, parts = run_method(method, q, top_k, P)
        ms = (time.perf_counter() - t0) * 1000
        st.caption(f"**{len(results)} hasil** · {method} · {ms:.0f} ms" + (f" · {info}" if info else ""))
        view = st.segmented_control("Tampilan", ["Hasil", "Bandingkan metode"], default="Hasil",
                                    key="view", label_visibility="collapsed") or "Hasil"
        if view == "Hasil":
            render_results(results, q, "s", show_parts=parts)
        else:
            cols = {}
            for m in ["BM25", "TF-IDF", "LSA", "Hybrid"] + (["Embedding"] if engine.semantic else []):
                try:
                    cols[m] = [engine.chunks[r["idx"]]["chunk_id"] for r in run_method(m, q, 5, P)[0]]
                except Exception:
                    cols[m] = []
            n = max((len(v) for v in cols.values()), default=0)
            st.dataframe(pd.DataFrame({k: v + [""] * (n - len(v)) for k, v in cols.items()},
                                      index=range(1, n + 1)), width="stretch")
            st.caption("Chunk yang muncul di banyak kolom adalah kandidat paling kuat; perbedaan peringkat "
                       "menunjukkan kekuatan tiap metode.")

# ===========================================================================
# TAB 2 — RAG
# ===========================================================================
if nav == "RAG":
    h1, h2, h3, h4 = st.columns([2, 1.2, 1.2, 0.8])
    h1.markdown("**Jawaban hanya dari dokumen yang diindeks**, dengan sitasi `[S#]` dan verifikasi otomatis per kalimat.")
    retr = h2.selectbox("Retriever", ["Hybrid", "BM25", "Embedding"], label_visibility="collapsed")
    n_ctx = h3.slider("Konteks (chunk)", 1, 8, 4)
    if h4.button("🗑️ Bersihkan", width="stretch"):
        st.session_state["chat"] = []
        st.rerun()
    if not api_key:
        st.info(f"Belum ada {PROVIDERS[provider]['env']} — mode **retrieval-only**: sistem menampilkan sumber "
                "paling relevan beserta halaman, tanpa membuat jawaban generatif.")

    def retrieve_for_rag(q):
        if retr == "Embedding" and engine.semantic:
            return engine.search_semantic(q, n_ctx)
        if retr == "Hybrid" and n_chunks >= 3:
            src = "embedding" if engine.semantic else "lsa"
            return engine.search_hybrid(q, 0.5, "rrf", src, 50, 1.5, 0.75, LSA_K, n_ctx)[0]
        return engine.search_bm25(q, 1.5, 0.75, n_ctx)

    def render_assistant(msg):
        st.markdown(ui.cite_markdown(msg["content"]))
        v = msg.get("verdict")
        if v:
            if v["grounding"] is None:
                st.markdown(ui.verdict(None, "Tidak ada klaim untuk diverifikasi", "Jawaban berupa penolakan atau terlalu pendek."),
                            unsafe_allow_html=True)
            else:
                st.markdown(ui.verdict(v["grounding"], f"{v['n_supported']} dari {v['n_total']} klaim didukung sumber",
                                       "Heuristik tumpang-tindih term dengan sumber yang dikutip, bukan jaminan kebenaran."),
                            unsafe_allow_html=True)
            weak = [c for c in v["claims"] if c["status"] != "supported"]
            if weak:
                with st.expander(f"⚠️ {len(weak)} klaim perlu dicek manual"):
                    label = {"weak": "dukungan lemah", "no_cite": "tanpa sitasi", "bad_cite": "sitasi tidak valid"}
                    for c in weak:
                        st.markdown(f"- {ui.esc(c['sentence'])}  \n  _{label[c['status']]}_")
        for i, s in enumerate(msg.get("sources", []), 1):
            with st.expander(f"[S{i}] {s['title']} · {s['pasal']} · hlm. {s['pages']} · skor {s['score']:.3f}"):
                st.markdown(f"**{s['category']}** · Tahun {s['year']} · `{s['file']}` · `{s['chunk_id']}`")
                st.write(s["text"])

    for m in st.session_state["chat"]:
        with st.chat_message(m["role"], avatar="🧑‍⚖️" if m["role"] == "assistant" else None):
            st.markdown(m["content"]) if m["role"] == "user" else render_assistant(m)

    if prompt := st.chat_input("Tanyakan, mis. “apa hak pekerja yang di-PHK?”"):
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🧑‍⚖️"):
            msg = {"role": "assistant", "content": "", "sources": [], "verdict": None}
            try:
                hits = retrieve_for_rag(prompt)
            except Exception as e:
                hits = []
                msg["content"] = f"Pencarian gagal: {e}"
            if not hits:
                msg["content"] = msg["content"] or REFUSAL        # LLM tidak dipanggil bila retrieval kosong
            else:
                msg["sources"] = [source_record(engine, r) for r in hits]
                if not api_key:
                    msg["content"] = evidence_only_text(len(hits), f"{PROVIDERS[provider]['env']} belum diatur")
                else:
                    try:
                        text = st.write_stream(stream_llm(provider, api_key, llm_model, prompt,
                                                          build_context(msg["sources"])))
                        msg["content"] = text
                        msg["verdict"] = validate_answer(text, [s["text"] for s in msg["sources"]])
                    except Exception as e:
                        msg["content"] = (f"**LLM gagal dipanggil** ({type(e).__name__}: {e}). "
                                          + evidence_only_text(len(hits), "terjadi kesalahan API"))
        st.session_state["chat"].append(msg)
        st.rerun()

# ===========================================================================
# TAB 3 — Sitasi
# ===========================================================================
if nav == "Sitasi":
    saved = st.session_state["saved"]
    if not saved:
        st.markdown(ui.empty("📌", "Belum ada sitasi", "Klik tombol “Simpan” pada hasil pencarian untuk mengumpulkan "
                                                      "kutipan dan daftar pustaka."), unsafe_allow_html=True)
    else:
        md = "\n\n".join(f"{i}. {v['cite']}\n   > {v['text'][:300]}…" for i, v in enumerate(saved.values(), 1))
        txt = "\n".join(f"{i}. {v['cite']}" for i, v in enumerate(saved.values(), 1))
        d1, d2, d3 = st.columns([1, 1, 4])
        d1.download_button("⬇️ Markdown", md, "sitasi.md", "text/markdown", width="stretch")
        d2.download_button("⬇️ Teks", txt, "daftar_pustaka.txt", "text/plain", width="stretch")
        if d3.button("Hapus semua"):
            st.session_state["saved"] = {}
            st.rerun()
        for cid, v in list(saved.items()):
            with st.container(border=True):
                st.markdown(f"`{cid}`")
                st.write(v["cite"])
                st.caption(v["text"][:280] + "…")
                st.button("Hapus", key=f"del_{cid}", on_click=toggle_save, args=(cid, v["cite"], v["text"]))

# ===========================================================================
# TAB 4 — Lab
# ===========================================================================
if nav == "Lab":
    lab = st.tabs(["Boolean & Index", "TF-IDF", "BM25", "LSA"])
    with lab[0]:
        st.caption('Operator huruf besar: AND, OR, NOT (juga DAN/ATAU/TIDAK), tanda kurung, dan "frasa dua kata".')
        lookup = st.text_input("Jelajahi inverted index (term setelah stemming; bigram: kata1_kata2)", "upah")
        term = lookup.strip().lower()
        if term:
            tried = engine.query_tokens(term)
            if term not in engine.index.index and tried:
                term = tried[0]
            if term in engine.index.index:
                e = engine.index.index[term]
                a, b_, c_ = st.columns(3)
                a.metric("Term", term)
                b_.metric("DF", e["df"])
                c_.metric("IDF (smooth)", f"{engine.idf(term):.4f}")
                st.dataframe(pd.DataFrame([{"chunk_id": engine.chunks[d]["chunk_id"], "TF": len(p),
                                            "posisi": ", ".join(map(str, p[:15])) + (" …" if len(p) > 15 else "")}
                                           for d, p in list(e["postings"].items())[:50]]),
                             width="stretch", hide_index=True)
            else:
                st.info("Term tidak ada di index.")
        top_terms = sorted(engine.index.index.items(), key=lambda x: -x[1]["df"])[:20]
        st.markdown("**Term dengan DF tertinggi**")
        st.bar_chart(pd.DataFrame({"DF": [e["df"] for _, e in top_terms]}, index=[t for t, _ in top_terms]),
                     color="#E5484D")
    with lab[1]:
        tq = st.text_input("Kueri TF-IDF", "hak pekerja atas upah", key="lab_tq")
        if tq.strip():
            comp = {m: [engine.chunks[r["idx"]]["chunk_id"] for r in engine.search_tfidf(tq, m, True, True, 8)]
                    for m in ("raw", "log")}
            n = max((len(v) for v in comp.values()), default=0)
            st.dataframe(pd.DataFrame({f"{m}-tf": v + [""] * (n - len(v)) for m, v in comp.items()},
                                      index=range(1, n + 1)), width="stretch")
            st.caption("Bandingkan peringkat raw-tf vs log-tf (1 + log10 tf) dengan normalisasi cosine.")
    with lab[2]:
        bq = st.text_input("Kueri BM25", "perlindungan konsumen barang", key="lab_bq")
        st.caption("skor = Σ IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))")
        if bq.strip():
            grid = {f"k1={kk}, b={bb}": [engine.chunks[r["idx"]]["chunk_id"] for r in engine.search_bm25(bq, kk, bb, 5)]
                    for kk, bb in [(0.5, 0.25), (1.2, 0.75), (1.5, 0.75), (2.0, 0.9), (1.5, 0.0)]}
            n = max((len(v) for v in grid.values()), default=0)
            st.dataframe(pd.DataFrame({k: v + [""] * (n - len(v)) for k, v in grid.items()},
                                      index=range(1, n + 1)), width="stretch")
    with lab[3]:
        if n_chunks < 3:
            st.warning("Korpus terlalu kecil untuk LSA.")
        else:
            lsa = engine.get_lsa(LSA_K)
            st.metric("Varians terjelaskan", f"{lsa.explained * 100:.1f}%", help=f"{lsa.k} komponen laten")
            for i, topic in enumerate(lsa.topics(6, 8), start=1):
                st.markdown(f"**Topik {i}:** " + ", ".join(f"`{t.replace('_', ' ')}`" for t, _ in topic))

# ===========================================================================
# TAB 5 — Evaluasi (qrels manual, sama dengan template notebook bagian 8)
# ===========================================================================
def precision_recall_at_k(retrieved_ids, relevant_ids, k=5):
    """Sama dengan notebook: P@k = |top-k ∩ relevan| / |top-k|, R@k = |top-k ∩ relevan| / |relevan|."""
    retrieved, relevant = set(retrieved_ids[:k]), set(relevant_ids)
    if not retrieved:
        return 0.0, 0.0
    hit = len(retrieved & relevant)
    return hit / len(retrieved), (hit / len(relevant) if relevant else float("nan"))


if nav == "Evaluasi":
    st.markdown("Ukur retrieval pada **tingkat dokumen** dengan *relevance judgments* (qrels) yang kamu nilai manual: "
                "tiap kueri diberi daftar `doc_id` peraturan yang benar-benar menjawabnya. "
                "**Tanpa qrels tidak ada metrik** — tabel di bawah sengaja kosong.")
    if engine.corpus.config.get("source") == "sample":
        st.warning("Mode demo: korpus contoh tidak boleh dipakai untuk melaporkan evaluasi.", icon="🧪")
    with st.expander("📋 Daftar doc_id yang tersedia (salin ke kolom qrels)"):
        st.dataframe(pd.DataFrame([{"doc_id": d["doc_id"], "judul": d["title"], "kategori": d["category"],
                                    "tahun": d.get("year")} for d in engine.corpus.docs]),
                     width="stretch", hide_index=True)
    qrels_df = st.data_editor(
        pd.DataFrame({"kueri": pd.Series([], dtype="str"), "doc_id_relevan": pd.Series([], dtype="str")}),
        num_rows="dynamic", width="stretch", key="qrels_editor",
        column_config={"kueri": st.column_config.TextColumn("Kueri", width="large"),
                       "doc_id_relevan": st.column_config.TextColumn("doc_id relevan (pisahkan dengan koma)", width="large")})
    ec1, ec2 = st.columns([3, 1])
    eval_methods = ec1.multiselect("Metode", ["BM25", "TF-IDF", "LSA", "Hybrid", "Embedding"],
                                   default=["BM25", "TF-IDF", "Hybrid"])
    k_eval = ec2.slider("K (dokumen)", 1, 10, 5)

    rows_q = []
    for _, t in qrels_df.iterrows():
        qq = str(t.get("kueri") or "").strip()
        ids = [x.strip() for x in str(t.get("doc_id_relevan") or "").split(",") if x.strip()]
        if qq and ids:
            rows_q.append((qq, ids))
    unknown = sorted({i for _, ids in rows_q for i in ids if i not in doc_by_id})
    if unknown:
        st.warning(f"doc_id tidak ada di korpus: {', '.join(unknown[:8])}")
    if not rows_q:
        st.markdown(ui.empty("🧾", "QRELS belum diisi", "Tambahkan baris: kueri + doc_id relevan. "
                                                       "Disarankan 30–50 kueri agar hasilnya bermakna."),
                    unsafe_allow_html=True)
    elif st.button("▶️ Jalankan evaluasi", type="primary", disabled=not eval_methods):
        base_p = {"k1": 1.5, "b": 0.75, "tf_mode": "log", "norm": True, "alpha": 0.5, "fusion": "rrf", "hy_src": "embedding"}
        rows, detail = [], []
        prog = st.progress(0.0)
        for mi, m in enumerate(eval_methods):
            ps, rs, rr, hits = [], [], [], 0
            for qq, rel in rows_q:
                try:
                    res = run_method(m, qq, max(k_eval * 6, 30), base_p)[0]
                except Exception:
                    res = []
                ranked = list(dict.fromkeys(engine.chunks[r["idx"]]["doc_id"] for r in res))  # chunk -> dokumen unik
                p, r_ = precision_recall_at_k(ranked, rel, k_eval)
                first = next((i for i, d in enumerate(ranked, 1) if d in rel), None)
                ps.append(p); rs.append(r_); rr.append(1.0 / first if first else 0.0); hits += first is not None and first <= k_eval
                detail.append({"metode": m, "kueri": qq, "P@k": round(p, 3), "R@k": round(r_, 3),
                               "rank relevan pertama": first or "-", "top dokumen": ", ".join(ranked[:k_eval])})
            n = len(rows_q)
            rows.append({"Metode": m, f"Precision@{k_eval}": sum(ps) / n, f"Recall@{k_eval}": sum(rs) / n,
                         "MRR": sum(rr) / n, f"Hit@{k_eval}": hits / n})
            prog.progress((mi + 1) / len(eval_methods))
        prog.empty()
        out = pd.DataFrame(rows).set_index("Metode")
        st.dataframe(out.style.format("{:.3f}").background_gradient(cmap="Reds"), width="stretch")
        st.bar_chart(out, color=["#E5484D", "#F5B971", "#8B93AD", "#3DD68C"])
        with st.expander("Rincian per kueri"):
            st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
        st.download_button("⬇️ Unduh hasil (CSV)", out.to_csv().encode("utf-8"), "evaluasi_retrieval.csv", "text/csv")
        st.caption(f"{len(rows_q)} kueri · hasil hanya berlaku untuk korpus dan qrels ini. "
                   "Evaluasi ini hanya menilai retrieval, bukan kualitas jawaban LLM (faithfulness/correctness).")

# ===========================================================================
# TAB 6 — Korpus
# ===========================================================================
if nav == "Korpus":
    dtm, vocab, doc_ids = engine.doc_dtm()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dokumen", len(engine.corpus.docs))
    c2.metric("Chunk", n_chunks)
    c3.metric("Vocabulary (DTM)", f"{len(vocab):,}")
    c4.metric("Sparsity DTM", f"{sparsity(dtm) * 100:.2f}%")
    st.markdown("**Distribusi kategori**")
    st.bar_chart(pd.Series([d["category"] for d in engine.corpus.docs]).value_counts(), color="#E5484D")

    st.markdown("**Demo pipeline preprocessing**")
    demo = st.text_area("Teks contoh", "Setiap pekerja berhak memperoleh upah yang layak bagi kemanusiaan.",
                        height=80, label_visibility="collapsed")
    folded = case_fold(demo)
    toks = tokenize(folded)
    nostop = remove_stopwords(toks)
    stemmed = stem_tokens(nostop)
    st.dataframe(pd.DataFrame({
        "Tahap": ["1. Case folding", "2. Tokenisasi", "3. Stopword removal", "4. Stemming", "5. BoW (uni+bigram)"],
        "Hasil": [folded, " | ".join(toks), " | ".join(nostop), " | ".join(stemmed),
                  " | ".join(make_terms(stemmed, True))]}), width="stretch", hide_index=True)

    labels = [doc_by_id[i]["doc_id"] for i in doc_ids]
    n_terms = st.slider("Jumlah term pada DTM / incidence", 5, 40, 15)
    with st.expander("Document-Term Matrix (DF tertinggi)"):
        st.dataframe(dtm_dataframe(dtm, vocab, labels, n_terms, 20), width="stretch")
        st.caption(f"DTM penuh: {dtm.shape[0]} × {dtm.shape[1]} · non-zero {dtm.nnz:,} · sparsity {sparsity(dtm) * 100:.3f}%")
    with st.expander("Tabel incidence (1 = ada, 0 = tidak ada)"):
        st.dataframe(incidence_dataframe(dtm, vocab, labels, n_terms, 20), width="stretch")
    with st.expander("Metadata JSON"):
        st.json(engine.corpus.docs[:5], expanded=False)
        st.download_button("⬇️ Unduh metadata.json", json.dumps(engine.corpus.docs, ensure_ascii=False, indent=2),
                           "metadata.json", "application/json")
        if st.button("💾 Simpan ke data/processed/metadata.json"):
            st.success(f"Tersimpan: {save_metadata_json(engine.corpus, 'data/processed/metadata.json')}")

st.markdown("---")
st.caption("Dataset: Kaggle · hermansugiharto/law-indonesia · Inverted index, TF-IDF, BM25, dan Boolean "
           "diimplementasikan dari nol. Jawaban RAG bukan nasihat hukum; verifikasi ke sumber resmi.")
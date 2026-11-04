"""
preprocessing.py
================
Modul Ingestion Dataset & Preprocessing (Soal 1 - Sub-CPMK 10.1.1)

Isi modul:
  1. Ingestion folder dataset Kaggle `law-indonesia` (UU, PP, UUD, UUDRT, dst.)
  2. Pembacaan PDF/TXT + pembersihan header/footer dokumen hukum
  3. Case folding, tokenisasi, stopword removal, stemming (Sastrawi / fallback)
  4. Chunking teks (ukuran & overlap dalam jumlah kata)
  5. Bag-of-Words (unigram + bigram), Document-Term Matrix (sparse) + sparsity
  6. Tabel incidence (ada / tidak ada) dan metadata JSON
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
KNOWN_CATEGORIES: Dict[str, str] = {
    "UU": "Undang-Undang",
    "PP": "Peraturan Pemerintah",
    "UUD": "Undang-Undang Dasar",
    "UUDRT": "Undang-Undang Darurat",
}

STOPWORDS = set(
    """
    yang dan di ke dari dalam untuk pada dengan atau ini itu adalah ialah
    sebagai oleh karena akan dapat dalam telah sudah belum para serta juga
    tidak bukan hanya saja pun lah kah tah nya mereka kami kita saya anda
    ia dia beliau kepada terhadap antara atas bawah sebuah suatu setiap
    tersebut tersebutlah yaitu yakni yaitu bahwa jika apabila bila maka
    sehingga agar supaya namun tetapi tapi sedangkan sementara setelah
    sebelum selama sejak hingga sampai demi guna bagi lagi lain lainnya
    masing seperti sama sesuai menurut melalui tentang mengenai antar
    dimaksud dimaksudkan merupakan menjadi dilakukan melakukan ada
    adanya oleh olehnya pula amat sangat lebih paling cukup sekali
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------
_STEMMER = None
_STEM_CACHE: Dict[str, str] = {}


def _load_sastrawi():
    """Muat stemmer Sastrawi sekali saja; False jika tidak terpasang."""
    global _STEMMER
    if _STEMMER is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

            _STEMMER = StemmerFactory().create_stemmer()
        except Exception:
            _STEMMER = False
    return _STEMMER


_PREFIXES = sorted(
    ["meng", "meny", "men", "mem", "me", "peng", "peny", "pen", "pem", "pe",
     "ber", "bel", "be", "ter", "te", "di", "ke", "se"],
    key=len,
    reverse=True,
)


def fallback_stem(word: str) -> str:
    """Stemmer aturan sederhana (dipakai jika Sastrawi tidak tersedia)."""
    for suf in ("lah", "kah", "tah", "pun", "nya", "ku", "mu"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            word = word[: -len(suf)]
            break
    for suf in ("kan", "an", "i"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            word = word[: -len(suf)]
            break
    for pre in _PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 4:
            word = word[len(pre):]
            break
    return word


def stem_word(word: str) -> str:
    if word in _STEM_CACHE:
        return _STEM_CACHE[word]
    if word.isdigit() or len(word) <= 3:
        result = word
    else:
        stemmer = _load_sastrawi()
        result = stemmer.stem(word) if stemmer else fallback_stem(word)
    _STEM_CACHE[word] = result
    return result


def stemmer_name() -> str:
    return "Sastrawi" if _load_sastrawi() else "Fallback (aturan sederhana)"


# ---------------------------------------------------------------------------
# Tahapan preprocessing (dipisah agar bisa didemokan satu per satu di UI)
# ---------------------------------------------------------------------------
def case_fold(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 1]


def remove_stopwords(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def stem_tokens(tokens: Sequence[str]) -> List[str]:
    return [stem_word(t) for t in tokens]


def preprocess_text(text: str, stem: bool = True, remove_stop: bool = True) -> List[str]:
    """Pipeline lengkap: case folding -> tokenisasi -> stopword -> stemming."""
    tokens = tokenize(case_fold(text))
    if remove_stop:
        tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return tokens


def make_terms(tokens: Sequence[str], use_bigram: bool = True) -> List[str]:
    """Bag-of-Words: unigram + (opsional) bigram berformat 'kata1_kata2'."""
    terms = list(tokens)
    if use_bigram:
        terms += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return terms


# ---------------------------------------------------------------------------
# Pembersihan teks PDF hukum
# ---------------------------------------------------------------------------
_PAGE_NUM_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_NOISE_LINE_RES = [
    re.compile(r"^\s*halaman\s+\d+(\s+dari\s+\d+)?\s*$", re.I),
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I),
    re.compile(r"(www\.|https?://|jdih\.)", re.I),
    re.compile(r"^\s*sk\s+no\s*[\d\s\w]*$", re.I),
    re.compile(r"^\s*presiden\s+republik\s+indonesia\s*$", re.I),
    re.compile(r"^\s*(salinan|lembaran negara)\s*$", re.I),
]


def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def detect_repeating_lines(pages: Sequence[str], ratio: float = 0.4, max_len: int = 90) -> set:
    """Deteksi header/footer: baris pendek yang muncul di >= `ratio` halaman."""
    n = len(pages)
    if n < 3:
        return set()
    counter: Counter = Counter()
    for page in pages:
        counter.update({_norm_line(l) for l in page.splitlines() if l.strip()})
    return {l for l, c in counter.items() if c / n >= ratio and len(l) <= max_len}


def clean_pages(pages: Sequence[str]) -> List[str]:
    """Bersihkan setiap halaman: header/footer, nomor halaman, hyphenation."""
    repeating = detect_repeating_lines(pages)
    cleaned: List[str] = []
    for page in pages:
        kept = []
        for line in page.splitlines():
            if not line.strip():
                continue
            if _PAGE_NUM_RE.match(line):
                continue
            if any(rx.search(line) for rx in _NOISE_LINE_RES):
                continue
            if _norm_line(line) in repeating:
                continue
            kept.append(line.strip())
        text = "\n".join(kept)
        text = re.sub(r"-\n(\w)", r"\1", text)      # sambung kata terpotong
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_pages(pages_clean: Sequence[str], chunk_size: int = 100, overlap: int = 20) -> List[dict]:
    """
    Pecah dokumen menjadi chunk berjendela geser (sliding window) berbasis kata.
    Setiap chunk mencatat halaman awal & akhir.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap harus lebih kecil dari chunk_size")
    words: List[Tuple[str, int]] = []
    for pno, txt in enumerate(pages_clean, start=1):
        words.extend((w, pno) for w in txt.split())
    step = max(1, chunk_size - overlap)
    chunks: List[dict] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        if chunks and len(window) <= overlap:
            break  # seluruh isi window sudah tercakup chunk sebelumnya
        chunks.append(
            {
                "text": " ".join(w for w, _ in window),
                "page_start": window[0][1],
                "page_end": window[-1][1],
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Metadata dokumen
# ---------------------------------------------------------------------------
_TITLE_RE = re.compile(
    r"(UNDANG[\s-]*UNDANG(?:\s+DASAR)?(?:\s+DARURAT)?|PERATURAN\s+PEMERINTAH(?:\s+PENGGANTI\s+UNDANG[\s-]*UNDANG)?)"
    r"\s+(?:REPUBLIK\s+INDONESIA\s+)?NOMOR\s+(\d+[A-Za-z]?)\s+TAHUN\s+(\d{4})\s+TENTANG\s+(.{5,250}?)"
    r"(?=\s+(?:DENGAN\s+RAHMAT|PRESIDEN|MENIMBANG|BAB\s)|$)",
    re.I | re.S,
)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def extract_metadata(path: str, category: str, pages_raw: Sequence[str], root: str = "") -> dict:
    """Ekstrak metadata: doc_id, judul, penulis, tahun, bahasa, jumlah halaman."""
    stem_name = os.path.splitext(os.path.basename(path))[0]
    head = re.sub(r"\s+", " ", " ".join(pages_raw[:2]))[:2500]
    nomor, year, title = "", None, None
    m = _TITLE_RE.search(head)
    if m:
        jenis, nomor, year, topik = m.group(1), m.group(2), m.group(3), m.group(4)
        jenis = re.sub(r"\s+", " ", jenis).title()
        title = f"{jenis} Nomor {nomor} Tahun {year} tentang {topik.strip().title()}"
    if year is None:
        ym = re.search(r"(?:19|20)\d{2}", stem_name)
        year = ym.group(0) if ym else None
    if title is None:
        title = re.sub(r"[_\-]+", " ", stem_name).strip().title()
    rel = os.path.relpath(path, root) if root else path
    return {
        "title": title,
        "author": f"Pemerintah Republik Indonesia ({KNOWN_CATEGORIES.get(category, category)})",
        "year": int(year) if year else None,
        "language": "id",
        "pages": len(pages_raw),
        "category": category,
        "nomor": nomor,
        "source_path": rel,
    }


# ---------------------------------------------------------------------------
# Ingestion dataset Kaggle
# ---------------------------------------------------------------------------
def discover_documents(root: str, exts: Tuple[str, ...] = (".pdf", ".txt")) -> List[dict]:
    """Telusuri folder dataset (UU/PP/UUD/UUDRT/...). Kategori = nama folder dikenal."""
    items: List[dict] = []
    if not root or not os.path.isdir(root):
        return items
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(exts):
                continue
            full = os.path.join(dirpath, fn)
            rel_parts = os.path.relpath(full, root).split(os.sep)[:-1]
            category = None
            for part in rel_parts:
                if part.upper() in KNOWN_CATEGORIES:
                    category = part.upper()
                    break
            if category is None:
                category = rel_parts[0].upper() if rel_parts else "LAIN"
            items.append({"path": full, "category": category})
    items.sort(key=lambda x: x["path"])
    return items


def _balanced_limit(items: List[dict], max_docs: int) -> List[dict]:
    """Ambil max_docs dokumen secara bergiliran antar kategori agar seimbang."""
    buckets: Dict[str, List[dict]] = {}
    for it in items:
        buckets.setdefault(it["category"], []).append(it)
    out: List[dict] = []
    i = 0
    while len(out) < max_docs and any(buckets.values()):
        for cat in list(buckets):
            if buckets[cat] and len(out) < max_docs:
                out.append(buckets[cat].pop(0))
        i += 1
    return out


def read_document_pages(path: str) -> List[str]:
    """Baca file PDF (pypdf) atau TXT menjadi daftar halaman."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(path)
            return [(p.extract_text() or "") for p in reader.pages]
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().split("\f")
    except Exception:
        return []
    return []


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
@dataclass
class Corpus:
    docs: List[dict] = field(default_factory=list)
    chunks: List[dict] = field(default_factory=list)
    doc_tokens: Dict[str, List[str]] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def doc_by_id(self) -> Dict[str, dict]:
        return {d["doc_id"]: d for d in self.docs}


def _process_document(corpus: Corpus, meta: dict, pages_raw: Sequence[str],
                      chunk_size: int, overlap: int, stem: bool) -> None:
    pages_clean = clean_pages(pages_raw)
    if not any(pages_clean):
        return
    chunks = chunk_pages(pages_clean, chunk_size, overlap)
    for no, ch in enumerate(chunks, start=1):
        ch["chunk_id"] = f"{meta['doc_id']}#{no:03d}"
        ch["chunk_no"] = no
        ch["doc_id"] = meta["doc_id"]
        ch["tokens"] = preprocess_text(ch["text"], stem=stem)
    meta["n_chunks"] = len(chunks)
    corpus.docs.append(meta)
    corpus.chunks.extend(chunks)
    corpus.doc_tokens[meta["doc_id"]] = preprocess_text(" ".join(pages_clean), stem=stem)


def build_corpus(root: str, chunk_size: int = 100, overlap: int = 20,
                 max_docs: Optional[int] = None, stem: bool = True,
                 progress: Optional[Callable[[int, int], None]] = None) -> Corpus:
    """Bangun Corpus dari folder dataset Kaggle `law-indonesia`."""
    items = discover_documents(root)
    if max_docs:
        items = _balanced_limit(items, max_docs)
    corpus = Corpus(config={"chunk_size": chunk_size, "overlap": overlap, "stem": stem,
                            "source": root})
    used_ids: Dict[str, int] = {}
    for i, it in enumerate(items, start=1):
        pages = read_document_pages(it["path"])
        if not pages:
            continue
        meta = extract_metadata(it["path"], it["category"], pages, root)
        base = f"{it['category']}_{slugify(os.path.splitext(os.path.basename(it['path']))[0])}"
        used_ids[base] = used_ids.get(base, 0) + 1
        meta["doc_id"] = base if used_ids[base] == 1 else f"{base}_{used_ids[base]}"
        _process_document(corpus, meta, pages, chunk_size, overlap, stem)
        if progress:
            progress(i, len(items))
    return corpus


# ---------------------------------------------------------------------------
# Korpus contoh (fallback jika dataset Kaggle belum diunduh)
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    ("UU", 2003, "Ketenagakerjaan (Contoh)",
     "Setiap pekerja berhak memperoleh upah yang layak bagi kemanusiaan. Pengusaha wajib membayar upah "
     "pekerja tepat waktu sesuai perjanjian kerja. Upah minimum ditetapkan oleh gubernur dengan "
     "memperhatikan kebutuhan hidup layak dan produktivitas. Pemutusan hubungan kerja hanya dapat "
     "dilakukan setelah perundingan antara pengusaha dan serikat pekerja. Pekerja yang di-PHK berhak "
     "atas uang pesangon dan uang penghargaan masa kerja."),
    ("UU", 1999, "Perlindungan Konsumen (Contoh)",
     "Konsumen berhak atas kenyamanan, keamanan, dan keselamatan dalam mengonsumsi barang dan jasa. "
     "Pelaku usaha dilarang memperdagangkan barang yang tidak sesuai dengan label dan mutu yang "
     "dijanjikan. Konsumen yang dirugikan dapat mengajukan gugatan ganti rugi melalui badan "
     "penyelesaian sengketa konsumen. Pelaku usaha wajib memberikan informasi yang benar, jelas, "
     "dan jujur mengenai kondisi barang."),
    ("UU", 2022, "Pelindungan Data Pribadi (Contoh)",
     "Data pribadi adalah data tentang orang perseorangan yang teridentifikasi. Pengendali data "
     "pribadi wajib menjaga kerahasiaan data dan mencegah akses tidak sah. Subjek data pribadi berhak "
     "meminta penghapusan data dan menarik persetujuan pemrosesan. Kebocoran data wajib "
     "diberitahukan kepada subjek data dalam waktu tertentu. Pelanggaran dikenai sanksi administratif "
     "dan pidana."),
    ("UU", 2009, "Perlindungan dan Pengelolaan Lingkungan Hidup (Contoh)",
     "Setiap orang berhak atas lingkungan hidup yang baik dan sehat. Usaha yang berdampak penting "
     "terhadap lingkungan wajib memiliki analisis mengenai dampak lingkungan sebelum izin diterbitkan. "
     "Pencemaran udara, air, dan tanah dilarang melampaui baku mutu. Penanggung jawab usaha wajib "
     "melakukan pemulihan lingkungan akibat limbah yang dihasilkan. Masyarakat berhak berperan "
     "dalam pengawasan lingkungan."),
    ("PP", 2019, "Pajak Penghasilan Usaha Kecil (Contoh)",
     "Wajib pajak orang pribadi dengan peredaran bruto tertentu dikenai pajak penghasilan final. "
     "Tarif pajak dihitung dari omzet usaha setiap bulan dan disetor melalui sistem elektronik. "
     "Pelaku usaha mikro, kecil, dan menengah memperoleh kemudahan pelaporan pajak. Keterlambatan "
     "penyetoran dikenai sanksi bunga administrasi. Pemerintah dapat memberikan insentif "
     "perpajakan pada kondisi tertentu."),
    ("UU", 1999, "Pemberantasan Tindak Pidana Korupsi (Contoh)",
     "Setiap orang yang secara melawan hukum memperkaya diri sendiri yang merugikan keuangan negara "
     "dipidana penjara dan denda. Pegawai negeri yang menerima suap atau gratifikasi wajib "
     "melaporkannya kepada komisi pemberantasan. Penyidikan, penuntutan, dan pemeriksaan perkara "
     "korupsi didahulukan dari perkara lain. Harta benda hasil korupsi dapat dirampas untuk negara."),
    ("UU", 1974, "Perkawinan (Contoh)",
     "Perkawinan adalah ikatan lahir batin antara seorang pria dan wanita sebagai suami istri. "
     "Perkawinan sah apabila dilakukan menurut hukum agama dan kepercayaannya serta dicatat menurut "
     "peraturan. Calon mempelai harus telah mencapai batas usia minimal. Harta bersama suami istri "
     "dibagi apabila terjadi perceraian. Perceraian hanya dapat dilakukan di depan sidang pengadilan."),
    ("UU", 2003, "Sistem Pendidikan Nasional (Contoh)",
     "Setiap warga negara berhak mendapat pendidikan yang bermutu. Pemerintah wajib menyelenggarakan "
     "pendidikan dasar tanpa memungut biaya. Guru dan dosen berhak memperoleh penghasilan, jaminan "
     "kesejahteraan sosial, serta perlindungan hukum dalam menjalankan tugas. Kurikulum disusun "
     "dengan memperhatikan peningkatan iman, akhlak mulia, dan potensi peserta didik. Satuan "
     "pendidikan wajib menjamin mutu melalui akreditasi."),
]


def get_sample_corpus(chunk_size: int = 100, overlap: int = 20, stem: bool = True) -> Corpus:
    """Korpus contoh kecil (BUKAN dokumen resmi) agar aplikasi bisa dicoba tanpa dataset."""
    corpus = Corpus(config={"chunk_size": chunk_size, "overlap": overlap, "stem": stem,
                            "source": "sample"})
    for i, (cat, year, title, text) in enumerate(SAMPLE_DOCS, start=1):
        meta = {
            "doc_id": f"{cat}_CONTOH_{i:02d}",
            "title": f"{KNOWN_CATEGORIES[cat]} Tahun {year} tentang {title}",
            "author": "Pemerintah Republik Indonesia (korpus contoh)",
            "year": year,
            "language": "id",
            "pages": 1,
            "category": cat,
            "nomor": "",
            "source_path": "sample",
        }
        _process_document(corpus, meta, [text], chunk_size, overlap, stem)
    return corpus


# ---------------------------------------------------------------------------
# DTM, sparsity, incidence, metadata JSON
# ---------------------------------------------------------------------------
def build_dtm(term_lists: Sequence[Sequence[str]], min_df: int = 1,
              max_features: Optional[int] = None) -> Tuple[sparse.csr_matrix, List[str]]:
    """Document-Term Matrix (baris = dokumen, kolom = term) dalam format sparse CSR."""
    df: Counter = Counter()
    for terms in term_lists:
        df.update(set(terms))
    vocab = [t for t, c in df.items() if c >= min_df]
    if max_features and len(vocab) > max_features:
        vocab = [t for t, _ in sorted(((t, df[t]) for t in vocab), key=lambda x: -x[1])[:max_features]]
    vocab = sorted(vocab)
    t2i = {t: i for i, t in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for r, terms in enumerate(term_lists):
        for t, c in Counter(terms).items():
            j = t2i.get(t)
            if j is not None:
                rows.append(r)
                cols.append(j)
                vals.append(c)
    dtm = sparse.csr_matrix((vals, (rows, cols)), shape=(len(term_lists), len(vocab)), dtype=np.int32)
    return dtm, vocab


def sparsity(dtm: sparse.spmatrix) -> float:
    """Sparsity = 1 - (sel tidak nol / total sel)."""
    total = dtm.shape[0] * dtm.shape[1]
    return 1.0 - (dtm.nnz / total) if total else 0.0


def dtm_dataframe(dtm: sparse.csr_matrix, vocab: Sequence[str], labels: Sequence[str],
                  max_terms: int = 20, max_docs: int = 15) -> pd.DataFrame:
    """Potongan DTM (term dengan DF tertinggi) sebagai DataFrame untuk ditampilkan."""
    df_counts = np.asarray((dtm > 0).sum(axis=0)).ravel()
    top = np.argsort(-df_counts)[:max_terms]
    sub = dtm[:max_docs][:, top].toarray()
    return pd.DataFrame(sub, index=list(labels)[:max_docs], columns=[vocab[j] for j in top])


def incidence_dataframe(dtm: sparse.csr_matrix, vocab: Sequence[str], labels: Sequence[str],
                        max_terms: int = 20, max_docs: int = 15) -> pd.DataFrame:
    """Tabel incidence biner: 1 = term ada di dokumen, 0 = tidak ada."""
    return (dtm_dataframe(dtm, vocab, labels, max_terms, max_docs) > 0).astype(int)


def save_metadata_json(corpus: Corpus, path: str) -> str:
    """Simpan metadata seluruh dokumen ke berkas JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus.docs, f, ensure_ascii=False, indent=2)
    return path

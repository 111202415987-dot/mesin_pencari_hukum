"""
app.py
======
Prototipe Antarmuka Streamlit (Soal 3d) - Mesin Pencari Mini Hukum Indonesia
Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import os

import pandas as pd
import streamlit as st

from preprocessing import (KNOWN_CATEGORIES, build_corpus, case_fold, dtm_dataframe,
                           get_sample_corpus, incidence_dataframe, make_terms,
                           remove_stopwords, save_metadata_json, sparsity,
                           stem_tokens, stemmer_name, tokenize)
from retrieval_engine import (DEFAULT_EMBED_MODEL, SearchEngine, ast_to_string,
                              make_snippet)

st.set_page_config(page_title="Mesin Pencari Hukum Indonesia", page_icon="⚖️", layout="wide")

# ---------------------------------------------------------------------------
# Tema: Crimson #8B0000 | Cream #FFF8EC | Teks #1A1A1A
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] { color:#1A1A1A; }
.hero { background:linear-gradient(135deg,#8B0000 0%,#5E0000 100%); color:#FFF8EC;
        padding:1.4rem 1.8rem; border-radius:14px; margin-bottom:1.1rem;
        box-shadow:0 4px 14px rgba(139,0,0,.25); }
.hero h1 { color:#FFF8EC; margin:0; font-size:1.9rem; }
.hero p  { margin:.3rem 0 0 0; color:#F6E3C3; font-size:.95rem; }
.stTabs [data-baseweb="tab-list"] { gap:6px; overflow-x:auto; border-bottom:2px solid #8B0000; }
.stTabs [data-baseweb="tab"] { background:#F3E6CF; border-radius:10px 10px 0 0; padding:8px 16px;
        color:#1A1A1A; font-weight:600; white-space:nowrap; }
.stTabs [aria-selected="true"] { background:#8B0000 !important; color:#FFF8EC !important; }
.result-card { background:#FFFFFF; border:1px solid #E6D3B0; border-left:6px solid #8B0000;
        border-radius:10px; padding:.8rem 1rem; margin:.7rem 0 .2rem 0; }
.result-title { font-weight:700; font-size:1.02rem; color:#8B0000; }
.result-meta  { font-size:.8rem; color:#555; margin:.15rem 0 .45rem 0; }
.badge { display:inline-block; background:#8B0000; color:#FFF8EC; border-radius:20px;
         padding:1px 10px; font-size:.75rem; margin-right:6px; font-weight:600; }
.badge.soft { background:#F3E6CF; color:#1A1A1A; border:1px solid #D9C29A; }
.result-snippet { font-size:.92rem; line-height:1.5; color:#1A1A1A; }
mark { background:#F6D58E; color:#1A1A1A; padding:0 2px; border-radius:3px; }
section[data-testid="stSidebar"] { border-right:3px solid #8B0000; }
div.stButton > button { background:#8B0000; color:#FFF8EC; border:none; border-radius:8px; font-weight:600; }
div.stButton > button:hover { background:#6E0000; color:#FFFFFF; }
div[data-testid="stMetric"] { background:#F3E6CF; border-radius:10px; padding:.6rem .9rem;
        border-left:5px solid #8B0000; }
</style>
<div class="hero">
  <h1>⚖️ Mesin Pencari Mini — Peraturan Perundang-Undangan Indonesia</h1>
  <p>UTS Sistem Temu Kembali Informasi (STKI) · UDINUS &nbsp;|&nbsp; Boolean · Inverted Index ·
     TF-IDF · BM25 · LSA · Embedding · Hybrid</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: konfigurasi
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    data_dir = st.text_input("Folder dataset Kaggle (law-indonesia)", value="data/law-indonesia",
                             help="Folder berisi sub-folder UU, PP, UUD, UUDRT.")
    has_data = os.path.isdir(data_dir)
    use_sample = st.checkbox("Pakai korpus contoh (tanpa dataset)", value=not has_data)
    if not has_data and not use_sample:
        st.warning("Folder dataset tidak ditemukan — otomatis memakai korpus contoh.")
        use_sample = True
    max_docs = st.number_input("Batas jumlah dokumen (0 = semua)", 0, 5000, 150, step=10,
                               disabled=use_sample)
    chunk_size = st.slider("Ukuran chunk (kata)", 40, 400, 100, step=10)
    overlap = st.slider("Overlap chunk (kata)", 0, 100, 20, step=5)
    use_bigram = st.checkbox("Sertakan bigram", value=True)
    stem = st.checkbox("Stemming", value=True)
    st.caption(f"Stemmer aktif: **{stemmer_name()}**")
    top_k = st.slider("Jumlah hasil (Top-K)", 3, 30, 10)
    build_btn = st.button("🔄 Bangun / muat ulang indeks", width="stretch")


@st.cache_resource(show_spinner=False)
def load_engine(data_dir, use_sample, max_docs, chunk_size, overlap, use_bigram, stem):
    if use_sample:
        corpus = get_sample_corpus(chunk_size, overlap, stem)
    else:
        corpus = build_corpus(data_dir, chunk_size, overlap, max_docs or None, stem)
        if not corpus.chunks:
            corpus = get_sample_corpus(chunk_size, overlap, stem)
            corpus.config["fallback"] = True
    return SearchEngine(corpus, use_bigram=use_bigram)


params = (data_dir, use_sample, int(max_docs), chunk_size, overlap, use_bigram, stem)
if "engine" not in st.session_state or build_btn:
    with st.spinner("Membaca dokumen, preprocessing, dan membangun inverted index…"):
        st.session_state["engine"] = load_engine(*params)
        st.session_state["params"] = params
engine: SearchEngine = st.session_state["engine"]
if st.session_state.get("params") != params:
    st.info("Konfigurasi berubah — klik **Bangun / muat ulang indeks** di sidebar untuk menerapkan.")
if engine.corpus.config.get("fallback"):
    st.warning("Dataset tidak berisi PDF/TXT yang terbaca; memakai korpus contoh.")
if engine.corpus.config.get("source") == "sample":
    st.caption("ℹ️ Memakai **korpus contoh** (bukan dokumen resmi). Letakkan dataset Kaggle di "
               "`data/law-indonesia` untuk pencarian pada peraturan sesungguhnya.")

doc_by_id = engine.corpus.doc_by_id


# ---------------------------------------------------------------------------
# Komponen tampilan hasil
# ---------------------------------------------------------------------------
def citation(chunk: dict, doc: dict) -> str:
    year = doc.get("year") or "t.t."
    pages = (f"hlm. {chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
             else f"hlm. {chunk['page_start']}–{chunk['page_end']}")
    return (f"Pemerintah Republik Indonesia. ({year}). {doc['title']}. "
            f"[{doc['category']}] {chunk['chunk_id']}, {pages}. "
            f"Sumber data: Kaggle — hermansugiharto/law-indonesia.")


def render_results(results, query, key, extra_cols=False):
    if not results:
        st.info("Tidak ada dokumen yang cocok dengan kueri ini.")
        return
    st.markdown(f"**{len(results)} hasil ditampilkan**")
    for rank, r in enumerate(results, start=1):
        ch = engine.chunks[r["idx"]]
        doc = doc_by_id[ch["doc_id"]]
        badges = f'<span class="badge">Skor {r["score"]:.4f}</span>'
        if extra_cols:
            badges += (f'<span class="badge soft">leksikal {r.get("lex", 0):.3f}</span>'
                       f'<span class="badge soft">semantik {r.get("sem", 0):.3f}</span>')
        if r.get("matched"):
            shown = ", ".join(html.escape(m.replace("_", " ")) for m in r["matched"][:8])
            badges += f'<span class="badge soft">term cocok: {shown}</span>'
        card = (
            '<div class="result-card">'
            f'<div class="result-title">#{rank} · {html.escape(doc["title"])}</div>'
            f'<div class="result-meta">{badges}<br>'
            f'{html.escape(doc["category"])} · {doc.get("year") or "-"} · '
            f'hlm. {ch["page_start"]}–{ch["page_end"]} · <code>{html.escape(ch["chunk_id"])}</code></div>'
            f'<div class="result-snippet">{make_snippet(ch["text"], query, stem=engine.stem)}</div>'
            "</div>"
        )
        st.markdown(card, unsafe_allow_html=True)
        with st.expander("📎 Sitasi, metadata & teks lengkap"):
            st.markdown("**Sitasi**")
            st.code(citation(ch, doc), language=None)
            st.markdown("**Metadata dokumen**")
            st.json({k: v for k, v in doc.items()}, expanded=False)
            st.markdown("**Teks chunk**")
            st.write(ch["text"])


def query_box(label, default, key, placeholder=""):
    return st.text_input(label, value=default, key=key, placeholder=placeholder)


tabs = st.tabs([
    "📚 Dataset & Preprocessing",
    "🔎 Boolean & Inverted Index",
    "📐 TF-IDF",
    "🎯 BM25",
    "🧠 Semantic (LSA / Embedding)",
    "🔀 Hybrid",
])

# ===========================================================================
# TAB 1 — Dataset & Preprocessing
# ===========================================================================
with tabs[0]:
    dtm, vocab, doc_ids = engine.doc_dtm()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dokumen", len(engine.corpus.docs))
    c2.metric("Chunk", len(engine.chunks))
    c3.metric("Ukuran vocabulary (DTM)", f"{len(vocab):,}")
    c4.metric("Sparsity DTM", f"{sparsity(dtm) * 100:.2f}%")

    st.subheader("Distribusi kategori peraturan")
    cat_counts = pd.Series([d["category"] for d in engine.corpus.docs]).value_counts()
    st.bar_chart(cat_counts, color="#8B0000")

    st.subheader("Demo pipeline preprocessing")
    demo = st.text_area("Teks contoh", "Setiap pekerja berhak memperoleh upah yang layak bagi kemanusiaan.",
                        key="demo_text", height=80)
    folded = case_fold(demo)
    toks = tokenize(folded)
    nostop = remove_stopwords(toks)
    stemmed = stem_tokens(nostop)
    st.dataframe(pd.DataFrame({
        "Tahap": ["1. Case folding", "2. Tokenisasi", "3. Stopword removal",
                  "4. Stemming", "5. Bag-of-Words (uni+bigram)"],
        "Hasil": [folded, " | ".join(toks), " | ".join(nostop), " | ".join(stemmed),
                  " | ".join(make_terms(stemmed, True))],
    }), width="stretch", hide_index=True)

    st.subheader("Document-Term Matrix (potongan: term dengan DF tertinggi)")
    n_terms = st.slider("Jumlah term ditampilkan", 5, 40, 15, key="dtm_terms")
    labels = [doc_by_id[i]["doc_id"] for i in doc_ids]
    st.dataframe(dtm_dataframe(dtm, vocab, labels, n_terms, 20), width="stretch")
    st.caption(f"Ukuran DTM penuh: {dtm.shape[0]} dokumen × {dtm.shape[1]} term · "
               f"sel tidak nol: {dtm.nnz:,} · sparsity: {sparsity(dtm) * 100:.3f}%")

    st.subheader("Tabel incidence (1 = ada, 0 = tidak ada)")
    st.dataframe(incidence_dataframe(dtm, vocab, labels, n_terms, 20), width="stretch")

    st.subheader("Metadata JSON")
    st.json(engine.corpus.docs[:5], expanded=False)
    st.download_button("⬇️ Unduh metadata.json",
                       json.dumps(engine.corpus.docs, ensure_ascii=False, indent=2),
                       file_name="metadata.json", mime="application/json")
    if st.button("💾 Simpan ke data/processed/metadata.json"):
        st.success(f"Tersimpan: {save_metadata_json(engine.corpus, 'data/processed/metadata.json')}")

# ===========================================================================
# TAB 2 — Boolean & Inverted Index
# ===========================================================================
with tabs[1]:
    st.subheader("Boolean Search")
    st.caption('Operator huruf besar: **AND**, **OR**, **NOT** (juga DAN/ATAU/TIDAK). '
               'Gunakan tanda kurung dan **"frasa dua kata"**. Dua kata berurutan = AND.')
    bq = query_box("Kueri boolean", '(pekerja OR buruh) AND upah NOT pidana', "bool_q")
    if bq.strip():
        res, ast, n_match = engine.search_boolean(bq, top_k)
        st.code(f"Parse tree: {ast_to_string(ast)}\nChunk cocok: {n_match} dari {len(engine.chunks)}",
                language=None)
        render_results(res, bq, "bool")

    st.divider()
    st.subheader("Pencarian Frasa (positional index)")
    pq = query_box("Frasa (2 kata atau lebih)", "upah minimum", "phrase_q")
    if pq.strip():
        render_results(engine.search_phrase(pq, top_k), pq, "phrase")

    st.divider()
    st.subheader("Penjelajah Inverted Index")
    lookup = st.text_input("Cari term (bentuk setelah stemming, bigram: kata1_kata2)", value="upah",
                           key="idx_lookup")
    term = lookup.strip().lower()
    if term:
        stemmed_try = engine.query_tokens(term)
        if term not in engine.index.index and stemmed_try:
            term = stemmed_try[0]
        if term in engine.index.index:
            entry = engine.index.index[term]
            st.markdown(f"**Term:** `{term}` · **DF:** {entry['df']} · "
                        f"**IDF (smooth):** {engine.idf(term):.4f}")
            rows = [{"chunk_id": engine.chunks[d]["chunk_id"], "TF": len(p),
                     "posisi": ", ".join(map(str, p[:15])) + (" …" if len(p) > 15 else "")}
                    for d, p in list(entry["postings"].items())[:50]]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("Term tidak ada di index.")
    st.markdown("**Term dengan DF tertinggi**")
    top_terms = sorted(engine.index.index.items(), key=lambda x: -x[1]["df"])[:20]
    st.dataframe(pd.DataFrame([{"term": t, "DF": e["df"]} for t, e in top_terms]),
                 width="stretch", hide_index=True)

# ===========================================================================
# TAB 3 — TF-IDF
# ===========================================================================
with tabs[2]:
    st.subheader("TF-IDF + Cosine Similarity (dari nol)")
    tq = query_box("Kueri", "hak pekerja atas upah", "tfidf_q")
    c1, c2, c3 = st.columns(3)
    tf_mode = c1.radio("Skema TF", ["log", "raw"], horizontal=True, key="tf_mode",
                       help="log: 1+log10(tf) · raw: frekuensi mentah")
    norm = c2.checkbox("Normalisasi panjang dokumen (cosine)", True, key="tf_norm")
    smooth = c3.checkbox("IDF smooth", True, key="tf_smooth")
    if tq.strip():
        render_results(engine.search_tfidf(tq, tf_mode, norm, smooth, top_k), tq, "tfidf")
        with st.expander("🔬 Bandingkan raw-tf vs log-tf (peringkat)"):
            comp = {}
            for m in ("raw", "log"):
                comp[m] = [engine.chunks[r["idx"]]["chunk_id"]
                           for r in engine.search_tfidf(tq, m, norm, smooth, top_k)]
            n = max((len(v) for v in comp.values()), default=0)
            st.dataframe(pd.DataFrame({f"{m}-tf": v + [""] * (n - len(v)) for m, v in comp.items()},
                                      index=range(1, n + 1)), width="stretch")

# ===========================================================================
# TAB 4 — BM25
# ===========================================================================
with tabs[3]:
    st.subheader("BM25 (dari nol)")
    bmq = query_box("Kueri", "perlindungan konsumen barang", "bm25_q")
    c1, c2 = st.columns(2)
    k1 = c1.slider("k1 (saturasi TF)", 0.0, 3.0, 1.5, 0.1, key="bm_k1")
    b = c2.slider("b (normalisasi panjang)", 0.0, 1.0, 0.75, 0.05, key="bm_b")
    st.caption("Rumus: skor = Σ IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)), "
               "IDF = ln(1 + (N − df + 0.5)/(df + 0.5)).")
    if bmq.strip():
        render_results(engine.search_bm25(bmq, k1, b, top_k), bmq, "bm25")
        with st.expander("🔬 Uji sensitivitas parameter (k1, b)"):
            grid = {}
            for kk, bb in [(0.5, 0.25), (1.2, 0.75), (1.5, 0.75), (2.0, 0.9), (1.5, 0.0)]:
                grid[f"k1={kk}, b={bb}"] = [engine.chunks[r["idx"]]["chunk_id"]
                                            for r in engine.search_bm25(bmq, kk, bb, 5)]
            n = max((len(v) for v in grid.values()), default=0)
            st.dataframe(pd.DataFrame({k: v + [""] * (n - len(v)) for k, v in grid.items()},
                                      index=range(1, n + 1)), width="stretch")

# ===========================================================================
# TAB 5 — Semantic
# ===========================================================================
with tabs[4]:
    st.subheader("Pencarian Berdasarkan Makna")
    mode = st.radio("Metode", ["LSA (topic modeling)", "Sentence Embedding + Vector Store"],
                    horizontal=True, key="sem_mode")
    sq = query_box("Kueri", "gaji pekerja", "sem_q",
                   "contoh vocabulary mismatch: 'gaji' vs 'upah'")
    if mode.startswith("LSA"):
        max_k = max(2, min(300, min(len(engine.chunks), engine.index.vocab_size) - 1))
        k_lsa = st.slider("Jumlah komponen laten (k)", 2, max_k, min(50, max_k), key="lsa_k")
        if len(engine.chunks) < 3:
            st.warning("Korpus terlalu kecil untuk LSA.")
        else:
            lsa = engine.get_lsa(k_lsa)
            st.caption(f"Ragam (varians) terjelaskan oleh {lsa.k} komponen: {lsa.explained * 100:.1f}%")
            if sq.strip():
                render_results(engine.search_lsa(sq, k_lsa, top_k), sq, "lsa")
            with st.expander("🧩 Topik laten (term berbobot tertinggi per komponen)"):
                for i, topic in enumerate(lsa.topics(6, 8), start=1):
                    st.markdown(f"**Topik {i}:** " + ", ".join(f"`{t.replace('_', ' ')}`" for t, _ in topic))
    else:
        c1, c2 = st.columns(2)
        model_name = c1.text_input("Model sentence-transformers", DEFAULT_EMBED_MODEL, key="emb_model")
        backend = c2.selectbox("Vector store", ["faiss", "chroma", "numpy"], key="emb_backend")
        if st.button("🚀 Muat model & bangun vector store", key="emb_build"):
            with st.spinner("Menghitung embedding seluruh chunk (pertama kali mengunduh model)…"):
                try:
                    store = engine.build_semantic(model_name, backend)
                    st.success(f"Vector store siap · backend aktif: **{store.backend}** · "
                               f"{len(engine.chunks)} vektor.")
                    for note in store.notes:
                        st.info(note)
                except Exception as e:
                    st.error(f"Gagal membangun embedding: {e}")
        if engine.semantic is None:
            st.info("Klik tombol di atas untuk membangun vector store. Tab Hybrid otomatis memakai LSA "
                    "jika embedding belum tersedia.")
        elif sq.strip():
            render_results(engine.search_semantic(sq, top_k), sq, "emb")

# ===========================================================================
# TAB 6 — Hybrid
# ===========================================================================
with tabs[5]:
    st.subheader("Hybrid Retrieval: BM25 (leksikal) + Semantik")
    hq = query_box("Kueri", "gaji pekerja yang di-PHK", "hyb_q")
    c1, c2, c3 = st.columns(3)
    method = c1.radio("Metode fusi", ["weighted", "rrf"], horizontal=True, key="hyb_method",
                      help="weighted: α·leksikal + (1−α)·semantik (min-max) · rrf: Reciprocal Rank Fusion")
    alpha = c2.slider("α (bobot leksikal)", 0.0, 1.0, 0.5, 0.05, key="hyb_alpha",
                      disabled=(method == "rrf"))
    n_cand = c3.slider("Kandidat per metode", 10, 200, 50, 10, key="hyb_cand")
    source_options = ["embedding", "lsa"] if engine.semantic is not None else ["lsa"]
    source = st.radio("Sumber semantik", source_options, horizontal=True, key="hyb_src")
    if engine.semantic is None:
        st.caption("Embedding belum dibangun (tab Semantic) — memakai LSA sebagai komponen semantik.")
    if hq.strip():
        if len(engine.chunks) < 3:
            st.warning("Korpus terlalu kecil.")
        else:
            k_h = max(2, min(50, min(len(engine.chunks), engine.index.vocab_size) - 1))
            results, used = engine.search_hybrid(hq, alpha, method, source, n_cand, 1.5, 0.75, k_h, top_k)
            st.caption(f"Komponen semantik yang dipakai: **{used.upper()}**")
            render_results(results, hq, "hyb", extra_cols=True)
            with st.expander("🔬 Perbandingan peringkat antar metode"):
                lex = engine.search_bm25(hq, 1.5, 0.75, 5)
                sem = (engine.search_semantic(hq, 5) if used == "embedding"
                       else engine.search_lsa(hq, k_h, 5))
                cols = {"BM25": lex, used.upper(): sem, "Hybrid": results[:5]}
                n = max((len(v) for v in cols.values()), default=0)
                st.dataframe(pd.DataFrame(
                    {name: [engine.chunks[r["idx"]]["chunk_id"] for r in rs]
                     + [""] * (n - len(rs)) for name, rs in cols.items()},
                    index=range(1, n + 1)), width="stretch")

st.markdown("---")
st.caption("Dataset: Kaggle · hermansugiharto/law-indonesia · Seluruh komponen leksikal "
           "(inverted index, TF-IDF, BM25, boolean) diimplementasikan dari nol.")

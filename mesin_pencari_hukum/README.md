# ⚖️ HukumSearch — Chatbot RAG Hukum Indonesia (UTS STKI UDINUS)

Mesin pencari dan chatbot RAG untuk dataset Kaggle `hermansugiharto/law-indonesia`.
Komponen leksikal (inverted index, TF-IDF, BM25, Boolean) ditulis dari nol. Alur lengkap: [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app_rag.py        # sama dengan: streamlit run app.py
```
1. Ekstrak dataset ke `data/law-indonesia/` (berisi `UU`, `PP`, `UUD`, `UUDRT`; boleh dalam satu folder pembungkus) atau atur `LAW_DATA_DIR`.
2. Isi `GEMINI_API_KEY` (sidebar, env var, atau `.streamlit/secrets.toml`, lihat `secrets.toml.example`). Groq tersedia sebagai alternatif. Tanpa key, aplikasi berjalan sebagai retrieval-only.
3. Tanpa dataset aplikasi **tidak** memakai korpus contoh secara diam-diam; **Mode demo** harus dinyalakan manual dan diberi banner.

## Halaman
Cari · Tanya Hukum (RAG) · Sitasi · Lab · Evaluasi (qrels manual: Precision@k, Recall@k, MRR, Hit@k) · Korpus.

## Berkas
`app_rag.py` (pintu masuk) → `app.py` (UI) · `ui.py` (desain) · `rag_engine.py` (prompt, LLM, verifikasi sitasi) · `preprocessing.py` · `retrieval_engine.py`.

## Batasan
- Skor grounding adalah heuristik tumpang-tindih kata, bukan jaminan bebas halusinasi. Jawaban bukan nasihat hukum.
- Evaluasi hanya menilai retrieval; isi qrels sendiri (disarankan 30-50 kueri).
- Jangan commit API key.
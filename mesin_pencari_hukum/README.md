# ⚖️ Mesin Pencari Mini — Hukum Indonesia (UTS STKI UDINUS)

Mesin pencari dokumen peraturan perundang-undangan Indonesia dengan komponen leksikal
**ditulis dari nol** (tanpa scikit-learn) dan antarmuka **Streamlit** bertema Crimson–Cream.

## Struktur
```
mesin_pencari_hukum/
├── app.py               # Antarmuka Streamlit (6 tab)
├── preprocessing.py     # Ingestion, pembersihan PDF, stemming, chunking, BoW, DTM, incidence, metadata
├── retrieval_engine.py  # Inverted index, Boolean, TF-IDF, BM25, LSA, Embedding+FAISS/Chroma, Hybrid
├── requirements.txt
├── README.md
└── .streamlit/config.toml   # Tema warna
```

## Langkah menjalankan
1. **Python 3.10+**, lalu buat virtual environment (disarankan):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Unduh dataset** dari <https://www.kaggle.com/datasets/hermansugiharto/law-indonesia>,
   ekstrak sehingga strukturnya menjadi:
   ```
   data/law-indonesia/
   ├── UU/      *.pdf
   ├── PP/      *.pdf
   ├── UUD/     *.pdf
   └── UUDRT/   *.pdf
   ```
   (Atau lewat CLI: `kaggle datasets download -d hermansugiharto/law-indonesia -p data --unzip`
   lalu pastikan folder sub-kategori berada di `data/law-indonesia/`. Folder lain dapat diisi di sidebar.)
3. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```
4. Di sidebar: atur folder dataset, batas dokumen (mulai dari 100–200 agar cepat), ukuran/overlap
   chunk, bigram, stemming, lalu klik **Bangun / muat ulang indeks**.

> Tanpa dataset, aplikasi otomatis memakai **korpus contoh** (8 teks pendek buatan, bukan dokumen resmi)
> sehingga seluruh fitur tetap bisa dicoba.

## Fitur per tab
| Tab | Isi |
|---|---|
| 📚 Dataset & Preprocessing | Statistik korpus, demo pipeline, DTM + sparsity, tabel incidence, metadata JSON |
| 🔎 Boolean & Inverted Index | AND/OR/NOT + kurung, frasa 2 kata (positional), penjelajah posting list (DF, TF, posisi) |
| 📐 TF-IDF | raw-tf vs log-tf, normalisasi panjang dokumen, cosine similarity |
| 🎯 BM25 | Slider k1 dan b, tabel sensitivitas parameter |
| 🧠 Semantic | LSA (SVD, topik laten) dan Sentence Embedding + FAISS / ChromaDB |
| 🔀 Hybrid | Weighted-sum (min-max, α) dan Reciprocal Rank Fusion, perbandingan peringkat |

Setiap hasil menampilkan dokumen sumber, skor, cuplikan teks bersorot, dan expander **sitasi**.

## Catatan teknis
- Unit pencarian adalah **chunk** (jendela geser berbasis kata); metadata tingkat dokumen tersimpan terpisah.
- Operator boolean harus **huruf besar** agar tidak tertukar dengan kata biasa (`AND`, `OR`, `NOT`, atau `DAN`, `ATAU`, `TIDAK`).
- Model embedding default `paraphrase-multilingual-MiniLM-L12-v2` (mendukung bahasa Indonesia, memotong
  input ±128 token, sehingga ukuran chunk 100 kata dianjurkan). Model diunduh saat pertama kali dipakai;
  embedding disimpan di `.cache/embeddings`.
- Jika FAISS/ChromaDB gagal dimuat, vector store otomatis beralih ke NumPy.
- PDF hasil pindai (tanpa lapisan teks) tidak terbaca oleh `pypdf`; perlu OCR terlebih dahulu.
- Stemmer: Sastrawi; jika tidak terpasang, dipakai stemmer aturan sederhana sebagai cadangan.

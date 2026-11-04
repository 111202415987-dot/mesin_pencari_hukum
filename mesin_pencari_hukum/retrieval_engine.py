"""
retrieval_engine.py
===================
Mesin Pencari (Soal 2 & Soal 3)

Bagian leksikal ditulis dari nol (tanpa scikit-learn):
  * InvertedIndex  : term -> {dokumen: [posisi]}, DF, panjang dokumen
  * Boolean search : AND / OR / NOT, tanda kurung, pencarian frasa "dua kata"
  * TF-IDF         : raw-tf vs log-tf, normalisasi panjang dokumen, cosine similarity
  * BM25           : parameter k1 dan b dapat diubah
Bagian makna & hybrid:
  * LSA            : SVD terpotong (NumPy/SciPy) untuk mengatasi vocabulary mismatch
  * SemanticStore  : Sentence Embedding + vector store (FAISS / ChromaDB / NumPy)
  * Hybrid         : weighted-sum (min-max) dan Reciprocal Rank Fusion
"""
from __future__ import annotations

import hashlib
import heapq
import html
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from preprocessing import (Corpus, STOPWORDS, build_dtm, make_terms, preprocess_text)

DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ===========================================================================
# 1. INVERTED INDEX
# ===========================================================================
class InvertedIndex:
    """Inverted index positional: index[term] = {"df": int, "postings": {doc: [pos,...]}}."""

    def __init__(self) -> None:
        self.index: Dict[str, dict] = {}
        self.doc_len: List[int] = []
        self.n_docs = 0

    def add_document(self, doc_idx: int, tokens: Sequence[str], use_bigram: bool = True) -> None:
        positions: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(tokens):
            positions[t].append(i)
        if use_bigram:
            for i in range(len(tokens) - 1):
                positions[f"{tokens[i]}_{tokens[i + 1]}"].append(i)
        for term, pos in positions.items():
            entry = self.index.setdefault(term, {"df": 0, "postings": {}})
            entry["postings"][doc_idx] = pos
            entry["df"] += 1
        self.doc_len.append(len(tokens))
        self.n_docs = max(self.n_docs, doc_idx + 1)

    # -- akses dasar ------------------------------------------------------
    def df(self, term: str) -> int:
        return self.index[term]["df"] if term in self.index else 0

    def postings(self, term: str) -> Dict[int, List[int]]:
        return self.index[term]["postings"] if term in self.index else {}

    def tf(self, term: str, doc_idx: int) -> int:
        return len(self.postings(term).get(doc_idx, ()))

    @property
    def avgdl(self) -> float:
        return sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

    @property
    def vocab_size(self) -> int:
        return len(self.index)

    # -- pencarian frasa ----------------------------------------------------
    def phrase_docs(self, terms: Sequence[str]) -> Dict[int, List[int]]:
        """Dokumen yang memuat `terms` berurutan (posisi berdampingan). Return {doc: [posisi awal]}."""
        if not terms or any(t not in self.index for t in terms):
            return {}
        docs = set(self.postings(terms[0]))
        for t in terms[1:]:
            docs &= set(self.postings(t))
        result: Dict[int, List[int]] = {}
        for d in docs:
            starts = set(self.postings(terms[0])[d])
            for k, t in enumerate(terms[1:], start=1):
                nxt = set(self.postings(t)[d])
                starts = {s for s in starts if s + k in nxt}
                if not starts:
                    break
            if starts:
                result[d] = sorted(starts)
        return result


# ===========================================================================
# 2. BOOLEAN PARSER
# ===========================================================================
class BooleanParser:
    """
    Tata bahasa:  or := and (OR and)* ; and := not ((AND)? not)* ; not := NOT not | atom
                  atom := ( or ) | "frasa" | kata
    Operator ditulis HURUF BESAR: AND/OR/NOT (atau DAN/ATAU/TIDAK). Dua term berurutan = AND.
    """
    AND = ("AND", "DAN")
    OR = ("OR", "ATAU")
    NOT = ("NOT", "TIDAK")

    def __init__(self, normalizer) -> None:
        self.normalize = normalizer
        self.toks: List[str] = []
        self.pos = 0

    def parse(self, query: str):
        self.toks = re.findall(r'"[^"]*"|\(|\)|[^\s()"]+', query)
        self.pos = 0
        return self._or()

    def _peek(self) -> Optional[str]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    @staticmethod
    def _combine(op, a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (op, a, b)

    def _or(self):
        left = self._and()
        while self._peek() in self.OR:
            self.pos += 1
            left = self._combine("OR", left, self._and())
        return left

    def _and(self):
        left = self._not()
        while True:
            p = self._peek()
            if p is None or p == ")" or p in self.OR:
                break
            if p in self.AND:
                self.pos += 1
            left = self._combine("AND", left, self._not())
        return left

    def _not(self):
        if self._peek() in self.NOT:
            self.pos += 1
            inner = self._not()
            return None if inner is None else ("NOT", inner)
        return self._atom()

    def _atom(self):
        p = self._peek()
        if p is None:
            return None
        self.pos += 1
        if p == "(":
            node = self._or()
            if self._peek() == ")":
                self.pos += 1
            return node
        if p == ")":
            return None
        if p.startswith('"'):
            terms = self.normalize(p.strip('"'))
            if not terms:
                return None
            return ("TERM", terms[0]) if len(terms) == 1 else ("PHRASE", terms)
        terms = self.normalize(p)
        return ("TERM", terms[0]) if terms else None


def ast_to_string(ast) -> str:
    if ast is None:
        return "(kosong)"
    k = ast[0]
    if k == "TERM":
        return ast[1]
    if k == "PHRASE":
        return '"' + " ".join(ast[1]) + '"'
    if k == "NOT":
        return f"NOT {ast_to_string(ast[1])}"
    return f"({ast_to_string(ast[1])} {k} {ast_to_string(ast[2])})"


def ast_positive_terms(ast, negated: bool = False) -> List[str]:
    """Term yang tidak berada di bawah NOT (dipakai untuk mengurutkan hasil boolean)."""
    if ast is None:
        return []
    k = ast[0]
    if k == "TERM":
        return [] if negated else [ast[1]]
    if k == "PHRASE":
        return [] if negated else list(ast[1])
    if k == "NOT":
        return ast_positive_terms(ast[1], not negated)
    return ast_positive_terms(ast[1], negated) + ast_positive_terms(ast[2], negated)


# ===========================================================================
# 3. LSA (Latent Semantic Analysis)
# ===========================================================================
class LSAModel:
    """LSA: TF-IDF (dari index kita) -> SVD terpotong -> ruang laten k dimensi."""

    def __init__(self, index: InvertedIndex, n_components: int = 50,
                 min_df: int = 1, max_features: int = 60000) -> None:
        terms = [t for t, e in index.index.items() if e["df"] >= min_df]
        if len(terms) > max_features:
            terms = [t for t in heapq.nlargest(max_features, terms, key=index.df)]
        terms.sort()
        self.terms = terms
        self.t2i = {t: i for i, t in enumerate(terms)}
        n = index.n_docs
        self.n_docs = n
        self.idf = np.array([math.log((1 + n) / (1 + index.df(t))) + 1.0 for t in terms])
        rows, cols, vals = [], [], []
        for t, j in self.t2i.items():
            for d, pos in index.postings(t).items():
                rows.append(d)
                cols.append(j)
                vals.append((1.0 + math.log10(len(pos))) * self.idf[j])
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(terms)))
        norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        A = sparse.diags(1.0 / norms) @ A
        k = max(1, min(n_components, min(A.shape) - 1))
        U, S, Vt = svds(A.astype(np.float64), k=k)
        order = np.argsort(-S)
        self.S = S[order]
        self.Vt = Vt[order]
        U = U[:, order]
        self.k = k
        self.explained = float((S ** 2).sum() / max(A.multiply(A).sum(), 1e-12))
        self.doc_vecs = self._unit(U)

    @staticmethod
    def _unit(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return M / n

    def query_vector(self, query_terms: Sequence[str]) -> Optional[np.ndarray]:
        q = np.zeros(len(self.terms))
        for t, c in Counter(query_terms).items():
            j = self.t2i.get(t)
            if j is not None:
                q[j] = (1.0 + math.log10(c)) * self.idf[j]
        if not q.any():
            return None
        folded = (q @ self.Vt.T) / self.S          # fold-in: q V S^-1
        nrm = np.linalg.norm(folded)
        return folded / nrm if nrm else None

    def search(self, query_terms: Sequence[str], top_k: int = 10) -> List[Tuple[int, float]]:
        qv = self.query_vector(query_terms)
        if qv is None:
            return []
        sims = self.doc_vecs @ qv
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]

    def topics(self, n_topics: int = 5, n_terms: int = 8) -> List[List[Tuple[str, float]]]:
        out = []
        for j in range(min(n_topics, self.k)):
            row = self.Vt[j]
            top = np.argsort(-np.abs(row))[:n_terms]
            out.append([(self.terms[i], float(row[i])) for i in top])
        return out


# ===========================================================================
# 4. SENTENCE EMBEDDING + VECTOR STORE
# ===========================================================================
class SemanticStore:
    """Embedding kalimat (sentence-transformers) + vector store FAISS / ChromaDB / NumPy."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, backend: str = "faiss",
                 cache_dir: str = ".cache/embeddings") -> None:
        self.model_name = model_name
        self.requested_backend = backend
        self.backend = backend
        self.cache_dir = cache_dir
        self.model = None
        self.matrix: Optional[np.ndarray] = None
        self._faiss = None
        self._chroma = None
        self.notes: List[str] = []

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(self, texts: Sequence[str]) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        key = hashlib.md5((self.model_name + "||" + "\x1f".join(texts)).encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{key}.npy")
        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            self.notes.append("Embedding dimuat dari cache.")
        else:
            model = self._load_model()
            emb = model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False)
            emb = np.asarray(emb, dtype="float32")
            np.save(cache_path, emb)
        self.matrix = emb
        self._init_backend()

    def _init_backend(self) -> None:
        emb = self.matrix
        dim = emb.shape[1]
        if self.requested_backend == "faiss":
            try:
                import faiss

                idx = faiss.IndexFlatIP(dim)
                idx.add(emb)
                self._faiss = idx
                self.backend = "faiss"
                return
            except Exception as e:
                self.notes.append(f"FAISS tidak tersedia ({e.__class__.__name__}); memakai NumPy.")
        elif self.requested_backend == "chroma":
            try:
                import chromadb

                client = chromadb.Client()
                col = client.create_collection(name=f"law_{uuid.uuid4().hex[:8]}",
                                               metadata={"hnsw:space": "cosine"})
                ids = [str(i) for i in range(len(emb))]
                for s in range(0, len(emb), 2000):
                    col.add(ids=ids[s:s + 2000], embeddings=emb[s:s + 2000].tolist())
                self._chroma = col
                self.backend = "chroma"
                return
            except Exception as e:
                self.notes.append(f"ChromaDB tidak tersedia ({e.__class__.__name__}); memakai NumPy.")
        self.backend = "numpy"

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        model = self._load_model()
        qv = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
        k = min(top_k, len(self.matrix))
        if self.backend == "faiss" and self._faiss is not None:
            scores, ids = self._faiss.search(qv, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        if self.backend == "chroma" and self._chroma is not None:
            res = self._chroma.query(query_embeddings=qv.tolist(), n_results=k)
            return [(int(i), 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]
        sims = self.matrix @ qv[0]
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]


# ===========================================================================
# 5. HYBRID
# ===========================================================================
def _minmax(pairs: Sequence[Tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {i: 1.0 for i, _ in pairs}
    return {i: (s - lo) / (hi - lo) for i, s in pairs}


def hybrid_combine(lexical: Sequence[Tuple[int, float]], semantic: Sequence[Tuple[int, float]],
                   alpha: float = 0.5, method: str = "weighted", rrf_k: int = 60,
                   top_k: int = 10) -> List[dict]:
    """
    weighted : skor = alpha * minmax(leksikal) + (1 - alpha) * minmax(semantik)
    rrf      : skor = sum 1 / (k + rank)  (Reciprocal Rank Fusion)
    """
    scores: Dict[int, float] = defaultdict(float)
    parts: Dict[int, dict] = defaultdict(lambda: {"lex": 0.0, "sem": 0.0})
    if method == "rrf":
        for rank, (i, _) in enumerate(lexical, start=1):
            scores[i] += 1.0 / (rrf_k + rank)
            parts[i]["lex"] = 1.0 / (rrf_k + rank)
        for rank, (i, _) in enumerate(semantic, start=1):
            scores[i] += 1.0 / (rrf_k + rank)
            parts[i]["sem"] = 1.0 / (rrf_k + rank)
    else:
        ln, sn = _minmax(lexical), _minmax(semantic)
        for i in set(ln) | set(sn):
            parts[i]["lex"] = ln.get(i, 0.0)
            parts[i]["sem"] = sn.get(i, 0.0)
            scores[i] = alpha * ln.get(i, 0.0) + (1 - alpha) * sn.get(i, 0.0)
    ranked = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
    return [{"idx": i, "score": s, "lex": parts[i]["lex"], "sem": parts[i]["sem"], "matched": []}
            for i, s in ranked]


# ===========================================================================
# 6. SNIPPET
# ===========================================================================
def make_snippet(text: str, query: str, width: int = 280, stem: bool = True) -> str:
    """Cuplikan teks di sekitar kata kunci, dengan penyorotan <mark> (aman HTML)."""
    stems = [t for t in preprocess_text(query, stem=stem) if len(t) >= 3]
    stems += [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3 and w not in STOPWORDS]
    stems = sorted(set(stems), key=len, reverse=True)
    if not stems:
        return html.escape(text[:width]) + ("…" if len(text) > width else "")
    pattern = re.compile(r"\w*(?:" + "|".join(re.escape(s) for s in stems) + r")\w*", re.I)
    m = pattern.search(text)
    start = max(0, (m.start() if m else 0) - width // 3)
    end = min(len(text), start + width)
    seg = text[start:end]
    out, last = [], 0
    for mm in pattern.finditer(seg):
        out.append(html.escape(seg[last:mm.start()]))
        out.append(f"<mark>{html.escape(mm.group(0))}</mark>")
        last = mm.end()
    out.append(html.escape(seg[last:]))
    return ("… " if start > 0 else "") + "".join(out) + (" …" if end < len(text) else "")


# ===========================================================================
# 7. SEARCH ENGINE (fasad)
# ===========================================================================
class SearchEngine:
    """Menyatukan seluruh metode pencarian di atas satu korpus (unit pencarian = chunk)."""

    def __init__(self, corpus: Corpus, use_bigram: bool = True) -> None:
        self.corpus = corpus
        self.chunks = corpus.chunks
        self.use_bigram = use_bigram
        self.stem = corpus.config.get("stem", True)
        self.index = InvertedIndex()
        for i, ch in enumerate(self.chunks):
            self.index.add_document(i, ch["tokens"], use_bigram)
        self._norm_cache: Dict[Tuple[str, bool], List[float]] = {}
        self._lsa_cache: Dict[int, LSAModel] = {}
        self._dtm_cache = None
        self.semantic: Optional[SemanticStore] = None

    # -- util -------------------------------------------------------------
    def query_tokens(self, query: str) -> List[str]:
        return preprocess_text(query, stem=self.stem)

    def query_terms(self, query: str) -> List[str]:
        return make_terms(self.query_tokens(query), self.use_bigram)

    def doc_dtm(self):
        """DTM tingkat dokumen (baris = dokumen peraturan) untuk analisis sparsity."""
        if self._dtm_cache is None:
            ids = [d["doc_id"] for d in self.corpus.docs]
            lists = [make_terms(self.corpus.doc_tokens[i], self.use_bigram) for i in ids]
            dtm, vocab = build_dtm(lists)
            self._dtm_cache = (dtm, vocab, ids)
        return self._dtm_cache

    def _result(self, scores: Dict[int, float], matched: Dict[int, List[str]], top_k: int) -> List[dict]:
        ranked = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return [{"idx": i, "score": s, "matched": sorted(set(matched.get(i, [])))}
                for i, s in ranked if s > 0]

    # -- Boolean ----------------------------------------------------------
    def parse_boolean(self, query: str):
        return BooleanParser(self.query_tokens).parse(query)

    def _eval(self, ast) -> Set[int]:
        if ast is None:
            return set()
        k = ast[0]
        if k == "TERM":
            return set(self.index.postings(ast[1]))
        if k == "PHRASE":
            return set(self.index.phrase_docs(ast[1]))
        if k == "NOT":
            return set(range(len(self.chunks))) - self._eval(ast[1])
        a, b = self._eval(ast[1]), self._eval(ast[2])
        return a & b if k == "AND" else a | b

    def search_boolean(self, query: str, top_k: int = 20) -> Tuple[List[dict], object, int]:
        ast = self.parse_boolean(query)
        matched_docs = self._eval(ast)
        pos_terms = ast_positive_terms(ast)
        order = self._bm25_scores(pos_terms, 1.5, 0.75) if pos_terms else {}
        ranked = sorted(matched_docs, key=lambda d: (-order.get(d, 0.0), d))[:top_k]
        results = [{"idx": d, "score": order.get(d, 1.0) or 1.0,
                    "matched": [t for t in pos_terms if d in self.index.postings(t)]} for d in ranked]
        return results, ast, len(matched_docs)

    def search_phrase(self, phrase: str, top_k: int = 20) -> List[dict]:
        terms = self.query_tokens(phrase)
        hits = self.index.phrase_docs(terms)
        ranked = sorted(hits.items(), key=lambda x: (-len(x[1]), x[0]))[:top_k]
        return [{"idx": d, "score": float(len(p)), "matched": terms} for d, p in ranked]

    # -- TF-IDF -----------------------------------------------------------
    @staticmethod
    def _tf_weight(raw: int, mode: str) -> float:
        if raw <= 0:
            return 0.0
        return float(raw) if mode == "raw" else 1.0 + math.log10(raw)

    def idf(self, term: str, smooth: bool = True) -> float:
        n, df = self.index.n_docs, self.index.df(term)
        if df == 0:
            return 0.0
        return math.log((1 + n) / (1 + df)) + 1.0 if smooth else math.log10(n / df)

    def _doc_norms(self, tf_mode: str, smooth: bool) -> List[float]:
        key = (tf_mode, smooth)
        if key not in self._norm_cache:
            sq = [0.0] * self.index.n_docs
            for term, entry in self.index.index.items():
                w_idf = self.idf(term, smooth)
                for d, pos in entry["postings"].items():
                    w = self._tf_weight(len(pos), tf_mode) * w_idf
                    sq[d] += w * w
            self._norm_cache[key] = [math.sqrt(s) for s in sq]
        return self._norm_cache[key]

    def search_tfidf(self, query: str, tf_mode: str = "log", normalize: bool = True,
                     smooth_idf: bool = True, top_k: int = 10) -> List[dict]:
        q_terms = self.query_terms(query)
        qtf = Counter(t for t in q_terms if t in self.index.index)
        if not qtf:
            return []
        qw = {t: self._tf_weight(c, tf_mode) * self.idf(t, smooth_idf) for t, c in qtf.items()}
        q_norm = math.sqrt(sum(w * w for w in qw.values())) or 1.0
        scores: Dict[int, float] = defaultdict(float)
        matched: Dict[int, List[str]] = defaultdict(list)
        for t, w_q in qw.items():
            w_idf = self.idf(t, smooth_idf)
            for d, pos in self.index.postings(t).items():
                scores[d] += w_q * self._tf_weight(len(pos), tf_mode) * w_idf
                matched[d].append(t)
        if normalize:
            norms = self._doc_norms(tf_mode, smooth_idf)
            for d in list(scores):
                scores[d] /= (q_norm * (norms[d] or 1.0))      # cosine similarity
        return self._result(scores, matched, top_k)

    # -- BM25 -------------------------------------------------------------
    def _bm25_scores(self, terms: Sequence[str], k1: float, b: float,
                     matched: Optional[Dict[int, List[str]]] = None) -> Dict[int, float]:
        n, avgdl = self.index.n_docs, self.index.avgdl or 1.0
        scores: Dict[int, float] = defaultdict(float)
        for t in set(terms):
            df = self.index.df(t)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for d, pos in self.index.postings(t).items():
                tf, dl = len(pos), self.index.doc_len[d]
                scores[d] += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl))
                if matched is not None:
                    matched[d].append(t)
        return scores

    def search_bm25(self, query: str, k1: float = 1.5, b: float = 0.75, top_k: int = 10) -> List[dict]:
        matched: Dict[int, List[str]] = defaultdict(list)
        scores = self._bm25_scores(self.query_terms(query), k1, b, matched)
        return self._result(scores, matched, top_k)

    # -- LSA & Semantic ---------------------------------------------------
    def get_lsa(self, n_components: int = 50) -> LSAModel:
        if n_components not in self._lsa_cache:
            self._lsa_cache[n_components] = LSAModel(self.index, n_components)
        return self._lsa_cache[n_components]

    def search_lsa(self, query: str, n_components: int = 50, top_k: int = 10) -> List[dict]:
        pairs = self.get_lsa(n_components).search(self.query_terms(query), top_k)
        return [{"idx": i, "score": s, "matched": []} for i, s in pairs]

    def build_semantic(self, model_name: str = DEFAULT_EMBED_MODEL, backend: str = "faiss") -> SemanticStore:
        store = SemanticStore(model_name, backend)
        store.build([c["text"] for c in self.chunks])
        self.semantic = store
        return store

    def search_semantic(self, query: str, top_k: int = 10) -> List[dict]:
        if self.semantic is None:
            raise RuntimeError("Vector store belum dibangun.")
        return [{"idx": i, "score": s, "matched": []} for i, s in self.semantic.search(query, top_k)]

    # -- Hybrid -----------------------------------------------------------
    def search_hybrid(self, query: str, alpha: float = 0.5, method: str = "weighted",
                      semantic_source: str = "embedding", n_candidates: int = 50,
                      k1: float = 1.5, b: float = 0.75, lsa_components: int = 50,
                      top_k: int = 10) -> Tuple[List[dict], str]:
        lex = [(r["idx"], r["score"]) for r in self.search_bm25(query, k1, b, n_candidates)]
        used = semantic_source
        if semantic_source == "embedding" and self.semantic is not None:
            sem = [(r["idx"], r["score"]) for r in self.search_semantic(query, n_candidates)]
        else:
            used = "lsa"
            sem = [(r["idx"], r["score"]) for r in self.search_lsa(query, lsa_components, n_candidates)]
        return hybrid_combine(lex, sem, alpha, method, top_k=top_k), used

"""CPU lexical baseline and rank fusion; no model or network access."""
from collections import Counter, defaultdict
import math

from .text import tokenize


class BM25:
    def __init__(self, documents: list[dict], *, k1: float = 1.2, b: float = 0.75):
        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Nonempty documents, k1 > 0 and 0 <= b <= 1 required")
        self.documents = {}
        self.terms = {}
        self.df = Counter()
        self.k1, self.b = k1, b
        for document in documents:
            doc_id = document["document_id"]
            if not isinstance(doc_id, str) or not doc_id or doc_id in self.documents:
                raise ValueError("Document IDs must be nonempty, unique strings")
            self.documents[doc_id] = document
            counts = Counter(tokenize(document.get("title", "") + " " + document["text"]))
            self.terms[doc_id] = counts
            self.df.update(counts.keys())
        self.lengths = {doc_id: sum(terms.values()) for doc_id, terms in self.terms.items()}
        self.avg_length = sum(self.lengths.values()) / len(documents) or 1.0

    def search(self, query: str, *, top_k: int = 50) -> list[dict]:
        if top_k < 1 or not tokenize(query):
            raise ValueError("Positive top_k and a nonempty lexical query required")
        size = len(self.documents)
        query_terms = sorted(set(tokenize(query)))
        scores = []
        for doc_id, counts in self.terms.items():
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                idf = math.log(1 + (size - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.avg_length)
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append({"document_id": doc_id, "score": score})
        scores.sort(key=lambda row: (-row["score"], row["document_id"]))
        return [dict(row, rank=rank) for rank, row in enumerate(scores[:top_k], 1)]


def reciprocal_rank_fusion(lanes: dict[str, list[str]], *, k: int = 60, top_k: int = 50) -> list[dict]:
    """Equal-weight RRF over one-based source ranks; missing sources contribute zero."""
    if not lanes or k < 1 or top_k < 1:
        raise ValueError("Nonempty lanes and positive k/top_k required")
    scores = defaultdict(float)
    source_ranks = defaultdict(dict)
    for lane, ids in lanes.items():
        if not ids or len(ids) != len(set(ids)) or any(not isinstance(d, str) or not d for d in ids):
            raise ValueError(f"Invalid or duplicate document IDs in lane {lane}")
        for rank, doc_id in enumerate(ids, 1):
            scores[doc_id] += 1.0 / (k + rank)
            source_ranks[doc_id][lane] = rank
    order = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    return [{"document_id": doc_id, "score": scores[doc_id], "rank": rank,
             "source_ranks": source_ranks[doc_id]} for rank, doc_id in enumerate(order, 1)]

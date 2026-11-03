"""Deterministic Unicode normalization and lexical evidence selection."""
import re
import unicodedata


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def tokenize(text: str) -> list[str]:
    """Unicode word tokens; Vietnamese accents are retained, no learned segmenter."""
    return re.findall(r"[^\W_]+", normalize(text).casefold(), flags=re.UNICODE)


def chunks(text: str, *, words: int = 120, overlap: int = 20) -> list[str]:
    """Demo word windows, not the historical model-token or legal-unit chunker."""
    if words < 1 or not 0 <= overlap < words:
        raise ValueError("Require words > overlap >= 0")
    tokens = normalize(text).split()
    result = []
    for start in range(0, len(tokens), words - overlap):
        result.append(" ".join(tokens[start:start + words]))
        if start + words >= len(tokens):
            break
    return result


def select_evidence(query: str, passages: list[str], *, top_n: int = 2) -> list[dict]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    query_terms = set(tokenize(query))
    ranked = sorted(
        ((len(query_terms & set(tokenize(text))), index, normalize(text))
         for index, text in enumerate(passages)),
        key=lambda row: (-row[0], row[1]),
    )
    return [{"chunk_index": index, "lexical_overlap": score, "text": text}
            for score, index, text in ranked[:top_n]]

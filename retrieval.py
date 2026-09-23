"""
retrieval.py — pure-Python TF-IDF retrieval over the frozen EmiTechLogic
corpus. No external libraries, no embeddings, no vector database.

query -> tokenize -> TF-IDF -> cosine similarity -> top-k chunks
"""
import json
import math
import re
from collections import Counter

STOPWORDS = set("""
a an the is are was were be been being to of in on for and or with as by at
from this that these those it its can may might will would should could
you your we our their his her not no if then than so but into over under
about across out up down when where which who what how why do does did
doing has have had having i
""".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOPWORDS and len(w) > 2]


class Retriever:
    def __init__(self, chunks_path):
        self.chunks = json.load(open(chunks_path))
        self._doc_tokens = [tokenize(c["text"]) for c in self.chunks]
        self._build_index()

    def _build_index(self):
        df = Counter()
        for toks in self._doc_tokens:
            for w in set(toks):
                df[w] += 1
        n = len(self._doc_tokens)
        self.idf = {w: math.log(n / df[w]) for w in df}
        self._doc_vecs = [self._vectorize(toks) for toks in self._doc_tokens]

    def _vectorize(self, tokens):
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        return {w: (tf[w] / length) * self.idf.get(w, 0) for w in tf if w in self.idf}

    @staticmethod
    def _cosine(v1, v2):
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        num = sum(v1[w] * v2[w] for w in common)
        d1 = math.sqrt(sum(x * x for x in v1.values()))
        d2 = math.sqrt(sum(x * x for x in v2.values()))
        if d1 == 0 or d2 == 0:
            return 0.0
        return num / (d1 * d2)

    def search(self, query, top_k=3):
        qvec = self._vectorize(tokenize(query))
        scored = [(self._cosine(qvec, dv), i) for i, dv in enumerate(self._doc_vecs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, i in scored[:top_k]:
            c = self.chunks[i]
            results.append({
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "section": c["section"],
                "source_url": c["source_url"],
                "group": c["group"],
                "score": round(score, 4),
                "text": c["text"],
            })
        return results


if __name__ == "__main__":
    retriever = Retriever("corpus/chunks.json")
    for q in ["How should I handle missing values?", "how do agents coordinate tasks"]:
        print(f"\nQuery: {q}")
        for r in retriever.search(q, top_k=3):
            print(f"  {r['score']:.3f}  {r['title'][:45]:<45} :: {r['section'][:35]}")

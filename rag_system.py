"""
rag_system.py — retrieval only. This system can find relevant
information in the corpus and summarize what it found. It has NO
access to environment.py, so it cannot change ticket state under any
circumstances — even if the task asks it to.

Task -> TF-IDF retrieval -> top-k chunks -> knowledge-based response
"""
import re

from retrieval import Retriever

ACTION_VERBS_RE = re.compile(
    r"\b(set|assign|change|update|schedule|reserve|move|close|resolve|escalate)\b",
    re.IGNORECASE,
)
TICKET_RE = re.compile(r"\bT\d{3}\b")


class RAGSystem:
    def __init__(self, corpus_path="corpus/chunks.json"):
        self.retriever = Retriever(corpus_path)

    def run(self, task_text, top_k=3):
        results = self.retriever.search(task_text, top_k=top_k)

        looks_like_action_request = bool(
            ACTION_VERBS_RE.search(task_text) and TICKET_RE.search(task_text)
        )

        response = {
            "retrieved": [
                {"title": r["title"], "section": r["section"], "score": r["score"]}
                for r in results
            ],
            "action_taken": None,
        }

        if looks_like_action_request:
            response["result"] = "FAIL"
            response["note"] = (
                "Retrieved relevant background, but this system has no access to "
                "the ticket environment and cannot execute the requested state change."
            )
        else:
            response["result"] = "PASS" if results and results[0]["score"] > 0 else "FAIL"
            response["note"] = "Answered from retrieved knowledge only; no state was touched."

        return response


if __name__ == "__main__":
    rag = RAGSystem()

    print("--- knowledge-only task ---")
    r = rag.run("What are some common chunking strategies for RAG systems?")
    print(r["result"], "-", r["note"])
    for t in r["retrieved"]:
        print(f"   {t['score']:.3f}  {t['title'][:50]}")

    print("\n--- action-request task (should explicitly refuse) ---")
    r = rag.run("Set T101 priority to high and assign it to Alice.")
    print(r["result"], "-", r["note"])

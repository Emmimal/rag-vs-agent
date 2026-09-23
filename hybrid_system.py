"""
hybrid_system.py — connects retrieval to the agent planner explicitly.

Task -> retrieve knowledge -> pass retrieved context to planner ->
planner chooses action -> environment executes -> verify resulting state

The retrieval step and the planning step are separate, visible calls.
Nothing reaches into the corpus from inside the agent - this file is
the only place that wires the two together.
"""
from retrieval import Retriever
from environment import Environment
from agent_system import AgentPlanner


class HybridSystem:
    def __init__(self, corpus_path="corpus/chunks.json", environment=None):
        self.retriever = Retriever(corpus_path)
        self.env = environment or Environment()
        self.planner = AgentPlanner(self.env)

    def run(self, task_text, top_k=3):
        retrieved = self.retriever.search(task_text, top_k=top_k)
        plan_result = self.planner.run(task_text, retrieved_context=retrieved)
        plan_result["retrieved"] = [
            {"title": r["title"], "section": r["section"], "group": r["group"], "score": r["score"]}
            for r in retrieved
        ]
        return plan_result


if __name__ == "__main__":
    hybrid = HybridSystem()

    print("--- knowledge-dependent task (should now PASS) ---")
    result = hybrid.run(
        "T101 has a chunking-related retrieval problem. Determine the appropriate "
        "category from the knowledge base and update the ticket."
    )
    print(result["result"], "-", result["steps"])
    print("Retrieved from:", [r["title"][:40] for r in result["retrieved"]])
    print("Ticket after:", hybrid.env.get_ticket("T101"))

    print("\n--- pure action task (retrieval shouldn't matter) ---")
    result2 = hybrid.run("Set T102 priority to urgent and assign it to Bob.")
    print(result2["result"], "-", result2["steps"])

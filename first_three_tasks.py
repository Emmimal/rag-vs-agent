"""
first_three_tasks.py — sanity run of one task from each category
(knowledge-only, action-only, knowledge+action) through all three
systems, before building the full 9-task set.
"""
from environment import Environment
from agent_system import AgentPlanner
from rag_system import RAGSystem
from hybrid_system import HybridSystem

TASKS = [
    ("Task A: knowledge-only",
     "What are some common chunking strategies for RAG systems?"),
    ("Task B: action-only",
     "Set T102 priority to urgent and assign it to Bob."),
    ("Task C: knowledge + action",
     "T101 has a chunking-related retrieval problem. Determine the appropriate "
     "category from the knowledge base and update the ticket."),
]

rag = RAGSystem()

for label, task_text in TASKS:
    print("=" * 60)
    print(label)
    print(task_text)
    print("=" * 60)

    # fresh environment per task so systems don't see each other's state
    env_agent = Environment()
    env_hybrid = Environment()
    agent = AgentPlanner(env_agent)
    hybrid = HybridSystem(environment=env_hybrid)

    print("\nRAG")
    r = rag.run(task_text)
    print(f"  Retrieved: {'YES' if r['retrieved'] else 'NO'}")
    print(f"  Action:    {'N/A (no environment access)' if r['action_taken'] is None else r['action_taken']}")
    print(f"  Result:    {r['result']}  ({r['note']})")

    print("\nAgent")
    a = agent.run(task_text)
    print(f"  Retrieved: NO (no corpus access)")
    print(f"  Action:    {a['steps'] if a['steps'] else 'none taken'}")
    print(f"  Result:    {a['result']}" + (f"  ({a['reason']})" if a.get("reason") else ""))

    print("\nRAG + Agent")
    h = hybrid.run(task_text)
    print(f"  Retrieved: {'YES' if h['retrieved'] else 'NO'}")
    print(f"  Action:    {h['steps'] if h['steps'] else 'none taken'}")
    print(f"  Result:    {h['result']}" + (f"  ({h['reason']})" if h.get("reason") else ""))
    print()

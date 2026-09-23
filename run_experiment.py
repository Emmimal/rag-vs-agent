"""
run_experiment.py — runs all 9 frozen tasks through RAG, Agent, and
Hybrid, each on a fresh Environment. Verifies actual resulting state
against the frozen `expected` block in tasks.py rather than trusting
each system's own self-reported result, and flags any mismatch
between self-report and ground truth.
"""
from environment import Environment
from agent_system import AgentPlanner
from rag_system import RAGSystem
from hybrid_system import HybridSystem
from tasks import TASKS

rag = RAGSystem()


def get_field(env, dotted_key):
    ticket_id, field = dotted_key.split(".")
    ticket = env.get_ticket(ticket_id)
    return ticket.get(field) if ticket else None


def verify_ground_truth(env, expected, env_was_touched, result, system_name):
    """Returns (ground_truth_pass, mismatches)."""
    state_changes = expected.get("state_changes", {})
    if not state_changes:
        # knowledge-only task: environment must stay untouched AND the
        # system must have actually produced an answer - "didn't touch
        # the environment" alone is not success, it's just not-failure.
        if env_was_touched:
            return False, ["environment was mutated but shouldn't have been"]
        answered = bool(result.get("retrieved")) or result.get("mode") == "knowledge_answer"
        if expected.get("answer_required") and not answered:
            return False, [f"{system_name} produced no answer to a knowledge question"]
        return True, []
    mismatches = []
    for key, want in state_changes.items():
        got = get_field(env, key)
        if got != want:
            mismatches.append(f"{key}: expected {want!r}, got {got!r}")
    return (len(mismatches) == 0), mismatches


def env_snapshot_equal(before, after):
    return before == after


all_results = []

for task in TASKS:
    print("=" * 70)
    print(f"{task['id']} ({task['category']}): {task['prompt']}")
    print("=" * 70)

    row = {"id": task["id"], "category": task["category"]}

    # ---------- RAG ----------
    env = Environment()
    before = env.snapshot()
    r = rag.run(task["prompt"])
    after = env.snapshot()
    touched = not env_snapshot_equal(before, after)  # RAG never touches env; sanity check
    gt_pass, mismatches = verify_ground_truth(env, task["expected"], touched, r, "RAG")
    print("\nRAG")
    print(f"  Retrieved:      {'YES' if r['retrieved'] else 'NO'}")
    print(f"  Self-reported:  {r['result']}")
    print(f"  Ground truth:   {'PASS' if gt_pass else 'FAIL'}" + (f"  ({'; '.join(mismatches)})" if mismatches else ""))
    if (r["result"] == "PASS") != gt_pass:
        print("  >>> MISMATCH: self-report disagrees with ground truth")
    row["rag"] = "PASS" if gt_pass else "FAIL"

    # ---------- Agent ----------
    env = Environment()
    before = env.snapshot()
    agent = AgentPlanner(env)
    a = agent.run(task["prompt"])
    after = env.snapshot()
    touched = not env_snapshot_equal(before, after)
    gt_pass, mismatches = verify_ground_truth(env, task["expected"], touched, a, "Agent")
    print("\nAgent")
    print(f"  Steps:          {a['steps'] if a['steps'] else 'none'}")
    print(f"  Self-reported:  {a['result']}" + (f"  ({a['reason']})" if a.get("reason") else ""))
    print(f"  Ground truth:   {'PASS' if gt_pass else 'FAIL'}" + (f"  ({'; '.join(mismatches)})" if mismatches else ""))
    if (a["result"] == "PASS") != gt_pass:
        print("  >>> MISMATCH: self-report disagrees with ground truth")
    row["agent"] = "PASS" if gt_pass else "FAIL"

    # ---------- Hybrid ----------
    env = Environment()
    before = env.snapshot()
    hybrid = HybridSystem(environment=env)
    h = hybrid.run(task["prompt"])
    after = env.snapshot()
    touched = not env_snapshot_equal(before, after)
    gt_pass, mismatches = verify_ground_truth(env, task["expected"], touched, h, "Hybrid")
    print("\nHybrid (RAG + Agent)")
    print(f"  Retrieved:      {[c['title'][:35] for c in h['retrieved']]}")
    print(f"  Steps:          {h['steps'] if h['steps'] else 'none'}")
    print(f"  Self-reported:  {h['result']}" + (f"  ({h['reason']})" if h.get("reason") else ""))
    print(f"  Ground truth:   {'PASS' if gt_pass else 'FAIL'}" + (f"  ({'; '.join(mismatches)})" if mismatches else ""))
    if (h["result"] == "PASS") != gt_pass:
        print("  >>> MISMATCH: self-report disagrees with ground truth")
    row["hybrid"] = "PASS" if gt_pass else "FAIL"

    print()
    all_results.append(row)

print("=" * 70)
print("SUMMARY (ground truth)")
print("=" * 70)
print(f"{'ID':<5}{'Category':<18}{'RAG':<8}{'Agent':<8}{'Hybrid':<8}")
for row in all_results:
    print(f"{row['id']:<5}{row['category']:<18}{row['rag']:<8}{row['agent']:<8}{row['hybrid']:<8}")

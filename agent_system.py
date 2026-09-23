"""
agent_system.py — the smallest planner that can act on the Environment.
Deliberately NOT an "autonomous AI agent": it's a deterministic
rule-based planner that reads a task string, extracts explicit
instructions (ticket id, priority, status, assignee), and executes
matching Environment actions.

Critically, this planner has NO access to the corpus/retriever. If a
task requires knowledge that isn't stated in the task text itself
(e.g. "determine the correct category from the guidance"), this
planner cannot supply it — that's the boundary the experiment is
built around, not a bug to work around.
"""
import re

from environment import Environment, VALID_PRIORITIES, VALID_STATUSES

TICKET_RE = re.compile(r"\bT\d{3}\b")
ACTION_VERBS_RE = re.compile(
    r"\b(set|assign|change|update|schedule|reserve|move|close|resolve|escalate)\b",
    re.IGNORECASE,
)
PRIORITY_RE = re.compile(r"\bpriority\s+to\s+(\w+)", re.IGNORECASE)
# supports "status to X" and the reversed "to X status" phrasing
STATUS_TO_X_RE = re.compile(r"\bstatus\s+to\s+(\w+)", re.IGNORECASE)
TO_X_STATUS_RE = re.compile(r"\bto\s+(\w+)\s+status\b", re.IGNORECASE)
# supports "assign to Alice" and "assign T101 to Alice" (ticket id between verb and target)
ASSIGN_RE = re.compile(r"\bassign(?:ed)?\s+(?:T\d{3}\s+)?(?:it\s+)?to\s+([A-Za-z]+)", re.IGNORECASE)
CATEGORY_EXPLICIT_RE = re.compile(r"\bcategory\s+to\s+([\w-]+)", re.IGNORECASE)
NEEDS_KNOWLEDGE_RE = re.compile(
    r"determine the (?:appropriate|correct) category|"
    r"according to (?:the )?(?:relevant )?(?:emitechlogic )?guidance|"
    r"based on the knowledge base", re.IGNORECASE
)


class AgentPlanner:
    """Deterministic planner: task text -> environment actions."""

    def __init__(self, environment: Environment):
        self.env = environment

    @staticmethod
    def _resolve_category_from_context(retrieved_context):
        """Deterministic, rule-based mapping from retrieved chunk group to
        ticket category - majority vote across the passed-in chunks.
        No LLM, no fuzzy matching: just counting which corpus group the
        retrieved evidence came from."""
        group_to_category = {
            "rag": "retrieval",
            "agents": "agentic-system",
            "data_science": "data-quality",
            "python": "code-quality",
        }
        counts = {}
        for chunk in retrieved_context:
            cat = group_to_category.get(chunk.get("group"), "general")
            counts[cat] = counts.get(cat, 0) + 1
        return max(counts, key=counts.get) if counts else "general"

    def run(self, task_text, retrieved_context=None):
        """retrieved_context: optional list of chunk dicts (as returned by
        Retriever.search) already fetched by the caller. This planner
        never queries the corpus itself - the caller (hybrid_system.py)
        is responsible for retrieving and passing context explicitly."""
        steps = []

        # First question: does this task require an action at all, or is
        # it a knowledge question that just happens to be routed through
        # a planner? Earlier version skipped this check and assumed every
        # task was a ticket operation - that broke pure knowledge tasks.
        action_required = bool(ACTION_VERBS_RE.search(task_text)) and bool(TICKET_RE.search(task_text))

        if not action_required:
            if retrieved_context:
                return {
                    "steps": steps, "result": "PASS", "reason": None,
                    "mode": "knowledge_answer",
                    "note": "No action required; answered directly from retrieved context.",
                }
            return {
                "steps": steps, "result": "FAIL",
                "reason": "no action required, and this planner has no corpus access to answer a knowledge question",
            }

        ticket_match = TICKET_RE.search(task_text)
        if not ticket_match:
            return {"steps": steps, "result": "FAIL", "reason": "no ticket id found in task"}
        ticket_id = ticket_match.group(0)

        if self.env.get_ticket(ticket_id) is None:
            return {"steps": steps, "result": "FAIL", "reason": f"unknown ticket {ticket_id}"}

        acted = False

        priority_match = PRIORITY_RE.search(task_text)
        if priority_match:
            priority = priority_match.group(1).lower()
            ok = self.env.change_priority(ticket_id, priority)
            steps.append(("change_priority", priority, ok))
            acted = True

        status_match = STATUS_TO_X_RE.search(task_text) or TO_X_STATUS_RE.search(task_text)
        if status_match:
            status = status_match.group(1).lower()
            ok = self.env.change_status(ticket_id, status)
            steps.append(("change_status", status, ok))
            acted = True

        assign_match = ASSIGN_RE.search(task_text)
        if assign_match:
            assignee = assign_match.group(1)
            ok = self.env.assign_ticket(ticket_id, assignee)
            steps.append(("assign_ticket", assignee, ok))
            acted = True

        # explicit category ("set category to X") this planner can do directly
        category_match = CATEGORY_EXPLICIT_RE.search(task_text)
        if category_match:
            category = category_match.group(1).lower()
            ok = self.env.set_category(ticket_id, category)
            steps.append(("set_category", category, ok))
            acted = True

        # knowledge-dependent category assignment: this planner has no
        # corpus access of its own. It can only resolve this if the
        # caller explicitly passed retrieved_context (see hybrid_system.py).
        elif NEEDS_KNOWLEDGE_RE.search(task_text):
            if not retrieved_context:
                steps.append(("set_category", None,
                              "NOT AVAILABLE - requires knowledge base, agent has no corpus access"))
                return {"steps": steps, "result": "FAIL",
                        "reason": "task requires knowledge this planner cannot retrieve"}
            category = self._resolve_category_from_context(retrieved_context)
            ok = self.env.set_category(ticket_id, category)
            steps.append(("set_category", category, ok))
            acted = True

        if not acted:
            return {"steps": steps, "result": "FAIL", "reason": "no actionable instruction recognized"}

        all_ok = all(s[2] is True for s in steps)
        return {"steps": steps, "result": "PASS" if all_ok else "FAIL",
                "reason": None if all_ok else "one or more actions failed"}


if __name__ == "__main__":
    env = Environment()
    agent = AgentPlanner(env)

    print("--- action-only task ---")
    result = agent.run("Set T101 priority to high and assign it to Alice.")
    print(result)
    print(env.get_ticket("T101"))

    print("\n--- knowledge-dependent task (should fail, no corpus access) ---")
    result = agent.run(
        "T103 has a data quality problem. Determine the appropriate category "
        "from the knowledge base and update the ticket."
    )
    print(result)

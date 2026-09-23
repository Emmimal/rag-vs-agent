"""
environment.py — a tiny, deterministic ticket workspace. This is the
"real world" the agent acts on. No corpus, no retrieval, no knowledge
of RAG — just state and actions that mutate it.
"""

VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
VALID_STATUSES = {"open", "in_progress", "resolved", "closed"}

# categories a knowledge-dependent task might assign, sourced from the
# corpus at task-resolution time rather than hardcoded here
VALID_CATEGORIES = {"retrieval", "agentic-system", "data-quality", "code-quality", "general"}


class Environment:
    def __init__(self):
        self.tickets = {
            "T101": {
                "title": "RAG system returns irrelevant chunks for chunking questions",
                "status": "open",
                "priority": "normal",
                "category": None,
                "assigned_to": None,
                "knowledge_required": True,
            },
            "T102": {
                "title": "Agent keeps retrying the same failed action in a loop",
                "status": "open",
                "priority": "normal",
                "category": None,
                "assigned_to": None,
                "knowledge_required": True,
            },
            "T103": {
                "title": "Dataset has missing values causing model training to fail",
                "status": "open",
                "priority": "normal",
                "category": None,
                "assigned_to": None,
                "knowledge_required": True,
            },
            "T104": {
                "title": "Script fails with an unhandled exception when processing malformed input",
                "status": "open",
                "priority": "normal",
                "category": None,
                "assigned_to": None,
                "knowledge_required": True,
            },
        }
        self.log = []

    def _record(self, action, ok, detail):
        self.log.append({"action": action, "ok": ok, "detail": detail})
        return ok

    def assign_ticket(self, ticket_id, assignee):
        if ticket_id not in self.tickets:
            return self._record("assign_ticket", False, f"unknown ticket {ticket_id}")
        self.tickets[ticket_id]["assigned_to"] = assignee
        return self._record("assign_ticket", True, f"{ticket_id} -> {assignee}")

    def change_priority(self, ticket_id, priority):
        if ticket_id not in self.tickets:
            return self._record("change_priority", False, f"unknown ticket {ticket_id}")
        if priority not in VALID_PRIORITIES:
            return self._record("change_priority", False, f"invalid priority {priority}")
        self.tickets[ticket_id]["priority"] = priority
        return self._record("change_priority", True, f"{ticket_id} -> {priority}")

    def change_status(self, ticket_id, status):
        if ticket_id not in self.tickets:
            return self._record("change_status", False, f"unknown ticket {ticket_id}")
        if status not in VALID_STATUSES:
            return self._record("change_status", False, f"invalid status {status}")
        self.tickets[ticket_id]["status"] = status
        return self._record("change_status", True, f"{ticket_id} -> {status}")

    def set_category(self, ticket_id, category):
        if ticket_id not in self.tickets:
            return self._record("set_category", False, f"unknown ticket {ticket_id}")
        self.tickets[ticket_id]["category"] = category
        return self._record("set_category", True, f"{ticket_id} -> {category}")

    def get_ticket(self, ticket_id):
        return self.tickets.get(ticket_id)

    def snapshot(self):
        return {k: dict(v) for k, v in self.tickets.items()}


if __name__ == "__main__":
    env = Environment()
    print("Before:", env.get_ticket("T101"))
    env.change_priority("T101", "high")
    env.assign_ticket("T101", "alice")
    print("After: ", env.get_ticket("T101"))
    print("Log:", env.log)
    print("Invalid priority test:", env.change_priority("T101", "urgent-ish"))

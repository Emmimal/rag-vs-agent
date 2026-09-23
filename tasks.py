"""
tasks.py — the frozen 9-task set. 3 knowledge-only, 3 action-only,
3 knowledge+action. Each task records the prompt and the expected
final environment state (not just the prompt), so a run can be
verified against ground truth rather than trusting each system's own
self-report.

`state_changes: {}` means the task should NOT mutate the environment
at all - useful for catching a system that acts when it shouldn't.
"""

TASKS = [
    # --- A: knowledge-only ---
    {
        "id": "A1",
        "category": "knowledge",
        "prompt": "What are the main chunking strategies discussed in the RAG material?",
        "expected": {"answer_required": True, "state_changes": {}},
    },
    {
        "id": "A2",
        "category": "knowledge",
        "prompt": "What approaches does the Data Science material describe for handling missing values?",
        "expected": {"answer_required": True, "state_changes": {}},
    },
    {
        "id": "A3",
        "category": "knowledge",
        "prompt": "What is the difference between try, except, else, and finally in Python?",
        "expected": {"answer_required": True, "state_changes": {}},
    },

    # --- B: action-only ---
    {
        "id": "B1",
        "category": "action",
        "prompt": "Assign T101 to Alice and set its priority to high.",
        "expected": {"state_changes": {"T101.assigned_to": "Alice", "T101.priority": "high"}},
    },
    {
        "id": "B2",
        "category": "action",
        "prompt": "Change T101's status to resolved.",
        "expected": {"state_changes": {"T101.status": "resolved"}},
    },
    {
        "id": "B3",
        "category": "action",
        "prompt": "Move T103 to in_progress status and assign it to Priya.",
        "expected": {"state_changes": {"T103.status": "in_progress", "T103.assigned_to": "Priya"}},
    },

    # --- C: knowledge + action ---
    {
        "id": "C1",
        "category": "knowledge_action",
        "prompt": ("T101 is returning irrelevant chunks. Determine the appropriate "
                   "category from the knowledge base and update the ticket."),
        "expected": {"state_changes": {"T101.category": "retrieval"}},
    },
    {
        "id": "C2",
        "category": "knowledge_action",
        "prompt": ("T103 has a data quality problem related to missing values in the "
                   "dataset. Determine the appropriate category using the knowledge "
                   "base and update the ticket."),
        "expected": {"state_changes": {"T103.category": "data-quality"}},
    },
    {
        "id": "C3",
        "category": "knowledge_action",
        "prompt": ("T104 is failing due to an unhandled exception in the processing "
                   "script. Determine the appropriate category from the knowledge "
                   "base and update the ticket."),
        "expected": {"state_changes": {"T104.category": "code-quality"}},
    },
]

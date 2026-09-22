# rag-vs-agent

A pure-Python experiment: RAG, a deterministic agent, and a hybrid system tested against the same nine tasks — showing exactly where retrieval stops and action starts.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Most "agentic RAG" writeups never actually test whether the action step uses what was retrieved. This repo does: three small systems, one shared ticket environment, one shared 334-chunk knowledge base, and a ground-truth check that verifies actual state changes instead of trusting each system's own self-report.

Read the full write-up on Towards Data Science → [RAG Isn't an Agent — I Built the Layer Between Retrieval and Action](https://towardsdatascience.com/author/emmimalp.alexander/)

## What It Does

```
                    Task Text
                        │
        ┌───────────────┼───────────────────┐
        ▼               ▼                   ▼
   RAG System      Agent (standalone)   RAG + Agent (Hybrid)
        │               │                   │
        ▼               ▼                   ▼
   Retriever      AgentPlanner          Retriever
        │               │                   │
        ▼               ▼                   ▼
   Corpus          Environment            Corpus
 (334 chunks)      (ticket store)      (334 chunks)
        │               │                   │
        ✕               ✕            retrieved_context
   no environment   no corpus              │
      access          access               ▼
                                       AgentPlanner
                                            │
                                            ▼
                                       Environment
                                      (ticket store)
```

Five files, one job each:

| Component | File | Job |
|---|---|---|
| Retriever | `retrieval.py` | TF-IDF + cosine similarity over the corpus. No embeddings, no vector DB. |
| Environment | `environment.py` | A deterministic ticket store — 4 tickets, 4 actions, nothing silent or partial. |
| Agent Planner | `agent_system.py` | Regex-based planner that acts on the environment. Never imports the retriever. |
| RAG System | `rag_system.py` | Retrieves and answers. Never imports the environment — explicitly refuses action requests. |
| Hybrid System | `hybrid_system.py` | The only file that imports both. Retrieves first, then passes context into the planner explicitly. |

No API calls, no external LLM, no vector database, no embedding service. Standard library only.

## Installation

```bash
git clone https://github.com/Emmimal/rag-vs-agent.git
cd rag-vs-agent
```

No dependencies to install. Everything runs on the Python standard library.

## Quick Start

```python
from environment import Environment
from agent_system import AgentPlanner
from rag_system import RAGSystem
from hybrid_system import HybridSystem

# RAG only — can find information, cannot act
rag = RAGSystem()
result = rag.run("What are the main chunking strategies discussed in the RAG material?")
print(result["result"])  # PASS

# Agent only — can act, has no corpus access
env = Environment()
agent = AgentPlanner(env)
result = agent.run("Assign T101 to Alice and set its priority to high.")
print(result["result"])              # PASS
print(env.get_ticket("T101"))

# Hybrid — retrieves first, then acts on what it found
env = Environment()
hybrid = HybridSystem(environment=env)
result = hybrid.run(
    "T101 is returning irrelevant chunks. Determine the appropriate "
    "category from the knowledge base and update the ticket."
)
print(result["result"])                      # PASS
print(env.get_ticket("T101")["category"])    # retrieval
```

## Running the Experiment

```bash
python3 run_experiment.py
```

Runs the frozen 9-task set (`tasks.py`) against all three systems on a fresh environment each time. Every result is checked against the expected final ticket state — not trusted from what the system reports about itself — and any disagreement between the two is flagged explicitly (`>>> MISMATCH`).

```bash
python3 first_three_tasks.py
```

A smaller sanity check — one task per category — if you want to see the shape of it without all 27 executions.

### The Nine Tasks

| Category | Tasks | Example | What it requires |
|---|---|---|---|
| Knowledge-only | A1–A3 | "What are the main chunking strategies discussed in the RAG material?" | Corpus answer, zero ticket mutation |
| Action-only | B1–B3 | "Assign T101 to Alice and set its priority to high." | Field updates, no corpus lookup |
| Knowledge + action | C1–C3 | "T101 is returning irrelevant chunks. Determine the appropriate category from the knowledge base and update the ticket." | Retrieve first, then write the resolved category |

## Results

27 executions (9 tasks × 3 systems), each verified against the frozen expected ticket state:

| ID | Category | RAG | Agent | Hybrid |
|---|---|---|---|---|
| A1–A3 | Knowledge | PASS | FAIL | PASS |
| B1–B3 | Action | FAIL | PASS | PASS |
| C1–C3 | Knowledge + action | FAIL | FAIL | PASS |

Zero disagreements between self-reported and independently verified ground truth in the final run. That wasn't true on the first two attempts — see [`dev_history/`](./dev_history) for the actual before-and-after terminal output of two real bugs found along the way:

1. **The first hybrid planner assumed every task was a ticket operation.** A pure knowledge question failed even though retrieval had already found the right answer. Fixed by checking whether an action is required before assuming a ticket exists.
2. **Two action tasks failed on phrasing, not architecture.** `"assign T101 to Alice"` didn't match a regex written for `"assign to Alice"`, and the planner reported `PASS` anyway because it only checks the steps it *did* run, never whether it extracted every instruction in the task. Fixed by broadening the two regexes to the phrasing actually used, then rerunning the complete matrix — not just the two failing tasks.

## Project Structure

```
rag-vs-agent/
├── LICENSE
├── README.md
├── corpus/
│   └── chunks.json            # 334 chunks from 22 real EmiTechLogic articles
├── retrieval.py                 # TF-IDF + cosine similarity retriever
├── environment.py                # deterministic ticket store
├── agent_system.py                # rule-based planner — no corpus access
├── rag_system.py                   # retrieval only — no environment access
├── hybrid_system.py                 # retrieval feeds context into the planner
├── tasks.py                          # the frozen 9-task set with expected outcomes
├── run_experiment.py                  # runs all 9 tasks x 3 systems, verifies ground truth
├── first_three_tasks.py                # smaller 3-task sanity check
└── dev_history/                         # terminal output from two real bugs, before and after
    ├── task_a_prefix_hybrid_failure.txt
    ├── task_abc_postfix_clean.txt
    ├── full_9task_run.txt
    ├── full_9task_run_corrected.txt
    └── full_9task_run_FINAL.txt
```

## Performance (CPU only, 334-chunk corpus, mean of 200 runs)

| Operation | Latency | Notes |
|---|---|---|
| Retriever init (load + build TF-IDF index) | ~29.4 ms | Pays the full indexing cost once |
| Retrieval query (index already built) | ~1.4 ms | What one `search()` call actually costs |
| Environment action | ~0.001 ms | Dict lookup and assignment |
| Agent planner full run | ~0.009 ms | No corpus involved |
| RAG system full run | ~1.3 ms | Retrieval only |
| Hybrid full run, index reused | ~1.5 ms | Retrieve + resolve category + mutate ticket |

`run_experiment.py` builds a fresh `HybridSystem` per task for environment isolation, which rebuilds the TF-IDF index every time — about 20x the cost of actually running a query against an index that's already built. In a real deployment you'd build the retriever once and reuse it across requests; the 1.4–1.5ms numbers above are what that looks like.

## When to Use This

Worth it if you want to see exactly where retrieval and action need to be explicitly connected rather than assumed to come together automatically, you're evaluating whether an "agentic RAG" claim actually uses what it retrieved, or you want a small, fully-readable reference with no framework layer hiding the decision logic.

Skip it if you need production-grade retrieval — this uses TF-IDF specifically so every result stays explainable, not because it's state of the art — or a general-purpose action parser, since the planner here only recognizes the phrasings that were actually tested.

## Known Limitations

- **TF-IDF, not semantic retrieval.** No sense of synonymy — "unhandled exception" and "uncaught error" are unrelated to it. This is also the direct cause of the noisy top-1 retrieval seen on the knowledge+action tasks (documented in the write-up).
- **Regex parsing, not a real intent parser.** The planner only recognizes the phrasings it was tested against. New phrasing needs a new pattern.
- **No confidence threshold on category resolution.** A 2-1 majority vote and a 3-0 unanimous vote are treated identically.
- **Nine tasks, one corpus, one author's writing style.** Enough to demonstrate the retrieval/action boundary and catch two real bugs — not a sample size to generalize a hit rate from.

## License

MIT — see [LICENSE](./LICENSE).

"""Build the worked examples shown to the Session Judge's conversation check.

Examples never come from the held-out part of the multi-turn evaluation (split.json):
  attacks  SafeMTData Attack_600 sessions, one per harm category plus 2 more, taken from harmful queries
           (query_id) used by neither part; for a category with no such query left, from a dev-part query
           (those dev sessions are then excluded from dev measurements by multiturn_eval.py). Each example
           keeps the hidden harmful goal (`plain_query`) so the model sees what the chain was building towards.
  benign   no_robots Chat conversations that are in neither part.
Output: data/multiturn_examples.json (read by UnsafeContentDetectorAgent in conversation mode).

  python build_multiturn_examples.py
"""

import json
import random
from collections import defaultdict

from common import PROJECT_ROOT, _load, write_json
from multiturn_eval import load_split

SEED = 1
N_ATTACK, N_BENIGN = 8, 4
MAX_TURN_CHARS = 220
OUT = PROJECT_ROOT / "data" / "multiturn_examples.json"


def _short(t):
    t = " ".join(str(t).split())
    return t if len(t) <= MAX_TURN_CHARS else t[:MAX_TURN_CHARS - 3] + "..."


def main():
    rng = random.Random(SEED)
    sessions, split = load_split()
    used_queries = {s["query_id"] for s in sessions if s["label"] == 1}
    used_ids = set(split)

    rows = [dict(r) for r in _load("SafeMTData/SafeMTData", "Attack_600")["Attack_600"]]
    heldout_queries = {x["query_id"] for x in sessions if x["label"] == 1 and split[x["id"]] == "heldout"}
    dev_queries = used_queries - heldout_queries
    free = [r for r in rows if r["query_id"] not in used_queries]
    dev_pool = [r for r in rows if r["query_id"] in dev_queries]
    picked, seen_q = [], set()
    for cat in sorted({r["category"] for r in rows}):
        for pool in (free, dev_pool):
            options = [x for x in pool if x["category"] == cat and x["query_id"] not in seen_q]
            if options:
                r = rng.choice(sorted(options, key=lambda x: x["id"]))
                picked.append(r)
                seen_q.add(r["query_id"])
                break
    rest = [r for r in free if r["query_id"] not in seen_q]
    rng.shuffle(rest)
    for r in rest:
        if len(picked) == N_ATTACK:
            break
        if r["query_id"] not in seen_q:
            picked.append(r)
            seen_q.add(r["query_id"])

    nr = _load("HuggingFaceH4/no_robots")
    benign = []
    for sp in ("train", "test"):
        for i, r in enumerate(nr[sp]):
            sid = f"nr-{sp}-{i}"
            turns = [m["content"] for m in r["messages"] if m["role"] == "user" and m["content"].strip()]
            if sid not in used_ids and 3 <= len(turns) <= 5:
                benign.append({"id": sid, "turns": turns})
    rng.shuffle(benign)

    out = {
        "source": "SafeMTData Attack_600 (queries not in split.json) and no_robots Chat (sessions not in split.json)",
        "attacks": [{"query_id": r["query_id"], "category": r["category"], "goal": _short(r["plain_query"]),
                     "turns": [_short(t) for t in r["multi_turn_queries"] if str(t).strip()]} for r in picked],
        "benign": [{"id": b["id"], "turns": [_short(t) for t in b["turns"]]} for b in benign[:N_BENIGN]],
    }
    example_queries = {a["query_id"] for a in out["attacks"]}
    assert not (example_queries & heldout_queries), "an example shares a query with the held-out part"
    assert not ({b["id"] for b in out["benign"]} & used_ids)
    out["dev_queries_used_as_examples"] = sorted(example_queries & dev_queries)
    write_json(OUT, out)
    print(f"wrote {len(out['attacks'])} attack + {len(out['benign'])} benign examples to {OUT}")
    print("categories:", sorted(a["category"] for a in out["attacks"]))


if __name__ == "__main__":
    main()

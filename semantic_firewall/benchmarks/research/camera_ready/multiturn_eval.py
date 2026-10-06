"""Multi-turn evaluation with adversarial AND benign conversations, so a multi-turn FPR can be reported.

Data (fixed seed, written to results_camera_ready/multiturn/split.json):
  attacks  SafeMTData/SafeMTData "Attack_600" (ActorAttack multi-turn jailbreaks; 3 actor variants per
           harmful query). One session per query, stratified by harm category; user turns = `multi_turn_queries`.
  benign   HuggingFaceH4/no_robots "Chat" conversations with >= 2 user turns (train + test); user turns only.
Parts: "heldout" = 100 attack + 100 benign sessions (report this); "dev" = a disjoint 50 + 50 used for any
tuning or inspection. No harmful query (query_id) appears in both parts.

Each session is sent turn by turn with one session_id. A session is detected when any turn's action is
BLOCK, FLAG or REDACT (same rule as decision_record). Reported per class: session recall / FPR, precision,
F1, the turn of first detection, LLM calls and LLM failures.

  python multiturn_eval.py --part dev --config fast_layers        # free dry run, no LLM
  python multiturn_eval.py --part heldout --config full           # paper numbers
"""

import argparse
import json
import random
import time
import uuid
from collections import Counter, defaultdict

from common import (CONFIGS, OUT_ROOT, PROJECT_ROOT, _load, build_firewall, decision_record, quiet_stdout,
                    run_metadata, write_json)

N_HELDOUT = 100
N_DEV = 50
SEED = 0
SPLIT_PATH = OUT_ROOT / "multiturn" / "split.json"


def _stratified(rows, key, n, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    for g in groups.values():
        rng.shuffle(g)
    picked, order = [], sorted(groups)
    while len(picked) < n and any(groups.values()):
        for k in order:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


def build_sessions():
    rng = random.Random(SEED)
    attack_rows = [dict(r) for r in _load("SafeMTData/SafeMTData", "Attack_600")["Attack_600"]]
    # one session per harmful query (pick one actor variant), so the two parts never share a query
    by_query = defaultdict(list)
    for r in attack_rows:
        by_query[r["query_id"]].append(r)
    one_per_query = [rng.choice(sorted(v, key=lambda x: x["id"])) for _, v in sorted(by_query.items())]
    picked = _stratified(one_per_query, "category", N_HELDOUT + N_DEV, rng)
    attacks = [{"id": f"safemt-{r['id']}", "label": 1, "category": r["category"], "query_id": r["query_id"],
                "turns": [t for t in r["multi_turn_queries"] if t and t.strip()]} for r in picked]

    nr = _load("HuggingFaceH4/no_robots")
    benign_rows = []
    for split in ("train", "test"):
        for i, r in enumerate(nr[split]):
            user_turns = [m["content"] for m in r["messages"] if m["role"] == "user" and m["content"].strip()]
            if len(user_turns) >= 2:
                benign_rows.append({"id": f"nr-{split}-{i}", "label": 0, "category": "chat", "turns": user_turns})
    rng.shuffle(benign_rows)
    benign = benign_rows[:N_HELDOUT + N_DEV]

    split = {}
    for cls in (attacks, benign):
        ids = [s["id"] for s in cls]
        rng.shuffle(ids)
        split.update({i: "heldout" for i in ids[:N_HELDOUT]})
        split.update({i: "dev" for i in ids[N_HELDOUT:]})
    return attacks + benign, split


def load_split():
    sessions, split = build_sessions()
    if SPLIT_PATH.exists():
        saved = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        if saved["assignment"] != split:
            raise SystemExit("split.json differs from the regenerated split; refusing to continue")
    else:
        write_json(SPLIT_PATH, {"seed": SEED, "n_heldout_per_class": N_HELDOUT, "n_dev_per_class": N_DEV,
                                "assignment": split})
    return sessions, split


def summarize(records):
    att = [r for r in records if r["label"] == 1]
    ben = [r for r in records if r["label"] == 0]
    tp = sum(r["detected"] for r in att)
    fp = sum(r["detected"] for r in ben)
    rec = tp / len(att) if att else 0.0
    fpr = fp / len(ben) if ben else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    first = [r["first_detection_turn"] for r in att if r["detected"]]
    return {
        "sessions_attack": len(att), "sessions_benign": len(ben),
        "tp": tp, "fn": len(att) - tp, "fp": fp, "tn": len(ben) - fp,
        "session_recall": rec, "session_fpr": fpr, "precision": prec, "f1": f1,
        "mean_first_detection_turn": sum(first) / len(first) if first else None,
        "mean_turns_attack": sum(r["n_turns"] for r in att) / len(att) if att else None,
        "detected_by_category": dict(Counter(r["category"] for r in att if r["detected"])),
        "attacks_by_category": dict(Counter(r["category"] for r in att)),
        "llm_calls": sum(r["llm_calls"] for r in records),
        "sessions_with_llm_failure": sum(r["llm_failed"] for r in records),
        "benign_flag_reasons": dict(Counter(a for r in ben if r["detected"] for a in r["first_detection_agents"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["dev", "heldout"], required=True)
    parser.add_argument("--config", default="full", choices=sorted(CONFIGS))
    parser.add_argument("--cache", default="warm_frozen", choices=["cold", "warm_frozen", "warm_online"])
    parser.add_argument("--max-sessions", type=int, default=0, help="smoke test only")
    parser.add_argument("--tag", default="", help="suffix for the run directory")
    parser.add_argument("--no-session", action="store_true",
                        help="ablation: analyse every turn on its own (no session id, so no Session Judge)")
    args = parser.parse_args()

    sessions, split = load_split()
    sessions = [s for s in sessions if split[s["id"]] == args.part]
    examples = PROJECT_ROOT / "data" / "multiturn_examples.json"
    if examples.exists():  # sessions whose harmful query is shown as a worked example are not measured
        ex_queries = {a["query_id"] for a in json.loads(examples.read_text(encoding="utf-8"))["attacks"]}
        if args.part == "heldout" and any(s.get("query_id") in ex_queries for s in sessions):
            raise SystemExit("a held-out session shares its query with a worked example")
        sessions = [s for s in sessions if s.get("query_id") not in ex_queries]
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
    run_name = f"multiturn-{args.part}-{args.config}-{args.cache}" + (f"-{args.tag}" if args.tag else "")
    fw, env, seeded = build_firewall(CONFIGS[args.config], run_name, cache_protocol=args.cache)

    out_dir = OUT_ROOT / "multiturn" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / "sessions.jsonl").open("w", encoding="utf-8") as fh:
        for n, s in enumerate(sessions, 1):
            sid = str(uuid.uuid4())
            turns, detected_at, agents = [], None, []
            for t, text in enumerate(s["turns"], 1):
                started = time.perf_counter()
                with quiet_stdout():
                    decision = fw.analyze(text, session_id=None if args.no_session else sid)
                rec = decision_record({"id": f"{s['id']}#{t}", "label": s["label"], "text": text}, decision,
                                      (time.perf_counter() - started) * 1000)
                turns.append(rec)
                if rec["pred"] and detected_at is None:
                    detected_at, agents = t, rec["triggered_agents"]
            r = {"id": s["id"], "label": s["label"], "category": s["category"], "n_turns": len(s["turns"]),
                 "detected": detected_at is not None, "first_detection_turn": detected_at,
                 "first_detection_agents": agents,
                 "llm_calls": sum(x["llm_called"] for x in turns), "llm_failed": any(x["llm_failed"] for x in turns),
                 "turns": turns}
            records.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(sessions)} sessions")

    summary = {"part": args.part, "config": args.config, "cache_protocol": args.cache, "seeded_cache": seeded,
               "session_judge": not args.no_session,
               **summarize(records), "env": env, "meta": run_metadata()}
    write_json(out_dir / "summary.json", summary)
    keys = ("session_recall", "session_fpr", "precision", "f1", "mean_first_detection_turn",
            "llm_calls", "sessions_with_llm_failure")
    print(json.dumps({k: summary[k] for k in keys}, indent=1))
    print("benign flags by agent:", summary["benign_flag_reasons"])
    print("detected by category:", summary["detected_by_category"], "of", summary["attacks_by_category"])


if __name__ == "__main__":
    main()

"""PII detector on ai4privacy/pii-masking-200k: all rows, all 12 target entity types.

Matching is at the value level (lower-cased, stripped). Two protocols are reported:

  legacy       FP = every detected value that is not a ground-truth value of the 12 target
               types (the protocol of the original submission). It counts real PII
               of other types (names, ages, ...) as false positives.
  type_mapped  only detections whose detector type maps to one of the 12 target types are
               scored; a detection that equals a ground-truth value of a non-target type is
               "out of scope", not a false positive.

Per-type recall and a per-language breakdown are also written. Runs offline in ~10 minutes.
  python pii_eval.py                  # all rows
  python pii_eval.py --part dev       # rows with index % 5 == 0 (used for error analysis)
  python pii_eval.py --part heldout   # the other 80% (never inspected; report this)
"""

import argparse
import time
from collections import defaultdict

from common import OUT_ROOT, quiet_stdout, run_metadata, write_json

TARGET_LABELS = {"EMAIL", "IPV4", "IPV6", "MAC", "PHONENUMBER", "IBAN", "DOB", "CREDITCARDNUMBER",
                 "CREDITCARDCVV", "SSN", "ZIPCODE", "ACCOUNTNUMBER"}
TYPE_MAP = {
    "email": "EMAIL", "ipv4": "IPV4", "ipv6": "IPV6", "mac_address": "MAC",
    "phone_india": "PHONENUMBER", "phone_us": "PHONENUMBER", "phone_uk": "PHONENUMBER", "phone_generic": "PHONENUMBER",
    "iban": "IBAN", "date_of_birth": "DOB",
    "credit_card_visa": "CREDITCARDNUMBER", "credit_card_mastercard": "CREDITCARDNUMBER",
    "credit_card_amex": "CREDITCARDNUMBER", "credit_card_generic": "CREDITCARDNUMBER",
    "cvv": "CREDITCARDCVV", "ssn_us": "SSN", "zipcode_us": "ZIPCODE", "pincode_india": "ZIPCODE",
    "bank_account_india": "ACCOUNTNUMBER", "account_number": "ACCOUNTNUMBER",
    # ai4privacy profile (SEMANTIC_FIREWALL_PII_PROFILE=ai4privacy)
    "account_number_bare": "ACCOUNTNUMBER", "ssn_ch_ahv": "SSN",
}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--part", choices=["all", "dev", "heldout"], default="all")
    parser.add_argument("--profile", choices=["default", "ai4privacy"], default="default",
                        help="ai4privacy adds the dataset-adapted patterns (bare 8-digit accounts, 756-prefixed SSNs)")
    args = parser.parse_args()

    import os
    from datasets import load_dataset
    from semantic_firewall.core.agents.pii_detector import PIIDetectorAgent

    os.environ["SEMANTIC_FIREWALL_PII_PROFILE"] = "" if args.profile == "default" else args.profile

    rows = load_dataset("ai4privacy/pii-masking-200k", "default")["train"]
    if args.part != "all":
        want_dev = args.part == "dev"
        rows = rows.select([i for i in range(len(rows)) if (i % 5 == 0) == want_dev])
    if args.max_rows:
        rows = rows.select(range(args.max_rows))
    agent = PIIDetectorAgent()

    legacy = defaultdict(int)
    mapped = defaultdict(int)
    out_of_scope = 0
    per_type = defaultdict(lambda: {"tp": 0, "fn": 0})
    per_lang = defaultdict(lambda: defaultdict(int))
    latencies = []
    for item in rows:
        gt_all = {}
        for mask in item.get("privacy_mask") or []:
            gt_all[mask["value"].strip().lower()] = mask["label"]
        gt_target = {v: l for v, l in gt_all.items() if l in TARGET_LABELS}

        started = time.perf_counter()
        with quiet_stdout():
            result = agent.run(item["source_text"])
        latencies.append((time.perf_counter() - started) * 1000)
        found = {m.value.strip().lower(): m.pii_type for m in result.matched}

        # legacy protocol (all detections, target ground truth)
        legacy["tp"] += len(gt_target.keys() & found.keys())
        legacy["fp"] += len(found.keys() - gt_target.keys())
        legacy["fn"] += len(gt_target.keys() - found.keys())

        # type-mapped protocol
        scored = {v for v, t in found.items() if t in TYPE_MAP}
        tp_vals = scored & gt_target.keys()
        oos = {v for v in scored - gt_target.keys() if v in gt_all}
        fp_vals = scored - gt_target.keys() - oos
        fn_vals = gt_target.keys() - found.keys()
        mapped["tp"] += len(tp_vals)
        mapped["fp"] += len(fp_vals)
        mapped["fn"] += len(fn_vals)
        out_of_scope += len(oos)
        lang = item.get("language", "?")
        per_lang[lang]["tp"] += len(tp_vals)
        per_lang[lang]["fp"] += len(fp_vals)
        per_lang[lang]["fn"] += len(fn_vals)
        per_lang[lang]["rows"] += 1
        for v in gt_target:
            per_type[gt_target[v]]["tp" if v in found else "fn"] += 1

    latencies.sort()
    result = {
        "dataset": "ai4privacy/pii-masking-200k (train, all languages)",
        "part": args.part,
        "profile": args.profile,
        "rows": len(rows),
        "ground_truth_entities_target_types": legacy["tp"] + legacy["fn"],
        "legacy": prf(legacy["tp"], legacy["fp"], legacy["fn"]),
        "type_mapped": {**prf(mapped["tp"], mapped["fp"], mapped["fn"]), "out_of_scope_true_pii": out_of_scope},
        "per_type_recall": {t: {**v, "recall": v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0}
                            for t, v in sorted(per_type.items())},
        "per_language_type_mapped": {l: {"rows": v["rows"], **prf(v["tp"], v["fp"], v["fn"])}
                                     for l, v in sorted(per_lang.items())},
        "latency_ms": {"mean": sum(latencies) / len(latencies), "p50": latencies[len(latencies) // 2],
                       "p95": latencies[int(len(latencies) * 0.95) - 1], "p99": latencies[int(len(latencies) * 0.99) - 1]},
        "meta": run_metadata(),
    }
    suffix = ("" if args.part == "all" else f"_{args.part}") + ("" if args.profile == "default" else f"_{args.profile}")
    write_json(OUT_ROOT / "pii" / f"pii_eval{suffix}.json", result)
    for key in ("legacy", "type_mapped"):
        r = result[key]
        print(f"  {key:12s} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")
    for t, v in result["per_type_recall"].items():
        print(f"  {t:18s} recall={v['recall']:.4f} (tp={v['tp']}, fn={v['fn']})")
    print(f"  latency p50={result['latency_ms']['p50']:.2f}ms p99={result['latency_ms']['p99']:.2f}ms")


if __name__ == "__main__":
    main()

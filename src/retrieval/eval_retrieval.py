#!/usr/bin/env python3
"""
Usage:
    python -m src.retrieval.eval_retrieval --dataset A  [Task A args...]
    python -m src.retrieval.eval_retrieval --dataset B  [Task B args...]

- --dataset A / B  : selects the QFR task; --data_path is the dataset file path.
- --mode, --index_dir, --model_key, --device, --k_values, --seed, --out_json,
  --trace_dir : shared across Task A and Task B.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .determinism import set_deterministic
from .retrieval import BM25Retriever, DenseRetriever

logger = logging.getLogger(__name__)


# Shared IO helpers

def iter_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_open_trace(trace_dir: Optional[str]) -> Tuple[Any, Optional[str]]:
    if not trace_dir:
        return None, None
    d = Path(trace_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "retrieval_traces.jsonl"
    return p.open("w", encoding="utf-8"), p.as_posix()


def parse_k_values(s: str) -> List[int]:
    if not s or not s.strip():
        return [1, 3, 5, 10, 20]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [1, 3, 5, 10, 20]


def build_retriever(mode: str, index_dir: str, model_key: Optional[str], device: Optional[str]) -> Any:
    if mode == "bm25":
        return BM25Retriever(index_dir)
    else:
        if not model_key:
            raise SystemExit("--model_key is required for dense mode.")
        return DenseRetriever(model_key=model_key, index_dir=index_dir, device=device)


def build_run_name(mode: str, index_dir: str, model_key: Optional[str], suffix: str = "") -> str:
    base = (
        f"bm25::{Path(index_dir).as_posix()}"
        if mode == "bm25"
        else f"dense::{model_key}::{Path(index_dir).as_posix()}"
    )
    return base + (f"::{suffix}" if suffix else "")


# Shared metric helpers

def mrr_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k_binary(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_set:
            dcg += 1.0 / np.log2(i + 1)
    m = min(len(gold_set), k)
    if m == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, m + 1))
    return float(dcg / idcg)


# Task A

def gold_to_full_chunk_ids(gold_evidence: List[Dict[str, str]]) -> List[str]:
    out = []
    for e in gold_evidence or []:
        doc_id = e.get("doc_id")
        chunk_id = e.get("chunk_id")
        if doc_id and chunk_id:
            out.append(f"{doc_id}_{chunk_id}")
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def hit_recall_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    for cid in ranked_ids[:k]:
        if cid in gold_set:
            return 1.0
    return 0.0


def set_recall_at_k(ranked_ids: List[str], gold_set: Set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return float(len(retrieved.intersection(gold_set)) / len(gold_set))


def evaluate_taskA(args: argparse.Namespace) -> None:
    """Replicate eval_taskA_retrieval.py behaviour."""
    set_deterministic(args.seed, deterministic_torch=False)

    k_values = parse_k_values(args.k_values)
    if any(k <= 0 for k in k_values):
        raise SystemExit("All k must be positive integers.")

    retriever = build_retriever(args.mode, args.index_dir, args.model_key, args.device)
    run_name = build_run_name(args.mode, args.index_dir, args.model_key)

    trace_fp, traces_path = _safe_open_trace(args.trace_dir)

    rows = iter_jsonl(args.data_path)
    logger.info(f"Loaded {len(rows)} questions from {args.data_path}")

    per_query: List[Dict[str, Any]] = []
    agg: Dict[str, List[float]] = {f"MRR@{k}": [] for k in k_values}
    agg.update({f"HitRecall@{k}": [] for k in k_values})
    agg.update({f"SetRecall@{k}": [] for k in k_values})
    agg.update({f"nDCG@{k}": [] for k in k_values})
    agg["latency_sec"] = []

    try:
        for r in rows:
            qid = r.get("id", "")
            question = r.get("question", "")
            gold_full = gold_to_full_chunk_ids(r.get("gold_evidence", []))
            gold_set = set(gold_full)

            if not question or not gold_set:
                per_query.append({
                    "id": qid,
                    "skipped": True,
                    "reason": "missing question or gold_evidence",
                })
                continue

            k_max = max(k_values)
            t0 = time.time()
            hits = retriever.search(question, k=k_max)
            elapsed = time.time() - t0
            ranked_ids = [cid for (cid, _score) in hits]

            pq: Dict[str, Any] = {
                "id": qid,
                "question": question,
                "num_gold": len(gold_set),
                "latency_sec": float(elapsed),
            }

            for k in k_values:
                pq[f"MRR@{k}"] = mrr_at_k(ranked_ids, gold_set, k)
                pq[f"HitRecall@{k}"] = hit_recall_at_k(ranked_ids, gold_set, k)
                pq[f"SetRecall@{k}"] = set_recall_at_k(ranked_ids, gold_set, k)
                pq[f"nDCG@{k}"] = ndcg_at_k_binary(ranked_ids, gold_set, k)

                agg[f"MRR@{k}"].append(pq[f"MRR@{k}"])
                agg[f"HitRecall@{k}"].append(pq[f"HitRecall@{k}"])
                agg[f"SetRecall@{k}"].append(pq[f"SetRecall@{k}"])
                agg[f"nDCG@{k}"].append(pq[f"nDCG@{k}"])
            agg["latency_sec"].append(float(elapsed))

            if trace_fp:
                trace_record = {
                    "query_id": qid,
                    "question": question,
                    "run": run_name,
                    "mode": args.mode,
                    "model_key": args.model_key,
                    "index_dir": Path(args.index_dir).as_posix(),
                    "seed": int(args.seed),
                    "retrieve_k": int(k_max),
                    "k_values": k_values,
                    "gold_chunk_ids": sorted(gold_set),
                    "retrieved": [
                        {"rank": i + 1, "chunk_id": cid, "score": float(score)}
                        for i, (cid, score) in enumerate(hits)
                    ],
                }
                trace_fp.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            per_query.append(pq)

    finally:
        if trace_fp:
            trace_fp.close()

    summary = {
        "run": run_name,
        "dataset": Path(args.data_path).as_posix(),
        "mode": args.mode,
        "model_key": args.model_key,
        "index_dir": Path(args.index_dir).as_posix(),
        "seed": int(args.seed),
        "n_questions_total": len(rows),
        "n_questions_scored": int(sum(1 for x in per_query if not x.get("skipped"))),
        "k_values": k_values,
        "mean": {m: float(np.mean(v)) if v else 0.0 for m, v in agg.items()},
        "traces_path": traces_path,
    }

    print("\n=== TASK A RETRIEVAL EVAL (binary relevance) ===")
    print(f"run      : {summary['run']}")
    print(f"dataset  : {summary['dataset']}")
    print(f"scored   : {summary['n_questions_scored']}/{summary['n_questions_total']}")
    if summary["traces_path"]:
        print(f"traces   : {summary['traces_path']}")
    print("")
    print(f"mean latency: {summary['mean']['latency_sec']:.4f} sec/query\n")

    for k in k_values:
        print(
            f"@{k:>2}  "
            f"MRR={summary['mean'][f'MRR@{k}']:.4f}  "
            f"HitRecall={summary['mean'][f'HitRecall@{k}']:.4f}  "
            f"SetRecall={summary['mean'][f'SetRecall@{k}']:.4f}  "
            f"nDCG={summary['mean'][f'nDCG@{k}']:.4f}"
        )

    if args.out_json:
        out = {"summary": summary, "per_query": per_query}
        write_json(args.out_json, out)
        print(f"\nWrote: {args.out_json}")


# Task B helpers

def normalize_slot_ids(v: Any) -> Set[str]:
    if v is None:
        return set()
    if isinstance(v, list):
        out: Set[str] = set()
        for x in v:
            s = str(x).strip()
            if s:
                out.add(s)
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return set()
        return {x.strip() for x in s.split(",") if x.strip()}
    return set()


def first_hit_rank_1based(retrieved_ids: List[str], gold_set: Set[str]) -> Optional[int]:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return i
    return None


def first_hit_rank_and_id_1based(
    retrieved_ids: List[str], gold_set: Set[str]
) -> Tuple[Optional[int], Optional[str]]:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return i, cid
    return None, None


@dataclass
class PerKAccum:
    q_recall_sum: float = 0.0
    slot_recall_sum: float = 0.0
    slot_mrr_sum: float = 0.0


def evaluate_taskB_slots(
    dataset_path: str,
    retriever: Any,
    k_values: List[int],
    slot_labels: Optional[List[str]] = None,
    trace_fp: Any = None,
    run_name: str = "",
    seed: int = 0,
    mode: str = "",
    model_key: Optional[str] = None,
    index_dir: str = "",
) -> Dict[str, Any]:
    """Slot-based retrieval evaluation for Task B (identical logic to original)."""
    if slot_labels is None:
        slot_labels = ["slot_A", "slot_B", "slot_C"]

    rows = list(iter_jsonl(dataset_path))
    if not rows:
        raise SystemExit(f"No rows found in dataset: {dataset_path}")

    k_values = sorted(set(int(k) for k in k_values))
    if not k_values:
        raise SystemExit("k_values must be non-empty")
    max_k = max(k_values)

    acc: Dict[int, PerKAccum] = {k: PerKAccum() for k in k_values}
    scored = 0
    skipped = 0
    total_latency = 0.0

    for row in rows:
        qid = row.get("id") or row.get("question_id") or row.get("qid")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            skipped += 1
            logger.warning(f"Skipping row with missing question (id={qid})")
            continue

        required_slots = row.get("required_slots", None)
        try:
            required_slots_n = int(required_slots) if required_slots is not None else None
        except Exception:
            required_slots_n = None

        slots: List[Tuple[str, Set[str]]] = []
        for lab in slot_labels:
            gold = normalize_slot_ids(row.get(lab))
            if gold:
                slots.append((lab, gold))

        if not slots:
            skipped += 1
            logger.warning(f"Skipping row with no non-empty slots (id={qid})")
            continue

        if required_slots_n is not None:
            slots = slots[:required_slots_n]

        t0 = time.time()
        hits = retriever.search(question, max_k)
        elapsed = time.time() - t0
        total_latency += float(elapsed)
        retrieved_ids_all = [cid for (cid, _score) in hits]

        if trace_fp is not None:
            slot_first_hit_rank: Dict[str, Optional[int]] = {}
            slot_first_hit_id: Dict[str, Optional[str]] = {}
            slots_payload: List[Dict[str, Any]] = []

            for lab, gold_set in slots:
                slots_payload.append({"label": lab, "gold_ids": sorted(gold_set)})
                r, hid = first_hit_rank_and_id_1based(retrieved_ids_all, gold_set)
                slot_first_hit_rank[lab] = r
                slot_first_hit_id[lab] = hid

            trace_record = {
                "query_id": qid,
                "question": question,
                "run": run_name,
                "mode": mode,
                "model_key": model_key,
                "index_dir": index_dir,
                "seed": int(seed),
                "retrieve_k": int(max_k),
                "k_values": k_values,
                "required_slots": required_slots_n,
                "slots": slots_payload,
                "slot_first_hit_rank": slot_first_hit_rank,
                "slot_first_hit_id": slot_first_hit_id,
                "retrieved": [
                    {"rank": i + 1, "chunk_id": cid, "score": float(score)}
                    for i, (cid, score) in enumerate(hits)
                ],
            }
            trace_fp.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

        for k in k_values:
            retrieved_ids = retrieved_ids_all[:k]
            satisfied = 0
            rr_sum = 0.0

            for _lab, gold_set in slots:
                r = first_hit_rank_1based(retrieved_ids, gold_set)
                if r is not None:
                    satisfied += 1
                    rr_sum += 1.0 / float(r)

            nslots = len(slots)
            acc[k].q_recall_sum += 1.0 if satisfied == nslots else 0.0
            acc[k].slot_recall_sum += satisfied / float(nslots)
            acc[k].slot_mrr_sum += rr_sum / float(nslots)

        scored += 1

    if scored == 0:
        raise SystemExit("No valid questions scored (all rows skipped).")

    metrics_by_k: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        metrics_by_k[k] = {
            "QuestionRecall": acc[k].q_recall_sum / float(scored),
            "SlotRecall": acc[k].slot_recall_sum / float(scored),
            "SlotMRR": acc[k].slot_mrr_sum / float(scored),
        }

    return {
        "dataset": str(dataset_path),
        "n_rows": len(rows),
        "scored": scored,
        "skipped": skipped,
        "k_values": k_values,
        "metrics": metrics_by_k,
        "mean_latency_sec": total_latency / float(scored) if scored > 0 else 0.0,
    }


def evaluate_taskB(args: argparse.Namespace) -> None:
    """Replicate eval_taskB_retrieval.py behaviour."""
    set_deterministic(args.seed, deterministic_torch=False)

    k_values = parse_k_values(args.k_values)

    retriever = build_retriever(args.mode, args.index_dir, args.model_key, args.device)
    run_name = build_run_name(args.mode, args.index_dir, args.model_key)

    trace_fp, traces_path = _safe_open_trace(args.trace_dir)

    try:
        summary = evaluate_taskB_slots(
            dataset_path=args.data_path,
            retriever=retriever,
            k_values=k_values,
            slot_labels=["slot_A", "slot_B", "slot_C"],
            trace_fp=trace_fp,
            run_name=run_name,
            seed=int(args.seed),
            mode=args.mode,
            model_key=args.model_key,
            index_dir=Path(args.index_dir).as_posix(),
        )
    finally:
        if trace_fp:
            trace_fp.close()

    print("\n=== TASK B RETRIEVAL EVAL (slot-based) ===")
    print(f"run      : {run_name}")
    print(f"dataset  : {summary['dataset']}")
    print(f"scored   : {summary['scored']}/{summary['n_rows']}  (skipped={summary['skipped']})")
    if traces_path:
        print(f"traces   : {traces_path}")
    print("")
    print(f"mean latency: {summary['mean_latency_sec']:.4f} sec/query\n")

    for k in summary["k_values"]:
        m = summary["metrics"][k]
        print(
            f"@{k:<2d}  "
            f"QuestionRecall={m['QuestionRecall']:.4f}  "
            f"SlotRecall={m['SlotRecall']:.4f}  "
            f"SlotMRR={m['SlotMRR']:.4f}"
        )

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": run_name,
            "seed": int(args.seed),
            "traces_path": traces_path,
            **summary,
        }
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote: {outp.as_posix()}")


# Argument parser & dispatcher

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Unified retrieval evaluation CLI for QFR-RAG. "
            "Select task with --dataset {A,B}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Task A
python -m src.retrieval.eval_retrieval --dataset A \\
    --data_path qfr_datasets/taskA.jsonl \\
    --mode bm25 --index_dir indexes/qfr/bm25

# Task B
python -m src.retrieval.eval_retrieval --dataset B \\
    --data_path qfr_datasets/taskB.jsonl \\
    --mode dense --index_dir indexes/qfr/dense --model_key medcpt
""",
    )

    # ---- Task selector ----
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["A", "B"],
        help="Which QFR evaluation to run: A (Task A) or B (Task B).",
    )

    # ---- Dataset path: used by Task A and Task B ----
    ap.add_argument(
        "--data_path",
        default=None,
        help="Path to the JSONL dataset file (required for --dataset A or B).",
    )

    # ---- Shared retrieval arguments ----
    ap.add_argument("--mode", choices=["bm25", "dense"], required=True)
    ap.add_argument("--index_dir", required=True, help="Index directory.")
    ap.add_argument("--model_key", default=None, help="Dense model key (e.g., medcpt); required for dense mode.")
    ap.add_argument("--device", default=None, help="cpu or cuda (dense only). Auto-detected if omitted.")
    ap.add_argument("--k_values", default="1,3,5,10,20", help="Comma-separated k values.")
    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument("--out_json", default=None, help="Optional path to write output metrics JSON.")
    ap.add_argument("--trace_dir", default=None, help="Optional directory to write retrieval_traces.jsonl.")

    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = build_parser()
    args = ap.parse_args()

    # Validate dataset-specific required args
    if not args.data_path:
        ap.error("--data_path is required.")

    if args.dataset == "A":
        evaluate_taskA(args)
    else:
        evaluate_taskB(args)


if __name__ == "__main__":
    main()


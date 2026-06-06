"""Pass 2: full RoMA pair sweep over screened-candidate hydrant sequences.

Reads screening/candidates.txt. For each candidate sequence:
  - iterate frames with step 2: pairs (0,2), (2,4), (4,6), ...
  - skip a pair if either frame is black (record invalid line, doesn't count toward window)
  - else compute frac_conf via RoMA-turbo
  - maintain a sliding window over the last WINDOW valid pairs. After WARMUP valid pairs,
    if the fraction of pairs with frac_conf > 0.8 within the window drops below STOP_THRESHOLD,
    stop the sequence (early termination).

Outputs:
  pairs/<sequence_id>.jsonl       one line per pair, plus a final {"done": true, ...} marker

Resume:
  Skip a sequence if pairs/<id>.jsonl exists AND its last line has "done": true.
  Otherwise (missing or partial) re-run from scratch (file is overwritten).

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/warps/sweep_hydrant_pairs.py
"""

import argparse
import json
import logging
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Optional

import torch

from warps._pair_sweep_utils import (
    ANNOT, DATA_ROOT, FRAC_CONF_GOOD, ROMA_SETTING,
    compute_frac_conf, is_black_image, load_annotations, make_roma,
)

FRAME_STEP = 2
WINDOW = 20
WARMUP = 20
STOP_THRESHOLD = 0.8   # fraction of good pairs in window below which we stop


def is_done_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    last = None
    with open(path, "rb") as f:
        try:
            f.seek(-2, 2)
            while f.read(1) != b"\n":
                f.seek(-2, 1)
        except OSError:
            f.seek(0)
        last = f.readline().decode().strip()
    if not last:
        return False
    try:
        return json.loads(last).get("done") is True
    except json.JSONDecodeError:
        return False


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def sweep_sequence(roma, device, seq_name, frames, out_path: Path, logger) -> dict:
    """Iterate pairs with step FRAME_STEP, write JSONL, return summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_path, "w")
    window = deque(maxlen=WINDOW)  # 1 for good, 0 for not-good (valid pairs only)
    n_pairs = 0
    n_valid = 0
    n_good = 0
    stopped_early: Optional[str] = None

    try:
        for i in range(0, len(frames) - FRAME_STEP, FRAME_STEP):
            j = i + FRAME_STEP
            path_a = DATA_ROOT / frames[i]["filepath"]
            path_b = DATA_ROOT / frames[j]["filepath"]
            n_pairs += 1

            if not path_a.exists() or not path_b.exists():
                f_out.write(json.dumps({"i": i, "j": j, "valid": False,
                                         "reason": "missing_file"}) + "\n")
                continue
            if is_black_image(path_a) or is_black_image(path_b):
                f_out.write(json.dumps({"i": i, "j": j, "valid": False,
                                         "reason": "black"}) + "\n")
                continue

            try:
                frac, _ = compute_frac_conf(roma, device, path_a, path_b)
            except Exception as e:
                f_out.write(json.dumps({"i": i, "j": j, "valid": False,
                                         "reason": f"roma_error: {e}"}) + "\n")
                continue

            good = frac > FRAC_CONF_GOOD
            f_out.write(json.dumps({"i": i, "j": j, "valid": True,
                                     "frac_conf": frac, "good": good}) + "\n")
            n_valid += 1
            n_good += int(good)
            window.append(1 if good else 0)

            if n_valid >= WARMUP and len(window) == WINDOW:
                window_good_frac = sum(window) / WINDOW
                if window_good_frac < STOP_THRESHOLD:
                    stopped_early = f"window_good={window_good_frac:.2f} < {STOP_THRESHOLD}"
                    break

        summary = {
            "done": True,
            "sequence": seq_name,
            "n_pairs": n_pairs,
            "n_valid": n_valid,
            "n_good": n_good,
            "stopped_early": stopped_early is not None,
            "stop_reason": stopped_early,
            "roma_setting": ROMA_SETTING,
            "frac_conf_threshold": FRAC_CONF_GOOD,
            "window": WINDOW,
            "stop_threshold": STOP_THRESHOLD,
        }
        f_out.write(json.dumps(summary) + "\n")
        return summary
    finally:
        f_out.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path,
                        default=Path("/visinf/home/lab_mozkan/computer-vision-proj-lab/data/hydrant_pair_sweep"))
    parser.add_argument("--annot", type=Path, default=ANNOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    assert 0 <= args.rank < args.world_size, "rank must be in [0, world_size)"

    candidates_path = args.out_dir / "screening" / "candidates.txt"
    pairs_dir = args.out_dir / "pairs"
    log_path = args.out_dir / (f"sweep.rank{args.rank}.log" if args.world_size > 1 else "sweep.log")
    pairs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | RoMA setting: %s | rank %d/%d",
                device, ROMA_SETTING, args.rank, args.world_size)

    if not candidates_path.exists():
        logger.error("No candidates file at %s. Run screen_hydrant_sequences.py first.",
                     candidates_path)
        sys.exit(1)

    candidates_all = [line.strip() for line in candidates_path.read_text().splitlines()
                      if line.strip()]
    candidates_all.sort()  # deterministic sharding
    candidates = candidates_all[args.rank::args.world_size]
    logger.info("Total candidates: %d | this shard: %d", len(candidates_all), len(candidates))

    todo = []
    skipped_done = 0
    for seq in candidates:
        out_path = pairs_dir / f"{seq.replace('/', '_')}.jsonl"
        if is_done_file(out_path):
            skipped_done += 1
            continue
        todo.append(seq)
    if args.limit is not None:
        todo = todo[:args.limit]
    logger.info("Already complete: %d | remaining: %d", skipped_done, len(todo))

    if not todo:
        logger.info("Nothing to do.")
        return

    logger.info("Loading annotations: %s", args.annot)
    annotations = load_annotations(str(args.annot))

    roma = make_roma(device)

    for i, seq_name in enumerate(todo):
        frames = annotations.get(seq_name)
        if not frames:
            logger.warning("[%d/%d] %s: not in annotations, skipping",
                           i + 1, len(todo), seq_name)
            continue
        out_path = pairs_dir / f"{seq_name.replace('/', '_')}.jsonl"
        try:
            summary = sweep_sequence(roma, device, seq_name, frames, out_path, logger)
        except Exception:
            logger.error("[%d/%d] %s: unexpected failure\n%s",
                         i + 1, len(todo), seq_name, traceback.format_exc())
            continue
        logger.info("[%d/%d] %s: pairs=%d valid=%d good=%d%s",
                    i + 1, len(todo), seq_name,
                    summary["n_pairs"], summary["n_valid"], summary["n_good"],
                    f" (stopped: {summary['stop_reason']})" if summary["stopped_early"] else "")

    logger.info("Done.")


if __name__ == "__main__":
    main()

"""Pass 1: screen CO3D hydrant sequences for downstream RoMA pair sweep.

For each sequence in hydrant_train.jgz (alphabetically sorted), sample 20 anchors
uniformly spaced along the sequence and compute RoMA frac_conf for the (anchor, anchor+2)
pair. If at least 10 of the 20 anchors produce frac_conf > 0.8, the sequence is added
to the candidates list.

Skips anchors where either frame is black. If fewer than 20 non-black anchors are
available, we still require 10 good pairs to pass.

Outputs (under --out_dir, default data/hydrant_pair_sweep/):
  screening/screened.txt    every processed sequence_id (resume marker)
  screening/candidates.txt  subset that passed screening
  screening/screen.log      detailed per-sequence log

Resume: re-running skips any sequence already in screened.txt.

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/warps/screen_hydrant_sequences.py
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch

from warps._pair_sweep_utils import (
    ANNOT, DATA_ROOT, FRAC_CONF_GOOD, ROMA_SETTING,
    compute_frac_conf, is_black_image, load_annotations, make_roma,
)

FRAME_GAP = 2
N_ANCHORS = 20
MIN_GOOD = 10


def uniform_anchors(n_frames: int, n_anchors: int) -> List[int]:
    """Uniformly spaced anchor indices that leave room for anchor+FRAME_GAP."""
    hi = n_frames - FRAME_GAP - 1
    if hi < 0:
        return []
    if hi + 1 < n_anchors:
        return list(range(hi + 1))
    return list(np.linspace(0, hi, n_anchors).round().astype(int))


def read_id_set(path: Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_line(path: Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("screen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def screen_sequence(roma, device, frames, logger) -> tuple:
    """Returns (passed, good_count, evaluated_count)."""
    anchors = uniform_anchors(len(frames), N_ANCHORS)
    good = 0
    evaluated = 0
    for aidx in anchors:
        path_a = DATA_ROOT / frames[aidx]["filepath"]
        path_b = DATA_ROOT / frames[aidx + FRAME_GAP]["filepath"]
        if not path_a.exists() or not path_b.exists():
            continue
        if is_black_image(path_a) or is_black_image(path_b):
            continue
        try:
            frac, _ = compute_frac_conf(roma, device, path_a, path_b)
        except Exception as e:
            logger.warning("  anchor f%04d failed: %s", aidx, e)
            continue
        evaluated += 1
        if frac > FRAC_CONF_GOOD:
            good += 1
    return good >= MIN_GOOD, good, evaluated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path,
                        default=Path("/visinf/home/lab_mozkan/computer-vision-proj-lab/data/hydrant_pair_sweep"))
    parser.add_argument("--annot", type=Path, default=ANNOT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of sequences (for smoke tests).")
    parser.add_argument("--rank", type=int, default=0,
                        help="This worker's rank in [0, world_size).")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of parallel workers (split by sequence).")
    args = parser.parse_args()
    assert 0 <= args.rank < args.world_size, "rank must be in [0, world_size)"

    screen_dir = args.out_dir / "screening"
    screen_dir.mkdir(parents=True, exist_ok=True)
    screened_path = screen_dir / "screened.txt"
    candidates_path = screen_dir / "candidates.txt"
    log_path = screen_dir / "screen.log"

    log_path = screen_dir / f"screen.rank{args.rank}.log" if args.world_size > 1 else log_path
    logger = setup_logger(log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | RoMA setting: %s | rank %d/%d",
                device, ROMA_SETTING, args.rank, args.world_size)

    logger.info("Loading annotations: %s", args.annot)
    annotations = load_annotations(str(args.annot))
    all_seqs = sorted(annotations.keys())
    # Shard by rank: every world_size-th sequence starting at rank.
    shard = all_seqs[args.rank::args.world_size]
    logger.info("Total sequences: %d | this shard: %d", len(all_seqs), len(shard))

    done = read_id_set(screened_path)
    todo = [s for s in shard if s not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    logger.info("Already screened: %d | remaining: %d", len(done), len(todo))

    if not todo:
        logger.info("Nothing to do.")
        return

    roma = make_roma(device)

    for i, seq_name in enumerate(todo):
        frames = annotations[seq_name]
        if len(frames) < FRAME_GAP + 1:
            logger.info("[%d/%d] %s: too short (%d frames), skipping",
                        i + 1, len(todo), seq_name, len(frames))
            append_line(screened_path, seq_name)
            continue
        try:
            passed, good, evaluated = screen_sequence(roma, device, frames, logger)
        except Exception:
            logger.error("[%d/%d] %s: unexpected failure\n%s",
                         i + 1, len(todo), seq_name, traceback.format_exc())
            # Do NOT mark screened — let a retry pick it up.
            continue

        verdict = "CANDIDATE" if passed else "reject"
        logger.info("[%d/%d] %s: good=%d/%d (eval=%d) -> %s",
                    i + 1, len(todo), seq_name, good, N_ANCHORS, evaluated, verdict)

        append_line(screened_path, seq_name)
        if passed:
            append_line(candidates_path, seq_name)

    logger.info("Done. Candidates so far: %d", len(read_id_set(candidates_path)))


if __name__ == "__main__":
    main()

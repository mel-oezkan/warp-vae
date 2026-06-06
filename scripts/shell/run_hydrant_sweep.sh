#!/usr/bin/env bash
# Launch the hydrant pair-sweep across 2 GPUs in a tmux session.
#
# Usage:
#   scripts/run_hydrant_sweep.sh screen [extra args ...]
#   scripts/run_hydrant_sweep.sh sweep  [extra args ...]
#
# Creates a tmux session "hydrant_<mode>" with two panes:
#   pane 0: CUDA_VISIBLE_DEVICES=0 --rank 0 --world_size 2
#   pane 1: CUDA_VISIBLE_DEVICES=1 --rank 1 --world_size 2
# and attaches. Detach with C-b d; reattach with `tmux a -t hydrant_<mode>`.

set -euo pipefail

MODE="${1:-}"
shift || true
case "$MODE" in
  screen) SCRIPT="warps/screen_hydrant_sequences.py" ;;
  sweep)  SCRIPT="warps/sweep_hydrant_pairs.py" ;;
  *) echo "Usage: $0 {screen|sweep} [extra args ...]" >&2; exit 1 ;;
esac

REPO="/visinf/home/lab_mozkan/computer-vision-proj-lab"
SESSION="hydrant_${MODE}"
CONDA_SH="/visinf/home/lab_mozkan/miniconda3/etc/profile.d/conda.sh"
EXTRA="$*"

# Build the per-rank command once.
make_cmd() {
  local gpu="$1" rank="$2"
  echo "source $CONDA_SH && conda activate cv && \
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO \
python $REPO/scripts/$SCRIPT --rank $rank --world_size 2 $EXTRA; \
echo; echo '[rank $rank] finished — press enter to close'; read"
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux a -t $SESSION" >&2
  exit 1
fi

tmux new-session  -d -s "$SESSION" -n work "$(make_cmd 0 0)"
tmux split-window -t "$SESSION:0" -h            "$(make_cmd 1 1)"
tmux select-layout -t "$SESSION:0" even-horizontal

echo "Launched tmux session '$SESSION'. Attach: tmux a -t $SESSION"
echo "Detach inside tmux with: Ctrl-b d"
tmux attach -t "$SESSION"

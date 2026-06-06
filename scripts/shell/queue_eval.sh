#!/bin/bash
# Queue evaluation after queue_ablations.sh finishes
# Usage: nohup bash scripts/queue_eval.sh > outputs/eval_queue.log 2>&1 &

set -e

ABLATION_PID=222104
CD_DIR="/visinf/home/lab_mozkan/computer-vision-proj-lab"

source /visinf/home/lab_mozkan/miniconda3/etc/profile.d/conda.sh
conda activate cv
cd "$CD_DIR"

echo "[$(date)] Waiting for PID $ABLATION_PID (queue_ablations.sh) to finish..."
while kill -0 "$ABLATION_PID" 2>/dev/null; do
    sleep 600  # Check every 10 minutes
done
echo "[$(date)] PID $ABLATION_PID finished."

echo "[$(date)] Starting: multi-model evaluation on native CO3D"
bash multi.sh
echo "[$(date)] Finished: multi-model evaluation on native CO3D"

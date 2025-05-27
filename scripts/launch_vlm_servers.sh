#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)
# 传入的第一个参数（默认为0）
GPUID="${1:-0}"
GPUNUM="${2:-0}"
REAL_OFFSET=$(( GPUNUM * (GPUID) ))

export BELIEFMAP_PYTHON=${BELIEFMAP_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-beliefmap/vlm/classes.txt}
export GROUNDING_DINO_PORT=$((12181 + REAL_OFFSET))
export BLIP2ITM_PORT=$((12182 + REAL_OFFSET))
export SAM_PORT=$((12183 + REAL_OFFSET))
export YOLOV7_PORT=$((12184 + REAL_OFFSET))
export BLIP2_PORT=$((12185 + REAL_OFFSET))

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "CUDA_VISIBLE_DEVICES=${GPUID} ${BELIEFMAP_PYTHON} -m beliefmap.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "CUDA_VISIBLE_DEVICES=${GPUID} ${BELIEFMAP_PYTHON} -m beliefmap.vlm.sam --port ${SAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "CUDA_VISIBLE_DEVICES=${GPUID} ${BELIEFMAP_PYTHON} -m beliefmap.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
echo "Using PORT_OFFSET=${REAL_OFFSET} => GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT}, etc."

#!/usr/bin/env bash


export VLFM_PYTHON=${VLFM_PYTHON:-$(which python)}


session_name=split_run_${RANDOM}


start_splits=(0 200 400 650 850 1050 1450 1750)
end_splits=(99 400 650 850 1050 1450 1750 2000)


gpu_ids=(0 1 2 3 4 9 10 11)
gpu_num=8


tmux new-session -d -s ${session_name} -n job_0


for i in {1..7}; do
    tmux new-window -t ${session_name} -n "job_${i}"
done


for i in {0..7}; do
    start=${start_splits[$i]}
    end=${end_splits[$i]}
    gpu=${gpu_ids[$i]}
    result_path="final_results/final_result_$((i+1))"


    ./scripts/launch_vlm_servers.sh "$gpu" "$gpu_num"
    sleep 20


    tmux send-keys -t ${session_name}:$i \
        "export start_split=${start} end_split=${end} result_path=${result_path} gpu_id=${gpu} gpu_num=${gpu_num}" C-m
    tmux send-keys -t ${session_name}:$i \
        "CUDA_VISIBLE_DEVICES=${gpu} ${VLFM_PYTHON} -m vlfm.run" C-m
done


echo "Created tmux session '${session_name}' with ${gpu_num} windows (jobs)."
echo "Attach with: tmux attach-session -t ${session_name}"
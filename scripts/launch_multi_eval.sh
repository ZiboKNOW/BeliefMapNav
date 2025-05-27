#!/usr/bin/env bash

export BELIEFMAP_PYTHON=${BELIEFMAP_PYTHON:-$(which python)}


session_name=split_run_${RANDOM}
# start_splits=(0 200 400 650 850 1050 1450 1750)
# end_splits=(200 400 650 850 1050 1450 1750 2000)
start_splits=(0)
end_splits=(2000)


gpu_ids=(1)
gpu_num=1

tmux new-session -d -s ${session_name} -n job_0

# for i in {0}; do
#     tmux new-window -t ${session_name} -n "job_${i}"
# done

for i in 0; do
    echo "Loop iteration $i"
    start=${start_splits[$i]}
    end=${end_splits[$i]}
    gpu=${gpu_ids[$i]}
    result_path="path_to_the_dir/final_result_$((i+1))"
    ./scripts/launch_vlm_servers.sh "$gpu" "$gpu_num"
    sleep 20
    tmux send-keys -t ${session_name}:$i \
        "export start_split=${start} end_split=${end} result_path=${result_path} gpu_id=${gpu} gpu_num=${gpu_num} export http_proxy='127.0.0.1:7890' export https_proxy='127.0.0.1:7890' " C-m
    tmux send-keys -t ${session_name}:$i \
        "mkdir -p  ${result_path}" C-m
    tmux send-keys -t ${session_name}:$i \
        "mkdir -p  ${result_path}/VLM" C-m
    tmux send-keys -t ${session_name}:$i \
        "mkdir -p  ${result_path}/merge_images" C-m 
    tmux send-keys -t ${session_name}:$i \
        "touch ${result_path}/final_result.txt" C-m       
    tmux send-keys -t ${session_name}:$i \
        "CUDA_VISIBLE_DEVICES=${gpu} ${BELIEFMAP_PYTHON} -m beliefmap.run" C-m
done

echo "Created tmux session '${session_name}' with ${gpu_num} windows (jobs)."
echo "Attach with: tmux attach-session -t ${session_name}"
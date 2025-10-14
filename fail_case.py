import numpy as np
from collections import defaultdict

def process_failure_log(file_path):
    failure_indices = []
    failure_count = defaultdict(int)

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if "failure reason:" in line:
                reason = line.strip().split("failure reason:")[1].strip()
                if reason != "did_not_fail":
                    failure_indices.append(idx)
                    failure_count[reason] += 1

    failure_array = np.array(failure_indices, dtype=int)
    return failure_array, dict(failure_count)


failure_array, failure_stats = process_failure_log("path to your final_result.txt")
failure_array = (failure_array - 1)/3

print("Failure line indices:", failure_array.shape)
print("Failure reason stats:", failure_stats)
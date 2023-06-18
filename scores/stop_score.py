# this file computes a soft score and a hard score for stop experiment!

# remember we want linear

# 1. hard score: the success rate: the num of actual stopped /
# 2. soft score: average stop rate

import argparse
import os

from tqdm import tqdm

from utils.direction_utils import (
    boot_sample_stop_mean,
    bootstrap_sample,
    get_mean_CI_for_each_step,
    get_sample_info,
)
from utils.file_utils import load_json

# def get_hard_success_rate(data):
#     success = 0
#     for item in data:
#         # if argmax is stop action, then it is a success stop
#         if np.argmax(item['final_logits'][0]) == len(item['candidates'][0][1]):
#             success += 1
#     return success / len(data)


# def get_soft_success_rate(data):
#     stop_rates = []
#     for item in data:
#         stop_rates.append(item['final_logits'][0]
#                           [len(item['candidates'][0][1])])
#     # ? or should this be bootstrapping?
#     return np.sum(stop_rates)/len(data)


if __name__ == "__main__":
    # load data

    args = argparse.ArgumentParser()
    # args.add_argument('--data_path', type=str, default=None, help="the input data for intervention")
    args.add_argument(
        "--results_path", type=str, default=None, help="the intervention results"
    )
    args.add_argument("--log_path", type=str, default=None, help="the score file path")
    args.add_argument(
        "--baseline_path", type=str, default=None, help="the baseline file path"
    )
    args.add_argument("--bootstrap_num", type=int, default=100)
    args.add_argument(
        "--envdrop", action="store_true", default=False, help="check cli envdrop"
    )
    args = args.parse_args()
    if args.envdrop:
        result_formt = "submit_{}.json"
    else:
        result_formt = "submit_{}_0.json"

    # no_end_result = load_json('./stop/data/submit_no_end_paired_with_ahead_0.json')
    names = [
        "with_end_paired_with_ahead",
        "no_end_paired_with_ahead",
        "full_partial_paired_with_ahead",
        "ahead_partial",
    ]
    datasets = []
    for name in names:
        datasets.append(
            load_json(os.path.join(args.results_path, result_formt.format(name)))
        )

    with_end_result, no_end_result, full_partial_result, ahead_partial_result = datasets

    cluster = ["scan_id", "path_id"]
    replace = [True, False]
    print("---start bootstrapping---")
    to_sample = get_sample_info(no_end_result)
    bootstrap_sample_result = bootstrap_sample(
        to_sample, cluster, replace, size=args.bootstrap_num
    )

    instr_id2no_end = {item["instr_id"]: item for item in no_end_result}
    instr_id2with_end = {item["instr_id"]: item for item in with_end_result}
    instr_id2ahead_partial = {item["instr_id"]: item for item in ahead_partial_result}
    # -------------------------start
    no_end_collects, with_end_collects, ahead_partial_collects = [], [], []
    for sample in tqdm(bootstrap_sample_result):
        no_end_collect = boot_sample_stop_mean(sample, instr_id2no_end)
        with_end_collect = boot_sample_stop_mean(sample, instr_id2with_end)
        ahead_partial_collect = boot_sample_stop_mean(sample, instr_id2ahead_partial)
        no_end_collects.append(no_end_collect)
        with_end_collects.append(with_end_collect)
        ahead_partial_collects.append(ahead_partial_collect)

    no_end_boot = get_mean_CI_for_each_step(no_end_collects)
    with_end_boot = get_mean_CI_for_each_step(with_end_collects)
    ahead_partial_boot = get_mean_CI_for_each_step(ahead_partial_collects)
    # ---------------------------end
    # size = 500
    print("---end bootstrapping---")
    print(
        f"with_end_soft: {with_end_boot['mean_over_all_steps'].iloc[0]}, no_end_soft: {no_end_boot['mean_over_all_steps'].iloc[0]}, ahead_partial_soft: {ahead_partial_boot['mean_over_all_steps'].iloc[0]}"
    )

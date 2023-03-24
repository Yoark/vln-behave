import os
import argparse
import matplotlib.pyplot as plt
from ..utils.room_utils import create_exact_x_hop_dataset_with_no_region_loop
from ..utils.file_utils import load_jsonl, load_json, save_jsonl

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--save_dir", type=str, default="")
    args.add_argument("--baseline_dir", type=str, default="")
    args.add_argument("--region_dir", type=str, default="")
    args.add_argument(
        "--result_dir",
        type=str,
        default="stop/data",
        help="this is only used to retrieve end viewpoint of path",
    )
    args = args.parse_args()

    save_dir = args.save_dir
    # create n hop dataset

    no_end = load_jsonl(
        os.path.join(args.baseline_dir, "rxr_no_end_paired_with_ahead.jsonl")
    )
    result_no_end = load_json(
        os.path.join(args.result_dir, "submit_no_end_paired_with_ahead_0.json")
    )

    scan_viewpoint_id = load_json(
        os.path.join(args.region_dir, "scan_viewpoint_id.json")
    )
    for x in range(2, 9):
        # * note result_no_end is only needed because we use it to retrieve the end viewpoint of truncation path as well as 
        # * the candidate viewpoints for the end viewpoint
        # * the end viewpoint is a result of following ground truth path, therefore is the last node of GT path.
        results = create_exact_x_hop_dataset_with_no_region_loop(
            no_end, result_no_end, x, scan_viewpoint_id, with_stop=True
        )
        intervene_dataset, no_intervene_dataset, region2hop_nums = (
            results["intervene_dataset"],
            results["no_intervene_dataset"],
            results["region2hop_nums"],
        )
        dir_name = f"{save_dir}/{str(x)}_hop"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        total_hop_nums = []
        for k, v in region2hop_nums.items():
            total_hop_nums.extend(v)
        plt.hist(total_hop_nums)
        dir_name = f"{save_dir}/{str(x)}_hop"
        save_jsonl(f"{dir_name}/rxr_{x}_hop_intervene_dataset.jsonl", intervene_dataset)
        save_jsonl(
            f"{dir_name}/rxr_{x}_hop_no_intervene_dataset.jsonl", no_intervene_dataset
        )

    # create for 1
    x = 1
    results = create_exact_x_hop_dataset_with_no_region_loop(
        no_end, result_no_end, x, scan_viewpoint_id, with_stop=True
    )
    intervene_dataset, no_intervene_dataset, region2hop_nums = (
        results["intervene_dataset"],
        results["no_intervene_dataset"],
        results["region2hop_nums"],
    )
    dir_name = f"{save_dir}/{str(x)}_hop"

    total_hop_nums = []
    for k, v in region2hop_nums.items():
        total_hop_nums.extend(v)
    # plt.hist(region2hop_nums)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_jsonl(f"{dir_name}/rxr_{x}_hop_intervene_dataset.jsonl", intervene_dataset)
    save_jsonl(
        f"{dir_name}/rxr_{x}_hop_no_intervene_dataset.jsonl", no_intervene_dataset
    )

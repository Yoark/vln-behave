# soft score: if prob weighted average angle
# hard score: if argmax is in the target direction, success
# hard score
import argparse
import os

import numpy as np
from ipdb import launch_ipdb_on_exception

from utils.direction_utils import (
    bootstrap_sample,
    filter_direction_instances,
    get_heading_and_probs_pairs_all_directions,
    get_sample_info,
    rad2deg_mod,
)
from utils.file_utils import load_json


def check_success_direction(item, target):
    """
    Check success or not for direction experiment, target is
    "left", "right", "forward", "backward", "back left", "back right"
    item: the inference result item
    """
    argmax_index = np.argmax(item["final_logits"][0])
    # agent choose stop
    stop = 0
    if argmax_index == len(item["candidates"][0][1]):
        stop = 1
        return 0, stop
    argmax_direction = rad2deg_mod(item["candidates"][0][1][argmax_index]["heading"])
    # print(argmax_direction)
    # ipdb.set_trace()
    success = False
    assert target in ["left", "right", "forward", "backward", "back left", "back right"]
    if target == "left":
        success = -90 < argmax_direction <= -15
    elif target == "right":
        success = 15 < argmax_direction <= 90
    elif target == "forward":
        success = -15 < argmax_direction <= 15
    elif target == "backward":
        success = -180 < argmax_direction <= -165 or 165 < argmax_direction <= 180
    elif target == "back left":
        success = -165 < argmax_direction <= -90
    elif target == "back right":
        success = 90 < argmax_direction <= 165
    return int(success), stop


def get_hard_success_rate(data, target, check_func=check_success_direction):
    total_success = 0
    count_stop = 0
    for item in data:
        success, stop = check_func(item, target)
        total_success += success
        count_stop += stop

    print(
        f"target: {target:<10} data num: {len(data):<4} stop number: {count_stop:<4} success_num: {total_success:<4}"
    )
    return total_success / len(data)


# soft score


def get_soft_success_rate(data, target):
    """compute correct rate for target direction
        sum (prob in target range) / total

    Args:
        data (_type_): inference result
        target (_type_): "left", "right", etc
    """
    _, direction_probs, stop_probs = get_heading_and_probs_pairs_all_directions(data)
    total = sum([it[1] for it in direction_probs]) + sum(stop_probs)
    target_probs = []
    if target == "left":
        for item in direction_probs:
            direction, prob = item
            if -90 < direction <= -15:
                target_probs.append(prob)
    elif target == "right":
        for item in direction_probs:
            direction, prob = item
            if 15 < direction <= 90:
                target_probs.append(prob)
    elif target == "forward":
        for item in direction_probs:
            direction, prob = item
            if -15 < direction <= 15:
                target_probs.append(prob)
    elif target == "backward":
        for item in direction_probs:
            direction, prob = item
            if -180 < direction <= -165 or 165 < direction <= 180:
                target_probs.append(prob)
    elif target == "back left":
        for item in direction_probs:
            direction, prob = item
            if -165 < direction <= -90:
                target_probs.append(prob)
    elif target == "back right":
        for item in direction_probs:
            direction, prob = item
            if 90 < direction <= 165:
                target_probs.append(prob)
    return sum(target_probs) / total
    # return sum(target_probs) / len(data)


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        args = argparse.ArgumentParser()
        # args.add_argument('--data_path', type=str, default=None, help="the input data for intervention")
        args.add_argument(
            "--results_path", type=str, default=None, help="the intervention results"
        )
        args.add_argument(
            "--baseline_path",
            type=str,
            default=None,
            help="the baseline file path, use this if not results_path is provided",
        )
        args.add_argument(
            "--envdrop",
            action="store_true",
            default=False,
            help="this is just used to change template",
        )

        args = args.parse_args()

        directions = [
            "turn left",
            "turn right",
            "walk forward",
            "walk backward",
            "turn back left",
            "turn back right",
        ]

        if args.envdrop:
            result_format = "submit_new_{}.json"
        else:
            result_format = "submit_new_{}_0.json"

        #  preload any result to do bootstrap (only ids are needed so which result is not important)

        dir_str = "_".join("turn left".split(" "))
        result_left = load_json(
            os.path.join(args.results_path, result_format.format(dir_str))
        )

        cluster = ["scan_id", "path_id"]
        replace = [True, False]
        to_sample = get_sample_info(result_left)
        bootstrap_sample_result = bootstrap_sample(
            to_sample, cluster, replace, size=100
        )

        soft_score = 0
        for dir in directions:
            dir_str = "_".join(dir.split(" "))
            if args.baseline_path:
                result = load_json(args.baseline_path)
            else:
                result = load_json(
                    os.path.join(args.results_path, result_format.format(dir_str))
                )

            inter_id2data = {i["instr_id"]: i for i in result}
            target_dir = " ".join(dir.split(" ")[1:])
            mean_accum_probs = []
            for sample in bootstrap_sample_result:
                sample = [inter_id2data[id] for id in sample["instr_id"]]
                result_filtered, _ = filter_direction_instances(sample, target_dir)
                mean_accum_prob = get_soft_success_rate(result_filtered, target_dir)
                mean_accum_probs.append(mean_accum_prob)
            soft_score += np.mean(mean_accum_probs)
            # hard_score += get_hard_success_rate(result_filtered, target_dir)
        soft_score /= len(directions)
        print("soft score: ", soft_score)
        # score_str = f"{args.results_path:>30}'\t'soft score: {soft_score:.3f}'\t' hard score{hard_score:.3f}"

    # write_to_record_file(score_str, record_file)

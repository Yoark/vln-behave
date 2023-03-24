# these scores are for room experiment (1 hop)
# hard score
import os
import argparse
import numpy as np

from ..utils.file_utils import load_json, load_jsonl
from ..utils.direction_utils import get_sample_info, bootstrap_sample

scan_viewpoint_label = load_json("./region_data/accurate_scan_viewpoint_label.json")


def check_room_success(result_item, input_item):
    argmax_index = np.argmax(result_item["final_logits"][0])
    # if stopped then fail
    if argmax_index == len(result_item["candidates"][0][1]):
        return 0
    goal_region_type = input_item["region_type"]
    argmax_next_viewpoint = result_item["next_pos"]
    assert (
        result_item["candidates"][0][1][argmax_index]["viewpointId"]
        == argmax_next_viewpoint
    )
    scan_id = input_item["scan"]
    next_region_type = scan_viewpoint_label[scan_id][argmax_next_viewpoint]
    if goal_region_type == next_region_type:
        return 1
    return 0


def get_room_success_rate(results, instr_id2item, check_func=None):
    # instr_id2item = {item['instruction_id']: item for item in data}
    success = 0
    for result_item in results:
        instr_id = "_".join(result_item["instr_id"].split("_")[1:])
        input_item = instr_id2item[instr_id]
        if check_func:
            success += check_func(result_item, input_item)
        else:
            print("Error: check_func is None")
            break
    return success / len(results)


def get_room_target_neighbor_probs(result_item, input_item):
    goal_region_type = input_item["region_type"]
    prob_mass_in_target_range = []
    for i, cand in enumerate(result_item["candidates"][0][1]):
        cand_viewpoint = cand["viewpointId"]
        scan_id = input_item["scan"]
        cand_region_type = scan_viewpoint_label[scan_id][cand_viewpoint]
        if goal_region_type == cand_region_type:
            prob_mass_in_target_range.append(result_item["final_logits"][0][i])
    return sum(prob_mass_in_target_range)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_path", type=str, default=None, help="the input data for intervention"
    )
    args.add_argument(
        "--results_path", type=str, default=None, help="the intervention results"
    )
    args.add_argument("--model_name", type=str, default=None, help="the model name")
    # args.add_argument('--log_path', type=str, default=None, help="the score file path")

    args.add_argument(
        "--envdrop", action="store_true", default=False, help="check cli envdrop"
    )

    args = args.parse_args()
    if args.envdrop:
        result_format = "submit_{}_dataset.json"
    else:
        result_format = "submit_{}_dataset_0.json"
    data_format = "rxr_{}_dataset.jsonl"
    # to bootstrap here!

    names = ["1_hop_intervene", "1_hop_no_intervene"]
    results = load_json(os.path.join(args.results_path, result_format.format(names[0])))

    cluster = ["scan_id", "path_id"]
    replace = [True, False]
    to_sample = get_sample_info(results)
    bootstrap_sample_result = bootstrap_sample(
        to_sample, cluster=cluster, replace=replace, size=100
    )

    data = []
    for name in names:
        data = load_jsonl(os.path.join(args.data_path, data_format.format(name)))
        instr_id2item = {item["instruction_id"]: item for item in data}
        results = load_json(os.path.join(args.results_path, result_format.format(name)))
        instr_id2result = {item["instr_id"]: item for item in results}

        scores = []
        for sample in bootstrap_sample_result:
            sample_instr_id = sample["instr_id"]
            sample_result = [instr_id2result[i] for i in sample_instr_id]

            soft_score = get_room_success_rate(
                sample_result, instr_id2item, check_func=get_room_target_neighbor_probs
            )
            scores.append(soft_score)

        mean_score = np.mean(scores)

        print(f"name: {name} model: {args.model_name}: soft score: {mean_score}")
        # print(f'name: {name} hard score: {hard_score}')
    print(len(data))

# CMD: python room_score.py --data_path ./rooms/data_Nov_10_new/1_hop --results_path ./rooms/data_Nov_10_new/1_hop --model_name hamt

# soft and hard score for object conditioned direction score
import argparse
import os

import numpy as np

from utils.file_utils import load_json, load_jsonl
from utils.object_utils import diff_with_direction

HFOV = 30


# hard score
def check_success_object_direction(result_item, input_item):
    """
    if the argmax direction is within a 30 of the target direction, it is a success
    result_item: inference result
    input_item: input data item
    """
    argmax_index = np.argmax(result_item["final_logits"][0])
    if argmax_index == len(result_item["candidates"][0][1]):
        return 0
    argmax_direction = result_item["candidates"][0][1][argmax_index][
        "normalized_heading"
    ]

    success = False
    target_direction = input_item["obj_heading"]  # this is a normalized heading
    if (
        np.abs(diff_with_direction(target_direction, argmax_direction, radians=True))
        < HFOV / 2
    ):
        success = True
    return int(success)


def get_success_rate(results, instr_id2item, check_func=None):
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


# soft score
def get_target_neighbor_probs(result_item, input_item):
    target_direction = input_item["obj_heading"]  # this is a normalized heading
    prob_mass_in_target_range = []
    for i, cand in enumerate(result_item["candidates"][0][1]):
        if (
            np.abs(diff_with_direction(target_direction, cand["normalized_heading"]))
            < HFOV / 2
        ):
            prob_mass_in_target_range.append(result_item["final_logits"][0][i])
    return sum(prob_mass_in_target_range)


if __name__ == "__main__":
    from direction.data_process import bootstrap_sample, get_sample_info
    from objects.objects import filter_instances_by_objects_heading

    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_path", type=str, default=None, help="the input data for intervention"
    )
    args.add_argument(
        "--results_path", type=str, default=None, help="the intervention results"
    )
    args.add_argument("--model_name", type=str, default="hamt", help="the model name")
    args.add_argument(
        "--envdrop", action="store_true", default=False, help="check cli envdrop"
    )

    args = args.parse_args()

    data_name_format = "rxr_reverie_{}_3m_sample.jsonl"
    if args.envdrop:
        result_name_format = "submit_reverie_{}_3m_sample.json"
    else:
        result_name_format = "submit_reverie_{}_3m_sample_0.json"

    # load data for outer bootstrap
    submit_obj_reverie_sample = load_json(
        os.path.join(args.results_path, result_name_format.format("obj"))
    )
    names = ["no_end", "obj"]
    cluster = ["scan_id", "path_id"]
    replace = [True, False]
    to_sample = get_sample_info(submit_obj_reverie_sample)
    bootstrap_sample_result = bootstrap_sample(
        to_sample, cluster=cluster, replace=replace, size=100
    )

    for name in names:
        result = load_json(
            os.path.join(args.results_path, result_name_format.format(name))
        )
        data = load_jsonl(os.path.join(args.data_path, data_name_format.format(name)))
        instr_id2data = {item["instruction_id"]: item for item in data}
        instr_id2result = {item["instr_id"]: item for item in result}
        scores = []
        for sample in bootstrap_sample_result:
            selected_samples = [instr_id2result[i] for i in sample["instr_id"]]
            idxes = filter_instances_by_objects_heading(
                result, instr_id2data, fov_object=30
            )
            result_filtered = [instr_id2result[i] for i in idxes]
            soft_score = get_success_rate(
                result_filtered, instr_id2data, check_func=get_target_neighbor_probs
            )
            scores.append(soft_score)

        mean_score = np.mean(scores)

        print(f"soft score: of {args.model_name} under {name} is {mean_score}")


# python object_score.py --data_path ./objects/data --results_path ./objects/data/ --model_name hamt

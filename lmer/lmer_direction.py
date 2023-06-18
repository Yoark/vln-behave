import argparse
import os

import pandas as pd

from utils.direction_utils import filter_direction_instances, rad2deg_mod
from utils.file_utils import load_json


def save2csv(data, name, store_path):
    all_s = pd.DataFrame(data)
    all_s.to_csv(f"{store_path}/{name}.csv", index=False)


def inside_checker(direction, deg):
    if direction == "left" and (-90 < deg <= -15):
        return 1
    if direction == "right" and (15 < deg <= 90):
        return 1
    if direction == "forward" and (-15 < deg <= 15):
        return 1
    if direction == "backward" and (
        -180 < deg <= -165 or 165 < deg <= 180
    ):  # check this
        return 1
    if direction == "back left" and (-165 < deg <= -90):
        return 1
    if direction == "back right" and (90 < deg <= 165):
        return 1

    return 0


def create_scan_id2intervene_label2turn_accum_prob(data, intervene_label, direction):
    scan_id_intervene_label_accum_prob = []
    for item in data:
        accum_prob_list = []
        scan_id = item["candidates"][0][1][0]["scanId"]
        for i, (cand, prob) in enumerate(
            zip(item["candidates"][0][1], item["final_logits"][0])
        ):  # only considering the first after supervision for now
            deg = rad2deg_mod(cand["heading"])
            is_inside = inside_checker(direction, deg)
            if is_inside:
                accum_prob_list.append(prob)
        scan_id_intervene_label_accum_prob.append(
            {
                "scan_id": scan_id,
                "path_id": item["instr_id"].split("_")[0],
                "step_id": item["instr_id"].split("_")[2],
                "intervention_yes_no": intervene_label,
                "accum_prob": sum(accum_prob_list),
                "direction": direction,
            }
        )
    return scan_id_intervene_label_accum_prob


def get_data(direction, data_name, args):
    if args.model_name == "hamt":
        result = load_json(f"{args.result_path}/submit_new_{data_name}_0.json")
    else:
        result = load_json(f"{args.result_path}/submit_new_{data_name}.json")
    result_filtered, _ = filter_direction_instances(result, direction=direction)
    if args.model_name == "hamt":
        result_no_end = load_json(
            f"{args.baseline_path}/submit_no_end_paired_with_ahead_0.json"
        )
    else:
        result_no_end = load_json(
            f"{args.baseline_path}/submit_no_end_paired_with_ahead.json"
        )
    no_end_filtered, _ = filter_direction_instances(result_no_end, direction=direction)
    return result_filtered, no_end_filtered


def get_formatted_total(direction, data_name, args):
    result_filtered, no_end_filtered = get_data(direction, data_name, args)
    intervene_format = create_scan_id2intervene_label2turn_accum_prob(
        result_filtered, 1, direction
    )
    no_intervention_format = create_scan_id2intervene_label2turn_accum_prob(
        no_end_filtered, 0, direction
    )
    total = intervene_format + no_intervention_format
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path", type=str, default="direction/data", help="Path to model result"
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        default="stop/data",
        help="Path to intervention samples",
    )
    parser.add_argument("--model_name", type=str, default="hamt", help="Model name")
    parser.add_argument(
        "--store_dir", default="lmer_data", help="Path to the store directory."
    )
    args = parser.parse_args()

    store_path = os.path.join(args.store_dir, args.model_name + "/direction")
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)

    total_left = get_formatted_total("left", "turn_left", args)
    total_right = get_formatted_total("right", "turn_right", args)
    total_forward = get_formatted_total("forward", "walk_forward", args)
    total_backward = get_formatted_total("backward", "walk_backward", args)
    total_back_left = get_formatted_total("back left", "turn_back_left", args)
    total_back_right = get_formatted_total("back right", "turn_back_right", args)

    save2csv(total_forward, "forward", store_path)
    save2csv(total_backward, "backward", store_path)
    save2csv(total_left, "left", store_path)
    save2csv(total_right, "right", store_path)
    save2csv(total_back_left, "back_left", store_path)
    save2csv(total_back_right, "back_right", store_path)


if __name__ == "__main__":
    main()


# python lmer/lmer_direction.py --result_path results/clip-envdrop --model_name clip-envdrop --baseline_path results/clip-envdrop

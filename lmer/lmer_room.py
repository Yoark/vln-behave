import argparse
import os

import pandas as pd

from utils.file_utils import load_json, load_jsonl
from utils.room_utils import process_result_n_hop


def save2csv(data, name, store_path):
    all_s = pd.DataFrame(data)
    all_s.to_csv(f"{store_path}/{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lmer room 1hop")
    parser.add_argument("--result_path", type=str, default="rooms/data_Nov_10_new")
    parser.add_argument("--input_path", type=str, default="rooms/data_Nov_10_new")
    parser.add_argument("--model_name", type=str, default="hamt", help="Model name")
    parser.add_argument(
        "--store_dir", default="lmer_data", help="Path to the store directory."
    )
    parser.add_argument("--hop", type=int, default=1, help="hop number")
    parser.add_argument(
        "--mode", type=str, default="delta", help="use dist to goal or delta"
    )
    args = parser.parse_args()

    store_path = os.path.join(args.store_dir, args.model_name + "/room")
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)

    hop = args.hop
    data_dir = args.input_path
    result_dir = args.result_path

    hop_data = load_jsonl(
        f"{data_dir}/{hop}_hop/rxr_{hop}_hop_intervene_dataset.jsonl"
    ), load_jsonl(f"{data_dir}/{hop}_hop/rxr_{hop}_hop_no_intervene_dataset.jsonl")
    if "envdrop" in result_dir:
        hop_result = load_json(
            f"{result_dir}/{hop}_hop/submit_{hop}_hop_intervene_dataset.json"
        ), load_json(
            f"{result_dir}/{hop}_hop/submit_{hop}_hop_no_intervene_dataset.json"
        )
    else:
        hop_result = load_json(
            f"{result_dir}/{hop}_hop/submit_{hop}_hop_intervene_dataset_0.json"
        ), load_json(
            f"{result_dir}/{hop}_hop/submit_{hop}_hop_no_intervene_dataset_0.json"
        )

    hop_processed = process_result_n_hop(hop_result, hop_data, argmax=False)

    if hop == 1:
        if args.mode == "delta":
            intervene = hop_processed["next_delta_intervene"]
            no_intervene = hop_processed["next_delta_no_intervene"]
            stop = hop_processed["dist_to_goal_stops"]
        else:
            intervene = hop_processed["dist_to_goal_next_weighted_avg_intervene"]
            no_intervene = hop_processed["dist_to_goal_next_weighted_avg_no_intervene"]
            stop = hop_processed["dist_to_goal_stops"]

        for item in intervene:
            item["intervention_yes_no"] = 1
        for item in no_intervene:
            item["intervention_yes_no"] = 0
        for item in stop:
            item["intervention_yes_no"] = 0
    else:
        # only 1 dist to goal mode is possible for n hop case
        intervene = hop_processed["dist_to_goal_intervene_final"]
        no_intervene = hop_processed["dist_to_goal_no_intervene_final"]
        stop = hop_processed["dist_to_goal_stops"]

        for item in intervene:
            item["intervention_yes_no"] = 1
        for item in no_intervene:
            item["intervention_yes_no"] = 0
        for item in stop:
            item["intervention_yes_no"] = 0

    # total = intervene + no_intervene
    total = intervene + stop

    name = f"{hop}_hop_{args.mode}_stop"
    save2csv(total, name, store_path)


# python lmer/lmer_room.py --hop 2 --mode dist --result_path results/envdrop-imagenet --model_name envdrop-imagenet

import os
import json
import argparse
from ..utils.file_utils import load_jsonl, load_json
from ..utils.room_utils import (
    process_result_n_hop,
    draw_distribution_on_delta_to_goal_room,
)
from ipdb import launch_ipdb_on_exception

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_dir",
        type=str,
        default="rooms/data_Nov_10_new/",
        help="the input data for intervention",
    )
    args.add_argument("--model_name", type=str, default="hamt", help="the model name")
    args.add_argument(
        "--result_dir",
        type=str,
        default="rooms/data_Nov_10_new/",
        help="the intervention results",
    )
    args.add_argument(
        "--save_dir", type=str, default="results/room", help="the fig file path"
    )
    args.add_argument("--hop", type=int, default=1, help="the hop number")
    args.add_argument(
        "--argmax_next_action",
        action="store_true",
        default=False,
        help="use argmax next action rather than a weighted next action when computing the distance",
    )
    args.add_argument('--bootstrap_num', type=int, default=100, help='the number of bootstrap samples')
    args.add_argument('--region_file_dir', type=str, default='/Users/zijiao/home/research/unit-vln/code/region_data/scan_region.json', help="the region file dir")

    args = args.parse_args()
    data_dir = args.data_dir
    model_name = args.model_name
    hop = args.hop
    result_dir = data_dir
    save_dir = os.path.join(args.save_dir, args.model_name, "room")
    save_dir = os.path.join(save_dir, f"{args.hop}_hop")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hop_data = load_jsonl(
        f"{data_dir}/{hop}_hop/rxr_{hop}_hop_intervene_dataset.jsonl"
    ), load_jsonl(f"{data_dir}/{hop}_hop/rxr_{hop}_hop_no_intervene_dataset.jsonl")
    if "envdrop" in result_dir:
        hop_result = load_json(
            f"{result_dir}/submit_{hop}_hop_intervene_dataset.json"
        ), load_json(f"{result_dir}/submit_{hop}_hop_no_intervene_dataset.json")
    else:
        hop_result = load_json(
            f"{result_dir}/{hop}_hop/submit_{hop}_hop_intervene_dataset_0.json"
        ), load_json(
            f"{result_dir}/{hop}_hop/submit_{hop}_hop_no_intervene_dataset_0.json"
        )

    instr_id2no_intervention_result = {item['instr_id']: item for item in hop_result[1]}
    instr_id2intervention_dataset = {item['instruction_id']: item for item in hop_data[0]}

    with open(args.region_file_dir) as t:
        scan_region = json.load(t)

    with launch_ipdb_on_exception():
        hop_processed = process_result_n_hop(hop_result, hop_data, argmax=False, scanregion=None, 
                                             instr_id2no_intervene_result=instr_id2no_intervention_result, 
                                             instr_id2intervene_dataset=instr_id2intervention_dataset)
        argmax = True if args.argmax_next_action else False
        draw_distribution_on_delta_to_goal_room(
            hop_processed,
            hop=hop,
            save_dir=save_dir,
            argmax=argmax,
            maxValue=8,
            delta=True,
            bootstrap=False,
        )


# python room_distribution_plot.py --hop 1 --model_name clip-envdrop --result_dir results/clip_envdrop --save_dir camera-ready-new
#  python room_distribution_plot.py --hop 1 --model_name envdrop-imagenet --result_dir results/envdrop-imagenet --save_dir camera-ready-new

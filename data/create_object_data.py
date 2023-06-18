import argparse
import os
import pathlib

from ipdb import launch_ipdb_on_exception
from tqdm import tqdm

from utils.file_utils import load_json, load_jsonl, save_jsonl
from utils.object_utils import (
    add_heading_info,
    get_obj_instances_for_all,
    template_go_toward,
    template_nothing,
)

if __name__ == "__main__":
    # Test data_create_test.py
    args = argparse.ArgumentParser()
    args.add_argument(
        "--baseline_dir",
        type=str,
        default="/Users/zijiao/home/research/unit-vln/code/baselines",
        help="path to truncation data",
    )
    args.add_argument(
        "--object_path",
        type=str,
        default="/Users/zijiao/home/research/unit-vln/REVERIE/tasks/REVERIE/data/BBox",
    )
    args.add_argument("--save_dir", type=str, default="")
    args = args.parse_args()

    with launch_ipdb_on_exception():
        no_end = load_jsonl(
            os.path.join(args.baseline_dir, "rxr_no_end_paired_with_ahead.jsonl")
        )
        # no_end_submit = load_json(work_dir.format('stop/data/submit_no_end_paired_with_ahead_0.json'))
        # ahead = load_jsonl(os.path.join(args.truncation_data_path, 'baselines/rxr_ahead_partial.jsonl'))
        # collect viewpoint from files into one
        # bbox_dir = '/Users/zijiao/home/research/unit-vln/data/BBoxes_v2'
        bbox_3m_dir = args.object_path
        bbox = pathlib.Path(bbox_3m_dir)
        # either create viewpoints dict or load from file viewpoints.pkl
        viewpoints = {}
        for box in tqdm(bbox.glob("*.json")):
            result = load_json(box)
            name = box.stem.split("_")[1]
            if result[name] != {}:
                viewpoints[name] = result[name]

        for vid, obj_info in viewpoints.items():
            add_heading_info(obj_info)

        no_end_intervene = get_obj_instances_for_all(
            no_end,
            viewpoints,
            template_func=template_nothing,
        )

        obj_intervene = get_obj_instances_for_all(
            no_end,
            viewpoints,
            template_func=template_go_toward,
        )
        all_instr_ids = [item["instruction_id"] for item in no_end_intervene]
        print(len(no_end_intervene), len(obj_intervene))

        save_jsonl(f"{args.save_dir}/rxr_reverie_obj_3m_sample.jsonl", obj_intervene)
        save_jsonl(
            f"{args.save_dir}/rxr_reverie_no_end_3m_sample.jsonl", no_end_intervene
        )

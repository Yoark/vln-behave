# scan id, path id, step id, intervention yes no, prob within the close abs angle
import argparse
import os

import numpy as np
import pandas as pd

from utils.file_utils import load_json, load_jsonl
from utils.object_utils import diff_with_direction, filter_instances_by_objects_heading


def save2csv(data, name, save_path):
    all_s = pd.DataFrame(data)
    all_s.to_csv(f"{save_path}/{name}.csv", index=False)


def create_scan_id2intervene_label2object(input_data, data, intervene_label):
    fov_object = 30
    scan_id_intervene_label_accum_prob = []
    for item in data:
        candidate_headings = [
            cand["normalized_heading"] for cand in item["candidates"][0][1]
        ]  # rel heading
        cand_probs = item["final_logits"][0][: len(candidate_headings)]
        gt = input_data["_".join(item["instr_id"].split("_")[1:])]
        object_gt_heading = gt["obj_heading"]
        # stop_prob = item['final_logits'][0][len(candidate_headings)]
        scan_id = item["candidates"][0][1][0]["scanId"]
        step_id = item["instr_id"].split("_")[2]
        path_id = item["instr_id"].split("_")[0]
        object_name = gt["obj_name"]

        # candidate_contain_object_within_fov = [abs(diff_with_direction(object_gt_heading, cand_heading)) < fov_object/2 for cand_heading in candidate_headings]
        # result_samples = []
        prob_within_fov_object = 0
        for prob, cand in zip(cand_probs, candidate_headings):
            sam = (
                np.abs(diff_with_direction(cand, object_gt_heading)),
                prob,
                scan_id,
                step_id,
                path_id,
            )
            if sam[0] < fov_object / 2:
                prob_within_fov_object += sam[1]

        scan_id_intervene_label_accum_prob.append(
            {
                "intervention_yes_no": intervene_label,
                "accum_prob": prob_within_fov_object,
                "scan_id": scan_id,
                "step_id": step_id,
                "path_id": path_id,
                "object_name": object_name,
            }
        )
    return scan_id_intervene_label_accum_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lmer object")
    parser.add_argument("--result_path", type=str, default="objects/data")
    parser.add_argument("--input_path", type=str, default="objects/data")
    parser.add_argument("--model_name", type=str, default="hamt", help="Model name")
    parser.add_argument(
        "--store_dir", default="lmer_data", help="Path to the store directory."
    )
    args = parser.parse_args()

    store_path = os.path.join(args.store_dir, args.model_name + "/object")
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)

    result_path = args.result_path
    if "envdrop" in args.model_name:
        submit_obj_reverie_sample = load_json(
            f"{result_path}/submit_reverie_obj_3m_sample.json"
        )
        # submit_ahead_reverie_sample = load_json("./objects/data/submit_reverie_ahead_3m_sample_0.json")
        submit_no_end_reverie_sample = load_json(
            f"{result_path}/submit_reverie_no_end_3m_sample.json"
        )
    else:
        submit_obj_reverie_sample = load_json(
            f"{result_path}/submit_reverie_obj_3m_sample.json"
        )
        # submit_ahead_reverie_sample = load_json("./objects/data/submit_reverie_ahead_3m_sample_0.json")
        submit_no_end_reverie_sample = load_json(
            f"{result_path}/submit_reverie_no_end_3m_sample.json"
        )

    input_path = args.input_path
    obj_reverie_sample = load_jsonl(f"{input_path}/rxr_reverie_obj_3m_sample.jsonl")
    # ahead_reverie_sample = load_jsonl('./objects/data/rxr_reverie_ahead_3m_sample.jsonl')
    no_end_reverie_sample = load_jsonl(
        f"{input_path}/rxr_reverie_ahead_3m_sample.jsonl"
    )

    instr_id2obj_reverie = {item["instruction_id"]: item for item in obj_reverie_sample}
    instr_id2no_end_reverie = {
        item["instruction_id"]: item for item in no_end_reverie_sample
    }
    instr_id2intervene_result = {
        item["instr_id"]: item for item in submit_obj_reverie_sample
    }
    instr_id2no_intervene_result = {
        item["instr_id"]: item for item in submit_no_end_reverie_sample
    }

    filter_indexes = filter_instances_by_objects_heading(
        submit_obj_reverie_sample, instr_id2obj_reverie, fov_object=30
    )

    filtered_intervene = [
        item for item in submit_obj_reverie_sample if item["instr_id"] in filter_indexes
    ]
    filtered_no_intervene = [
        item
        for item in submit_no_end_reverie_sample
        if item["instr_id"] in filter_indexes
    ]

    intervene_format = create_scan_id2intervene_label2object(
        instr_id2obj_reverie, filtered_intervene, 1
    )
    no_intervene_format = create_scan_id2intervene_label2object(
        instr_id2no_end_reverie, filtered_no_intervene, 0
    )
    total = intervene_format + no_intervene_format

    save2csv(total, "total_objects", store_path)

import argparse
import os

import pandas as pd

from utils.file_utils import load_json


def create_scan_id_intervene_label_stop_prob(data, intervene_label):
    sample_triplet = [
        {
            "scan_id": item["candidates"][0][1][0]["scanId"],
            "path_id": item["instr_id"].split("_")[0],
            "step_id": item["instr_id"].split("_")[-1],
            "intervention_yes_no": intervene_label,
            "stop_prob": item["stop_prob"][0],
        }
        for item in data
    ]
    return sample_triplet


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process data directory path.")
    parser.add_argument("--data_dir", help="Path to the data directory.")
    parser.add_argument(
        "--store_dir", default="lmer_data", help="Path to the store directory."
    )
    parser.add_argument("--file_name", help="File name for csv")
    parser.add_argument("--model_name", default="hamt", help="Model name")
    args = parser.parse_args()

    store_path = os.path.join(args.store_dir, args.model_name + "/stop")
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)

    data_dir = args.data_dir

    # no_end_result = load_json(f'{data_dir}/submit_no_end_paired_with_ahead_0.json')
    if "envdrop" in args.model_name:
        no_end_result = load_json(f"{data_dir}/submit_no_end_paired_with_ahead.json")
        with_end_result = load_json(
            f"{data_dir}/submit_with_end_paired_with_ahead.json"
        )
        # ahead_partial_result = load_json(f'{data_dir}/submit_ahead_partial.json')
    else:
        no_end_result = load_json(f"{data_dir}/submit_no_end_paired_with_ahead_0.json")
        with_end_result = load_json(
            f"{data_dir}/submit_with_end_paired_with_ahead_0.json"
        )
        # ahead_partial_result = load_json(f'{data_dir}/submit_ahead_partial_0.json')

    no_end_samples = create_scan_id_intervene_label_stop_prob(no_end_result, 0)
    with_end_samples = create_scan_id_intervene_label_stop_prob(with_end_result, 1)
    # ahead_partial_samples = create_scan_id_intervene_label_stop_prob(ahead_partial_result, 0)

    # print(len(with_end_samples), len(no_end_samples), len(ahead_partial_result))
    all_samples = no_end_samples + with_end_samples
    all_s = pd.DataFrame(all_samples)

    all_s.to_csv(os.path.join(store_path, args.file_name + ".csv"), index=False)
    print("Completed")


if __name__ == "__main__":
    main()

#  python lmer/lmer_stop.py --data_dir stop/data --file_name with_end1_ahead_partial0 --model_name hamt


# no end vs ahead python lmer/lmer_stop.py --data_dir stop/data --file_name no_end1_ahead_partial0
# with end vs ahead python lmer/lmer_stop.py --data_dir stop/data --file_name with_end1_ahead_partial0
# with end vs no end python lmer/lmer_stop.py --data_dir stop/data --file_name with_end1_no_end0

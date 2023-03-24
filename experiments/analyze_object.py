# from torch import gt
from ipdb import launch_ipdb_on_exception
import os
import argparse
from ..utils.file_utils import load_json, load_jsonl
from ..utils.object_utils import boot_draw_intervals
from ..utils.direction_utils import get_sample_info, bootstrap_sample


if __name__ == "__main__":
    # Test objects.py
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--input_path",
        type=str,
        default="objects/data",
        help="the input data for intervention",
    )
    parse.add_argument(
        "--forward_bias", action="store_true", default=False, help="check forward bias"
    )
    parse.add_argument(
        "--results_path",
        type=str,
        default="objects/data",
        help="the intervention results",
    )
    parse.add_argument(
        "--save_path", type=str, default="objects/data", help="the store file path"
    )
    parse.add_argument("--model_name", type=str, default="hamt", help="model name")
    parse.add_argument("--bootstrap_num", type=int, default=100)
    args = parse.parse_args()

    with launch_ipdb_on_exception():
        print("Testing objects on reverie dataset.py")
        # * change here
        result_path = args.results_path
        input_path = args.input_path
        save_dir = args.save_path
        save_dir = os.path.join(save_dir, args.model_name, "object")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if "envdrop" in result_path:
            submit_obj_reverie_sample = load_json(
                f"{result_path}/submit_reverie_obj_3m_sample.json"
            )
            submit_no_end_reverie_sample = load_json(
                f"{result_path}/submit_reverie_no_end_3m_sample.json"
            )
        else:
            submit_obj_reverie_sample = load_json(
                f"{result_path}/submit_reverie_obj_3m_sample_0.json"
            )
            submit_no_end_reverie_sample = load_json(
                f"{result_path}/submit_reverie_no_end_3m_sample_0.json"
            )

        obj_reverie_sample = load_jsonl(f"{input_path}/rxr_reverie_obj_3m_sample.jsonl")
        no_end_reverie_sample = load_jsonl(
            f"{input_path}/rxr_reverie_ahead_3m_sample.jsonl"
        )

        instr_id2obj_reverie = {
            item["instruction_id"]: item for item in obj_reverie_sample
        }
        instr_id2no_end_reverie = {
            item["instruction_id"]: item for item in no_end_reverie_sample
        }
        instr_id2intervene_result = {
            item["instr_id"]: item for item in submit_obj_reverie_sample
        }
        instr_id2no_intervene_result = {
            item["instr_id"]: item for item in submit_no_end_reverie_sample
        }

        cluster = ["scan_id", "path_id"]
        replace = [True, False]
        to_sample = get_sample_info(submit_obj_reverie_sample)
        bootstrap_sample_result = bootstrap_sample(
            to_sample, cluster=cluster, replace=replace, size=args.bootstrap_num
        )
        boot_draw_intervals(
            bootstrap_sample_result,
            instr_id2no_intervene_result,
            instr_id2no_end_reverie,
            instr_id2intervene_result,
            instr_id2obj_reverie,
            name1="No Intervention",
            name2="Object Intervention",
            xlabel_name="Absolute Difference: Predicted Heading vs Target Object Heading",
            use_next_gt_heading=False,
            gt_obj_gt_next=False,
            no_stop=True,
            accum_method="sum",
            polar=False,
            binsize=10,
            inverse_angular_diff=False,
            save_dir=save_dir,
            model_name=args.model_name,
        )


#  python normalized_model_prediction_obj_heading.py --results_path results/clip-envdrop --save_path camera-ready-new --model_name clip-envdro

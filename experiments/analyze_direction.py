import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from ipdb import launch_ipdb_on_exception

from utils.direction_utils import boot_draw_total, bootstrap_sample, get_sample_info
from utils.file_utils import load_json

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--result_path", type=str, default="rebuttal/more_templates")
    parse.add_argument(
        "--model_name",
        choices={"hamt", "envdrop-imagenet", "clip-envdrop"},
        type=str,
        default="hamt",
    )
    parse.add_argument("--save_dir", type=str, default="camera-ready")
    parse.add_argument("--baseline_dir", type=str, default="./baselines/more_templates")
    parse.add_argument("--turn_around", action="store_true", default=False)
    parse.add_argument("--combine", action="store_true", default=False)
    parse.add_argument("--more_templates", action="store_true", default=False)
    parse.add_argument("--bootstrap_num", type=int, default=100)

    args = parse.parse_args()

    result_path = f"{args.result_path}" + "/{}"
    baseline_path = f"{args.baseline_dir}" + "/{}"

    filename = f"direction_polar_{args.model_name}"
    save_path = f"{args.save_dir}/{args.model_name}/direction/{filename}_turn_around_{args.turn_around}_combine_{args.combine}.pdf"

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    print(f"save_path: {save_path}, result_path: {result_path}, filename: {filename} ")
    print(f"loading data from {result_path}")

    if not args.more_templates:
        # * below is the data name used for more templates situation!
        # filenames = {
        #     'forward': 'submit_new_forward_intervention_0.json',
        #     'backward': 'submit_new_backward_turn_around_intervention_0.json',
        #     'left': 'submit_new_left_intervention_0.json',
        #     'right': 'submit_new_right_intervention_0.json',
        #     # 'back_left': 'submit_new_back_left_no_combine_intervention_0.json',
        #     'back_left': 'submit_new_back_left_no_combine_intervention_0.json',
        #     # 'back_right': 'submit_new_back_right_no_combine_intervention_0.json',
        #     'back_right': 'submit_new_back_right_no_combine_intervention_0.json',
        #     'ahead_partial': 'submit_ahead_partial_0.json',
        #     'no_end': 'submit_no_end_paired_with_ahead_0.json'
        # }
        if args.model_name == "hamt":
            filenames = {
                "forward": "submit_new_walk_forward_0.json",
                "backward": "submit_new_walk_backward_0.json",
                "left": "submit_new_turn_left_0.json",
                "right": "submit_new_turn_right_0.json",
                # 'back_left': 'submit_new_back_left_no_combine_intervention_0.json',
                "back_left": "submit_new_turn_back_left_0.json",
                # 'back_right': 'submit_new_back_right_no_combine_intervention_0.json',
                "back_right": "submit_new_turn_back_right_0.json",
                # 'ahead_partial': 'submit_ahead_partial_0.json',
                "no_end": "submit_no_end_paired_with_ahead_0.json",
            }
        else:
            filenames = {
                "forward": "submit_new_walk_forward.json",
                "backward": "submit_new_walk_backward.json",
                "left": "submit_new_turn_left.json",
                "right": "submit_new_turn_right.json",
                # 'back_left': 'submit_new_back_left_no_combine_intervention_0.json',
                "back_left": "submit_new_turn_back_left.json",
                # 'back_right': 'submit_new_back_right_no_combine_intervention_0.json',
                "back_right": "submit_new_turn_back_right.json",
                # 'ahead_partial': 'submit_ahead_partial_0.json',
                "no_end": "submit_no_end_paired_with_ahead.json",
            }

    else:
        filenames = {
            "forward": "submit_new_forward_intervention.json",
            "backward": "submit_new_backward_no_turn_around_intervention.json",
            "left": "submit_new_left_intervention.json",
            "right": "submit_new_right_intervention.json",
            # 'back_left': 'submit_new_back_left_no_combine_intervention.json',
            "back_left": "submit_new_back_left_no_combine_intervention.json",
            # 'back_right': 'submit_new_back_right_no_combine_intervention.json',
            "back_right": "submit_new_back_right_no_combine_intervention.json",
            "ahead_partial": "submit_ahead_partial.json",
            "no_end": "submit_no_end_paired_with_ahead.json",
        }
    if args.turn_around:
        if args.model_name == "hamt":
            filenames[
                "backward"
            ] = "submit_new_backward_with_turn_around_intervention_0.json"
        else:
            filenames[
                "backward"
            ] = "submit_new_backward_with_turn_around_intervention.json"
    if args.combine:
        if args.model_name == "hamt":
            filenames["back_left"] = "submit_new_back_left_combine_intervention_0.json"
            filenames[
                "back_right"
            ] = "submit_new_back_right_combine_intervention_0.json"
        else:
            filenames["back_left"] = "submit_new_back_left_combine_intervention.json"
            filenames["back_right"] = "submit_new_back_right_combine_intervention.json"

    result_left = load_json(result_path.format(filenames["left"]))
    result_right = load_json(result_path.format(filenames["right"]))
    result_back_left = load_json(result_path.format(filenames["back_left"]))
    result_back_right = load_json(result_path.format(filenames["back_right"]))
    result_forward = load_json(result_path.format(filenames["forward"]))
    result_backward = load_json(result_path.format(filenames["backward"]))
    # result_ahead_partial = load_json(baseline_path.format(filenames['ahead_partial']))
    result_no_end = load_json(baseline_path.format(filenames["no_end"]))

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(21, 14))

    plt.style.use(["ggplot", "../data/vlnbehave.mplstyle"])
    # mpl.rcParams['axes.spines.bottom'] = False
    # mpl.rcParams['axes.spines.top'] = False
    # mpl.rcParams['axes.spines.left'] = False
    # mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams["legend.fontsize"] = 12

    directional_instructions = [
        [("Left", "left", result_left), ("Back Left", "back left", result_back_left)],
        [
            ("Forward", "forward", result_forward),
            ("Backward", "backward", result_backward),
        ],
        [
            ("Right", "right", result_right),
            ("Back Right", "back right", result_back_right),
        ],
    ]
    # bootstrap samples (since all the data is the same, we can just use one of them to sample some ids)

    cluster = ["scan_id", "path_id"]
    replace = [True, False]
    to_sample = get_sample_info(result_left)
    bootstrap_sample_result = bootstrap_sample(
        to_sample, cluster, replace, size=args.bootstrap_num
    )

    instr_id2no_intervene = {i["instr_id"]: i for i in result_no_end}

    with launch_ipdb_on_exception():
        for col, dir_grp in enumerate(directional_instructions):
            for row, dir in enumerate(dir_grp):
                name, filter_label, data = dir

                inter_id2intervene = {i["instr_id"]: i for i in data}

                save_dir = os.path.dirname(save_path)
                boot_draw_total(
                    bootstrap_sample_result,
                    instr_id2no_intervene,
                    inter_id2intervene,
                    "No Intervention",
                    name,
                    filter_label,
                    polar=True,
                    polar_ax=axs[row, col],  # type: ignore
                    binsize=10,
                    save_path=save_dir,
                    model_name=args.model_name,
                )

        plt.savefig(save_path, bbox_inches="tight", format="pdf")

# python draw_direction_nov_15.py --model_name clip-envdrop --result_path results/clip-envdrop --baseline_dir results/clip-envdrop --save_dir camera-ready-fix-boot --bootstrap_num 100

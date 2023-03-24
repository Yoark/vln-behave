# draw the abs difference between the object direction and the model predicted direction
import math
import os
from collections import defaultdict, namedtuple
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

from ..utils.file_utils import save_json

# set figure attr
# sns.set(color_codes=True)
# sns.set(rc={'figure.figsize':(12,10)})

category_mapping_dir = pathlib.Path('/Users/zijiao/home/research/unit-vln/Matterport/metadata/category_mapping.tsv')
filter_out_objects = {'misc', 'objects','unlabeled','void','wall', 'appliances', 'ceiling', 'floor', "lighting", "door", "railing", "stairs", "window", "shelving"}
go_toward_template = ["Walk towards the"]
category_mapping = pd.read_csv(category_mapping_dir, sep='\t')
image_width =  640
image_height = 480
wfov = math.radians(75)


def template_go_toward(instruction: str, obj_name: str, with_stop=False) -> str:
    if instruction[-1] == '.':
        new_instruction = instruction + ' ' + go_toward_template[0] + ' ' + obj_name + '.'
    else:
        new_instruction = instruction + '.' + ' ' + go_toward_template[0] + ' ' + obj_name + '.'
    if with_stop:
        new_instruction +=' This is your destination.' 
        
    return new_instruction

def get_viewId_heading(viewId):
    return (viewId % 12) * math.radians(30)


def dist_left_center_obj(bbox):
    """
    Compute the x coords of the center, which is center of bounding box in the image
    """
    return bbox[0] + bbox[2]/2

def add_heading_info(item: dict, only_one=False):
    """# this computes obj heading for each view at the viewpoint and store them."""
    # add avg heading info for object
    for obj_id,obj_attribs in item.items():
        # horizon_viewids = [viewId for viewId in obj_attribs['visible_pos'] if 12 <= viewId < 24]
        horizon_viewids = [viewId for viewId in obj_attribs['visible_pos']]
        # if not horizon_viewids:
            # continue
        # base_headings = [get_viewId_heading(viewid) for viewid in obj_attribs['visible_pos'] if 12 <= viewid <24]
        base_headings = [get_viewId_heading(viewid) for viewid in horizon_viewids]
        

        obj_attribs['norm_headings_radians'] = []
        obj_attribs['norm_headings_degree'] = []
        obj_attribs['has_bbox'] = True
        # obj_attribs['mod_headings'] = []
        obj_attribs['avg_heading'] = 0
        if not obj_attribs['bbox2d']:
            obj_attribs['has_bbox'] = False
            continue
        # box_w_h = []
        for bbox, base_heading in zip(obj_attribs['bbox2d'], base_headings):
            assert 0 <= base_heading <= 2*math.pi
            # rel_heading = (dist_left_center_obj(bbox) - image_width/2)/image_width * wfov
            rel_heading = np.arctan((dist_left_center_obj(bbox) - image_width/2)/415.692193817)
            # box_w_h.append(bbox[2] * bbox[3])
            # assert <= rel_heading <=
            norm_heading = base_heading + rel_heading # in radians
            # mod_heading = normalize_degree(norm_heading)
            obj_attribs['norm_headings_radians'].append(norm_heading)
            obj_attribs['norm_headings_degree'].append(math.degrees(norm_heading))
            if only_one:
                break
            # obj_attribs['mod_headings'].append(mod_heading)
        avg_angle_radians = average_angles(obj_attribs['norm_headings_radians'])
        # print("w*h: ", box_w_h)
        # print("average: ", np.rad2deg(avg_angle_radians))
            # avg_angle_degree = normalize_degree(math.degrees(avg_angle_radians))
        obj_attribs['avg_heading'] = avg_angle_radians # in radians


def average_angles(angles):
    # *
    """Average (mean) of angles

    Return the average of an input sequence of angles. The result is between
    ``0`` and ``2 * math.pi``.
    If the average is not defined (e.g. ``average_angles([0, math.pi]))``,
    a ``ValueError`` is raised.
    """
    
    # print("input:", np.rad2deg(angles))
    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)

    if x == 0 and y == 0:
        raise ValueError(
            "The angle average of the inputs is undefined: %r" % angles)

    # To get outputs from -pi to +pi, delete everything but math.atan2() here.
    return math.fmod(math.atan2(y, x) + 2 * math.pi, 2 * math.pi)

def get_mpcat40_from_raw_category(old_cat: str, category_mapping=category_mapping) -> str:
    if '#' in old_cat:
        old_cat = old_cat.replace("#", " ")

    new_cat = category_mapping[category_mapping['raw_category'] == old_cat]['mpcat40'].tolist()[0]
    if "_" in new_cat:
        new_cat = new_cat.replace("_", " ")
    return new_cat

def get_clean_objname(name):
    if '#' in name:
        name = name.replace("#", " ")
    return name

def obtain_obj_intervention_instances(item: dict, viewpoints: dict, template_func=template_go_toward, short_name=True):

    instr_id = item['instruction_id'].split('_')[0]
    # print(instr_id)

    viewpoint_id = item['path'][-1]
    
    try:
        obj_attrbs = viewpoints[viewpoint_id]
    except KeyError:
        return 'does not exist'

    objects = namedtuple('objects', ['object_id', 'object_name','object_avg_heading'])
    # submit_result = [sub for sub in no_end_submit if '_'.join(sub['instr_id'].split('_')[1:]) == item['instruction_id']][0]

    # get objects for the viewpoint 
    object_collection = []
    for obj_id, obj in obj_attrbs.items():
        # skip those without bbox
        if not obj['has_bbox']:
            continue
        if short_name:
            name = get_mpcat40_from_raw_category(obj['name'])
        else:
            cat_name = get_mpcat40_from_raw_category(obj['name'])
            name = get_clean_objname(obj['name'])
        if name in filter_out_objects:
            continue
        object_collection.append(objects(obj_id, name, obj['avg_heading']))    

    # get object intervention instances without rep object names
    saw_objs = set()
    obj_instances = []
    for i, obj in enumerate(object_collection):
        new_item = copy.deepcopy(item)

        if obj.object_name in saw_objs:
            continue
        saw_objs.add(obj.object_name)
        # heading_diff_deg = diff_with_direction(obj.object_avg_heading, submit_result['heading'], radians=True)
        # new_item['heading_diff_gt'] = heading_diff_deg # this should be useless
        new_item['obj_name'] = obj.object_name
        new_item['obj_heading'] = obj.object_avg_heading
        # new_item['agent_heading'] = submit_result['heading'] # this should be useless
        # assert -180 <= heading_diff_deg <= 180
        # modify temlate as needed
        instruction = template_func(item['instruction'], obj.object_name)
        new_item['instruction'] = instruction
        new_item['instruction_id'] = "_".join([new_item['instruction_id'], str(obj.object_name)])
        obj_instances.append(new_item)
    return obj_instances

def get_obj_instances_for_all(no_end, viewpoints, template_func=None, short_name=True):
    """for loop to ease process
    """
    total_instances = []
    for item in tqdm(no_end):
        obj_instances = obtain_obj_intervention_instances(item, viewpoints, template_func=template_func, short_name=short_name) # type: ignore
        if obj_instances and not isinstance(obj_instances, str):
            total_instances.extend(obj_instances)
    return total_instances

def template_nothing(instruction, obj_name):
    # *
    # but still check the end punctuation
    if instruction[-1] == '.':
        return instruction
    else: 
        return instruction[:-1] + '.'

def filter_instances_by_objects_heading(result_data, instr2input_data, fov_object=60):
    # *
    """
    filter out instances that does not have a candidate-object match
    """
    inst_with_matched_obj_cand = []

    for item in result_data:
        candidate_headings = [
            cand["normalized_heading"] for cand in item["candidates"][0][1]
        ]  # rel heading
        gt = instr2input_data["_".join(item["instr_id"].split("_")[1:])]
        object_gt_heading = gt["obj_heading"]
        candidate_contain_object_within_fov = [
            abs(diff_with_direction(object_gt_heading, cand_heading)) < fov_object / 2
            for cand_heading in candidate_headings
        ]
        if (
            any(candidate_contain_object_within_fov)
            and len(candidate_contain_object_within_fov) > 2
        ):
            inst_with_matched_obj_cand.append(item["instr_id"])
    return inst_with_matched_obj_cand


# --- filters-end -------------------------------------------------------------------------------------------------------
def binning_pairs(all_heading_probs_pairs, bin_range=(0, 181), binsize=30):
    # *
    """bin pairs into bins with binsize from 0,to 180.

    Args:
        all_heading_probs_pairs : list of pairs of (angle diff, prob)
        binsize :  Defaults to 30.

    Returns:
        dict: dict[bin_name] = [prob1, ...]
    """
    ranges = list(np.arange(bin_range[0], bin_range[1], binsize))
    direction_pairs = defaultdict(list)
    for item in all_heading_probs_pairs:
        for start, end in zip(ranges, ranges[1:]):
            if start <= item["abs_diff"] < end:
                mid = start + binsize / 2
                direction_pairs[mid].append(item["prob"])
                break
    return direction_pairs


def diff_with_direction(d1, d2, radians=True):
    # *
    """compute diff between d1 and d2 and return result into -180, 180 range"""
    if radians:
        diff = np.rad2deg(d1) - np.rad2deg(d2)
    else:
        diff = d1 - d2
    diff = np.mod(diff, 360)
    if diff > 180:
        diff = np.mod(diff, -360)
    return diff


def get_interval_probs_pairs(
    item,
    instr_id2input_data,
    with_dir=True,
    use_next_gt_heading=False,
    gt_obj_gt_next=False,
    uniform_prob=False,
    **kwargs,
):
    # *
    """compute obj heading model result heading diff, prob pair for one instance.

    Returns:
        tuple: pairs of diff, prob, and stop prob
    """
    candidate_headings = [
        cand["normalized_heading"] for cand in item["candidates"][0][1]
    ]  # rel heading
    cand_probs = item["final_logits"][0][: len(candidate_headings)]
    assert len(candidate_headings) == len(cand_probs)

    gt = instr_id2input_data["_".join(item["instr_id"].split("_")[1:])]

    object_gt_heading = gt["obj_heading"]
    stop_prob = item["final_logits"][0][len(candidate_headings)]

    if uniform_prob:
        cand_probs = [1 / (len(cand_probs) + 1)] * len(cand_probs)
        stop_prob = cand_probs[0]

    scan_id = item["candidates"][0][1][0]["scanId"]
    step_id = item["instr_id"].split("_")[2]
    path_id = item["instr_id"].split("_")[0]
    result_samples = []

    if kwargs["inverse_angular_diff"]:

        def normalize_heading(head_to_process):
            # *
            pi = np.pi
            if 0 <= head_to_process < pi:
                head_to_process = head_to_process
            if -pi <= head_to_process < 0:
                head_to_process = head_to_process
            if pi <= head_to_process <= 2 * pi:
                head_to_process = head_to_process % -2 * pi
            if -2 * pi <= head_to_process <= -pi:
                head_to_process = head_to_process % 2 * pi
            return head_to_process

        cand_probs = [
            1 / (0.1 + abs(normalize_heading(cand_heading)))
            for cand_heading in candidate_headings
        ]
        sum_cand_probs = np.sum(cand_probs)
        for cand in candidate_headings:
            prob = 1 / (0.1 + abs(normalize_heading(cand)))
            result_samples.append(
                {
                    "abs_diff": np.abs(diff_with_direction(cand, gt["obj_heading"])),
                    "prob": prob / sum_cand_probs,
                    "scan_id": scan_id,
                    "path_id": path_id,
                    "step_id": step_id,
                }
            )
        stop_prob = 1 - np.sum(cand_probs)
    if use_next_gt_heading and not gt_obj_gt_next:
        for prob, cand in zip(cand_probs, candidate_headings):
            sam = (
                np.abs(diff_with_direction(cand, gt["next_heading"])),
                prob,
                scan_id,
                step_id,
                path_id,
            )
            result_samples.append(
                {
                    "abs_diff": sam[0],
                    "prob": sam[1],
                    "scan_id": sam[2],
                    "step_id": sam[3],
                    "path_id": sam[4],
                }
            )
    elif not use_next_gt_heading and not gt_obj_gt_next:
        for prob, cand in zip(cand_probs, candidate_headings):
            sam = (
                np.abs(diff_with_direction(cand, object_gt_heading)),
                prob,
                scan_id,
                step_id,
                path_id,
            )
            result_samples.append(
                {
                    "abs_diff": sam[0],
                    "prob": sam[1],
                    "scan_id": sam[2],
                    "step_id": sam[3],
                    "path_id": sam[4],
                }
            )

    elif gt_obj_gt_next:
        for prob, cand in zip(cand_probs, candidate_headings):
            sam = (
                np.abs(diff_with_direction(object_gt_heading, item["next_gt_heading"])),
                1,
                scan_id,
                step_id,
                path_id,
            )
            result_samples.append(
                {
                    "abs_diff": sam[0],
                    "prob": sam[1],
                    "scan_id": sam[2],
                    "step_id": sam[3],
                    "path_id": sam[4],
                }
            )

    return result_samples, stop_prob


def get_all_interval_probs(
    result_data,
    instr_id2input_data,
    indexs=[],
    binsize=10,
    with_dir=True,
    use_next_gt_heading=False,
    gt_obj_gt_next=False,
    uniform_prob=False,
    **kwargs,
):
    # *
    """
    loop through model results and get ((abs diff between model pred and obj heading probability), stop_prob, instr_ids) triplets.
    """

    all_pairs = []
    idx = []
    stops = []
    for item in result_data:
        if indexs and item["instr_id"] in indexs:
            continue
        result = get_interval_probs_pairs(
            item,
            instr_id2input_data,
            with_dir=with_dir,
            use_next_gt_heading=use_next_gt_heading,
            gt_obj_gt_next=gt_obj_gt_next,
            uniform_prob=uniform_prob,
            **kwargs,
        )

        if result:
            all_pairs.extend(result[0])
            stops.append(result[1])
        else:
            idx.append(item["instr_id"])
            continue
    return all_pairs, stops, set(idx)


def boot_draw_intervals(
    bootstrap_sample_result,
    submit1,
    gt1,
    submit2,
    gt2,
    name1,
    name2,
    xlabel_name="absolute difference between ground truth object direction and model_prediction",
    use_next_gt_heading=False,
    gt_obj_gt_next=False,
    no_stop=False,
    uniform_prob=False,
    accum_method="sum",
    polar=False,
    binsize=10,
    bin_range=(0, 181),
    width=4,
    save_dir='',
    **kwargs,
):
    # *
    """given two model results: submit1 and submit2, along with the ground truth,
    draw the difference between obj direction and model direction.

    Args:
        submit1 (_type_):
        gt1 (_type_): _description_
        submit2 (_type_): _description_
        gt2 (_type_): _description_
        name1 (_type_): _description_
        name2 (_type_): _description_

    Returns:
        Tuple: (probs1, probs2, binned_probs1, binned_probs2)
    """
    plt.style.use("default")
    plt.style.use(
        [
            "./image_style/paper2.mplstyle",
        ]
    )
    mpl.rcParams["legend.fontsize"] = 25
    mpl.rcParams["ytick.labelsize"] = 25
    mpl.rcParams["xtick.labelsize"] = 25
    mpl.rcParams["axes.labelsize"] = 30

    plt.tight_layout()
    from file_util import my_colors

    colors = my_colors

    model_name = kwargs.get("model_name", "hamt")

    intervene_dists, no_intervene_dists = [], []
    intervene_stops, no_intervene_stops = [], []

    # Process each bootstrap sample
    for sample in tqdm(bootstrap_sample_result):
        intervene_dist, inter_stop = process_object_result(
            sample["instr_id"], gt2, submit2, bin_size=binsize, **kwargs
        )
        no_intervene_dist, no_inter_stop = process_object_result(
            sample["instr_id"], gt1, submit1, bin_size=binsize, **kwargs
        )
        intervene_dists.append(intervene_dist)
        no_intervene_dists.append(no_intervene_dist)
        intervene_stops.append(inter_stop)
        no_intervene_stops.append(no_inter_stop)

    # Compute the CI
    CI_low_intervene = (
        pd.concat(intervene_dists, axis=0)
        .sort_values("abs_diff")
        .groupby("abs_diff")
        .quantile(0.025)
    )
    CI_high_intervene = (
        pd.concat(intervene_dists, axis=0)
        .sort_values("abs_diff")
        .groupby("abs_diff")
        .quantile(0.975)
    )
    CI_low_intervene_stop = np.quantile(intervene_stops, 0.025)
    CI_high_intervene_stop = np.quantile(intervene_stops, 0.975)

    CI_low_no_intervene = (
        pd.concat(no_intervene_dists, axis=0)
        .sort_values("abs_diff")
        .groupby("abs_diff")
        .quantile(0.025)
    )
    CI_high_no_intervene = (
        pd.concat(no_intervene_dists, axis=0)
        .sort_values("abs_diff")
        .groupby("abs_diff")
        .quantile(0.975)
    )
    CI_low_no_intervene_stop = np.quantile(no_intervene_stops, 0.025)
    CI_high_no_intervene_stop = np.quantile(no_intervene_stops, 0.975)
    # Compute dist
    boot_overall = pd.concat(bootstrap_sample_result, axis=0)
    # ! compute the distribution by mean
    mean_intervene = (
        pd.concat(intervene_dists, axis=0)
        .sort_values("abs_diff")
        .groupby("abs_diff")
        .mean()
    )
    mean_no_intervene = (
        pd.concat(no_intervene_dists, axis=0)
        .sort_values("abs_diff")
        .groupby("abs_diff")
        .mean()
    )

    # intervene_overall_dist, intervene_stop_prob = process_object_result(boot_overall['instr_id'], gt2, submit2, bin_size=binsize, **kwargs)
    # no_inter_overall_dist, no_intervene_stop_prob = process_object_result(boot_overall['instr_id'], gt1, submit1, bin_size=binsize, **kwargs)
    # intervene_overall_dist = intervene_overall_dist.sort_values('abs_diff').set_index('abs_diff')
    # no_inter_overall_dist = no_inter_overall_dist.sort_values('abs_diff').set_index('abs_diff')

    # compute mean of stop prob by averaging stop prob of each bootstrap sample
    intervene_stop_overall = np.mean(intervene_stops)
    no_intervene_stop_overall = np.mean(no_intervene_stops)

    # combine all the data into a dataframe
    # intervene_collection_dframe = pd.concat([intervene_overall_dist, CI_low_intervene, CI_high_intervene], axis=1)
    intervene_collection_dframe = pd.concat(
        [mean_intervene, CI_low_intervene, CI_high_intervene], axis=1
    )
    intervene_collection_dframe.columns = ["dist", "CI_low_dist", "CI_high_dist"]
    no_inter_collection_dframe = pd.concat(
        [mean_no_intervene, CI_low_no_intervene, CI_high_no_intervene], axis=1
    )
    # no_inter_collection_dframe = pd.concat([no_inter_overall_dist, CI_low_no_intervene, CI_high_no_intervene], axis=1)
    no_inter_collection_dframe.columns = ["dist", "CI_low_dist", "CI_high_dist"]
    to_record = {
        # 'direction': direction,
        "model_name": model_name,
        "sample_size": len(boot_overall),
        "sample_repeat_num": len(bootstrap_sample_result),
        "intervene": {
            "stop": intervene_stop_overall,
            # 'dist': intervene_overall_dist.to_dict('index'),
            # 'CI_low_dist' : CI_low_intervene,
            # 'CI_high_dist': CI_high_intervene,
            "dist_related": intervene_collection_dframe.to_dict("index"),
            "CI_low_stop": CI_low_intervene_stop,
            "CI_high_stop": CI_high_intervene_stop,
        },
        "no_intervene": {
            "stop": no_intervene_stop_overall,
            # 'dist': no_inter_overall_dist,
            # 'CI_low_dist' : CI_low_no_intervene,
            # 'CI_high_dist': CI_high_no_intervene,
            "dist_related": no_inter_collection_dframe.to_dict("index"),
            "CI_low_stop": CI_low_no_intervene_stop,
            "CI_high_stop": CI_high_no_intervene_stop,
        },
    }
    # save the computed result.
    save_file = os.path.join(save_dir, f"{model_name}_object.json")
    save_json(save_file, to_record)
    # --- end of new cycle

    if polar:
        bin_range = (-180, 180)

    # ! now each item in bin contain everything

    # intervene_collection_dframe
    no_intervene_dist, no_intervene_low, no_intervene_high, no_intervene_bins = (
        no_inter_collection_dframe["dist"].to_numpy(),
        no_inter_collection_dframe["CI_low_dist"].to_numpy(),
        no_inter_collection_dframe["CI_high_dist"].to_numpy(),
        no_inter_collection_dframe.index.to_numpy(dtype=np.int32),
    )
    (
        with_intervene_dist,
        with_intervene_low,
        with_intervene_high,
        with_intervene_bins,
    ) = (
        intervene_collection_dframe["dist"].to_numpy(),
        intervene_collection_dframe["CI_low_dist"].to_numpy(),
        intervene_collection_dframe["CI_high_dist"].to_numpy(),
        intervene_collection_dframe.index.to_numpy(dtype=np.int32),
    )

    x = (no_intervene_bins, with_intervene_bins)
    assert x[0].tolist() == x[1].tolist()
    # y = list(list(zip(*no_end_mean))[1])
    y = no_intervene_dist
    z = with_intervene_dist

    yerr_low = y[np.newaxis, :] - no_intervene_low[np.newaxis, :] # type: ignore
    yerr_high = no_intervene_high[np.newaxis, :] - y[np.newaxis, :] # type: ignore

    zerr_low = z[np.newaxis, :] - with_intervene_low[np.newaxis, :] # type: ignore
    zerr_high = with_intervene_high[np.newaxis, :] - z[np.newaxis, :] # type: ignore

    err_bin1 = np.concatenate((yerr_low, yerr_high), axis=0)
    err_bin2 = np.concatenate((zerr_low, zerr_high), axis=0)

    # width=4
    if not polar:
        if not no_stop:
            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(16, 9),
                gridspec_kw={"width_ratios": [8, 1]},
                sharey="row",
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))
        if (
            accum_method == "avg"
            or accum_method == "bootstrap"
            or accum_method == "sum"
        ):
            if (
                not uniform_prob
                and not gt_obj_gt_next
                and not kwargs["inverse_angular_diff"]
            ):
                ax1.bar(
                    x[0] - width / 2,
                    y,
                    yerr=err_bin1,
                    width=width,
                    label=name1,
                    color=colors[0],
                    error_kw=dict(ecolor="black", ls="-", lw=2, capsize=5, capthick=2),
                    zorder=3,
                )
                ax1.bar(
                    x[1] + width / 2,
                    z,
                    yerr=err_bin2,
                    width=width,
                    label=name2,
                    color=colors[1],
                    error_kw=dict(ecolor="black", ls="-", lw=2, capsize=5, capthick=2),
                    zorder=3,
                )
            else:
                if uniform_prob:
                    ax1.bar(
                        x,
                        y,
                        width=width,
                        label="Random Agent",
                        color=colors[0],
                        yerr=err_bin1,
                        error_kw=dict(
                            ecolor="black", ls="-", lw=2, capsize=5, capthick=2
                        ),
                    )
                elif gt_obj_gt_next:
                    ax1.bar(
                        x,
                        y,
                        width=width,
                        label="Ground Truth Agent",
                        color=colors[0],
                        yerr=err_bin1,
                        error_kw=dict(
                            ecolor="black", ls="-", lw=2, capsize=5, capthick=2
                        ),
                    )
                elif kwargs["inverse_angular_diff"]:
                    ax1.bar(
                        x[0],
                        y,
                        width=width,
                        label="Forward Bias Agent",
                        color=colors[0],
                        zorder=3,
                    )

        ax1.set_xticks(x[0])
        ax1.set_xticklabels(list(map(int, list(x[0]))))
        ax1.set_xlabel(xlabel_name)
        ax1.set_ylabel("Density")
        if kwargs["inverse_angular_diff"]:
            ymax = max(
                max([y1 for y1, ci in zip(y, err_bin1[1])]),
                max([y2 for y2, ci in zip(z, err_bin2[1])]),
            )
        else:
            ymax = max(
                max([y1 + ci for y1, ci in zip(y, err_bin1[1])]),
                max([y2 + ci for y2, ci in zip(z, err_bin2[1])]),
            )
        ymax = math.ceil(ymax * 100) / 100
        ax1.set_ylim(0, ymax)
        ax1.tick_params(axis="both", which="major", direction="out", labelrotation=0)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

        width = 1.3
        # if not no_stop:
        #     if accum_method == "avg":
        #         if not uniform_prob:
        #             ax2.bar(
        #                 0.05 - width / 80,
        #                 norm_stop_prob1,
        #                 yerr=stop_ci1,
        #                 width=width / 40,
        #                 color="g",
        #                 label=name1,
        #             )
        #             ax2.bar(
        #                 0.05 + width / 80,
        #                 norm_stop_prob2,
        #                 yerr=stop_ci2,
        #                 width=width / 40,
        #                 color="b",
        #                 label=name2,
        #             )
        #         else:
        #             ax2.bar(
        #                 0.05,
        #                 norm_stop_prob1,
        #                 yerr=stop_ci1,
        #                 width=width / 40,
        #                 color="g",
        #                 label="random agent",
        #             )
        #     else:
        #         if not uniform_prob:
        #             ax2.bar(
        #                 0.05 - width / 80,
        #                 norm_stop_prob1,
        #                 width=width / 40,
        #                 color="g",
        #                 label=name1,
        #             )
        #             ax2.bar(
        #                 0.05 + width / 80,
        #                 norm_stop_prob2,
        #                 width=width / 40,
        #                 color="b",
        #                 label=name2,
        #             )
        #         else:
        #             ax2.bar(
        #                 0.05,
        #                 norm_stop_prob1,
        #                 width=width / 40,
        #                 color="g",
        #                 label="random agent",
        #             )
        #     ax2.set_xlim(-0.05, 0.15)
        #     ax2.set_xticks([0.05], ["stop"])
        #     # ax2.legend()
        #     ax2.tick_params(
        #         axis="both",
        #         which="major",
        #         direction="out",
        #         length=10,
        #         width=5,
        #         color="red",
        #         pad=15,
        #         labelsize=20,
        #         labelcolor="black",
        #         labelrotation=0,
        #     )

        xlabel_name = "_".join(xlabel_name.split())
        plt.savefig(f"{save_dir}/{xlabel_name}.pdf", bbox_inches="tight", format="pdf")
    elif polar:
        from fig_help.fig_util import angularDensityProb

        values = np.empty((len(x), 2))
        values[:, 0] = y
        values[:, 1] = z
        ang_input = (x, values, [name1, name2])
        ticks = np.linspace(-180, 180, 12, endpoint=False, dtype=int)
        angularDensityProb(ang_input, edge=0.3, ticks=ticks)


def process_object_result(sampled_ids, instr_id2data, instr_id2result, **kwargs):
    # *
    selected_samples = [instr_id2result[i] for i in sampled_ids]

    use_next_gt_heading = kwargs.get("use_next_gt_heading", False)
    gt_obj_gt_next = kwargs.get("gt_obj_gt_next", False)
    uniform_prob = kwargs.get("uniform_prob", False)
    bin_range = kwargs.get("bin_range", (0, 181))
    binsize = kwargs.get("binsize", 10)

    filter_indexes = filter_instances_by_objects_heading(
        selected_samples, instr_id2data, fov_object=30
    )

    filtered_result = [
        item for item in selected_samples if item["instr_id"] in filter_indexes
    ]
    pairs, stops, _ = get_all_interval_probs(
        filtered_result,
        instr_id2data,
        with_dir=False,
        use_next_gt_heading=use_next_gt_heading,
        gt_obj_gt_next=gt_obj_gt_next,
        uniform_prob=uniform_prob,
        **kwargs,
    )

    bins = binning_pairs(pairs, binsize=binsize, bin_range=bin_range)

    total_probs_exclude_stop = np.sum([item for k, v in bins.items() for item in v])
    total_probs_include_stop = total_probs_exclude_stop + np.sum(stops)

    norm_accum_prob = defaultdict(list)
    stop_prob = np.sum(stops) / total_probs_include_stop
    for k, v in bins.items():
        norm_accum_prob[k] = np.sum(v) / total_probs_exclude_stop

    return (
        pd.DataFrame(data=norm_accum_prob.items(), columns=["abs_diff", "prob"]),
        stop_prob,
    )

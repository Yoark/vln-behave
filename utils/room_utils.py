import copy
import json
import os
import pathlib
import random
from collections import defaultdict

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from utils.direction_utils import bootstrap_sample, get_sample_info, hierarchy_resample
from utils.file_utils import load_json, load_jsonl, my_colors

connectivity_work_dir = pathlib.Path(
    "/Users/zijiao/home/research/Matterport3DSimulator/connectivity"
)

room_type2region_name = {
    "a": "bathroom",
    "b": "bedroom",
    "c": "closet",
    "d": "dining room",
    "e": "entryway/foyer/lobby",
    "f": "familyroom",
    "g": "garage",
    "h": "hallway",
    "i": "library",
    "j": "laundryroom/mudroom",
    "k": "kitchen",
    "l": "living room",
    "m": "meetingroom/conferenceroom",
    "n": "lounge",
    "o": "office",
    "p": "porch/terrace/deck/driveway",
    "r": "rec/game",
    "s": "stairs",
    "t": "toilet",
    "u": "utilityroom/toolroom",
    "v": "tv",
    "w": "workout/gym/exercise",
    "x": "outdoor",
    "y": "balcony",
    "z": "other room",
    "B": "bar",
    "C": "classroom",
    "D": "dining booth",
    "S": "spa/sauna",
    "Z": "junk",
    "-": "no label",
}


def append_room_instruction(instruction: str, room_name: str, with_stop=False) -> str:
    # *
    template = "Walk towards the"
    if instruction[-1] == ".":
        new_instruction = instruction + " " + template + " " + room_name + "."
    else:
        new_instruction = instruction + "." + template + " " + room_name + "."
    if with_stop:
        new_instruction += " This is your destination."
    return new_instruction


def weight_1(start, end, attr_name):
    # *
    return 1


def closest_distance(scan, shortest_distances, goal_viewpoints_lst, agent_location):
    # *
    """
    find the shortest distance from agent_location to any goal_viewpoints

    Return: the shortest distance goal and shortest ditsance
    """
    shortest_dist = float("inf")
    shortest_goal = None
    for goal in goal_viewpoints_lst:
        dist = shortest_distances[scan][agent_location][goal]

        if dist < shortest_dist:
            shortest_dist = dist
            shortest_goal = goal
    return shortest_goal, shortest_dist


def room_type2room_names(room_type2region_name, room_type):
    if room_type in room_type2region_name:
        name = room_type2region_name[room_type]
        return name
    else:
        raise ValueError("room_type {} not in room_type2region_name".format(room_type))


def convert_room_type(room_type2region_name, room_type):
    if room_type in room_type2region_name:
        if "/" in room_type2region_name[room_type]:
            names = room_type2region_name[room_type].split("/")
            name = random.choice(names)
        else:
            name = room_type2region_name[room_type]
        return name
    else:
        raise ValueError("room_type {} not in room_type2region_name".format(room_type))


def load_nav_graphs(scans):
    # *
    """Load connectivity graph for each scan"""

    def distance(pose1, pose2):
        """Euclidean distance between two graph poses"""
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    graphs = {}
    for scan in scans:
        with open(connectivity_work_dir / ("%s_connectivity.json" % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            positions[item["image_id"]] = np.array(
                                [item["pose"][3], item["pose"][7], item["pose"][11]]
                            )
                            assert data[j]["unobstructed"][
                                i
                            ], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=distance(item, data[j]),
                            )
            nx.set_node_attributes(G, values=positions, name="position")
            graphs[scan] = G
    return graphs


# we need the graphs for each scan
# we need end supervision viewpointid (yes) and next viewpointId also final viewpointId
# label of the instance is the goal, goal viewpointIds


def process_result_n_hop(results, datasets, scanregion=None, argmax=False, **kwargs):
    # *
    #
    #    delta=args.use_delta, scanregion=scan_region, instr_id2intervene_dataset=instr_id2intervene_data,
    #    instr_id2no_intervene_result=instr_id2no_intervene_result)
    """process 'submit' result

    Args:
        results (list): inference result
        datasets (list): input
        argmax (bool): if True, use argmax next action.

    Returns:
        (dict): a dictionary of data collections
    """
    intervene_dataset, no_intervene_dataset = datasets
    result_intervene, result_no_intervene = results
    if not scanregion:
        with open(
            "/Users/zijiao/home/research/unit-vln/code/region_data/scan_region.json"
        ) as t:
            scan_region = json.load(t)
    else:
        scan_region = scanregion

    scans = list(scan_region.keys())
    graphs = load_nav_graphs(scans)

    shortest_paths = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G, weight=weight_1))  # type: ignore
    shortest_distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    # load inference data.
    # make dictionaly of all the results so we have much faster processing speed (indeed the case)
    # TODO make this work for both delta and dist2goal plots
    # instr_id2no_intervene_result = {item['instr_id']: item for item in result_no_intervene}
    # instr_id2intervene_dataset = {item['instruction_id']: item for item in intervene_dataset}
    instr_id2no_intervene_result = kwargs["instr_id2no_intervene_result"]
    instr_id2intervene_dataset = kwargs["instr_id2intervene_dataset"]

    final_deltas = []
    next_deltas = []
    next_delta_intervenes = []
    next_delta_no_intervenes = []
    final_delta_intervenes = []
    final_delta_no_intervenes = []

    roomtype2delta = defaultdict(list)

    dist_to_goal_intervenes_final = []
    dist_to_goal_no_intervenes_final = []

    dist_to_goal_intervenes = []
    dist_to_goal_no_intervenes = []
    dist_to_goal_stops = []
    result_data = {}
    dist_to_goal_next_weighted_avg_intervene = []
    dist_to_goal_next_weighted_avg_no_intervene = []

    to_boot_strap = []
    for item in result_intervene:
        scan_id = item["candidates"][0][1][0]["scanId"]
        path_id = item["instr_id"].split("_")[0]
        step_id = int(item["instr_id"].split("_")[2])

        instr_id = "_".join(item["instr_id"].split("_")[1:])
        train_instance = instr_id2intervene_dataset[instr_id]
        final_viewpoint = item["final_viewpoint_id"][0]
        stop_viewpoint = train_instance["stop_viewpoint_id"]
        next_viewpoint = item["next_pos"]
        result_instr_id = item["instr_id"]

        goal_cand_indexes = train_instance["goal_cand_indexes"]

        scan_id = train_instance["scan"]

        room_type = train_instance["region_type"]
        room_name = train_instance["region_name"]
        no_intervention_item = instr_id2no_intervene_result[result_instr_id]

        goal_viewpoints_lst = [target for _, target in goal_cand_indexes]
        _, dist_stop_to_goal = closest_distance(
            scan_id, shortest_distances, goal_viewpoints_lst, stop_viewpoint
        )
        dist_to_goal_stops.append(
            {
                "path_id": path_id,
                "dist2goal": dist_stop_to_goal,
                "scan_id": scan_id,
                "step_id": step_id,
            }
        )

        _, dist_intervene_to_goal_final = closest_distance(
            scan_id, shortest_distances, goal_viewpoints_lst, final_viewpoint
        )
        # dist_to_goal_intervenes_final.append(dist_intervene_to_goal_final)
        dist_to_goal_intervenes_final.append(
            {
                "path_id": path_id,
                "dist2goal": dist_intervene_to_goal_final,
                "scan_id": scan_id,
                "step_id": step_id,
            }
        )

        _, dist_no_intervene_to_goal_final = closest_distance(
            scan_id,
            shortest_distances,
            goal_viewpoints_lst,
            no_intervention_item["final_viewpoint_id"][0],
        )
        # dist_to_goal_no_intervenes_final.append(dist_no_intervene_to_goal_final)
        dist_to_goal_no_intervenes_final.append(
            {
                "path_id": path_id,
                "dist2goal": dist_no_intervene_to_goal_final,
                "scan_id": scan_id,
                "step_id": step_id,
            }
        )

        def get_cand_dist_and_prob(item, goal_viewpoints_lst):
            # *
            """use to compute weighted next delta"""
            candidate_len = len(item["candidates"][0][1])
            cand_probs = item["final_logits"][0][:candidate_len]

            dist_to_goal_next_and_prob_pairs = []
            for i, cand in enumerate(item["candidates"][0][1]):
                cand_prob = cand_probs[i]
                cand_viewpoint = cand["viewpointId"]
                _, cand_dist_intervene_to_goal_next = closest_distance(
                    scan_id, shortest_distances, goal_viewpoints_lst, cand_viewpoint
                )
                dist_to_goal_next_and_prob_pairs.append(
                    (cand_dist_intervene_to_goal_next, cand_prob)
                )
            return dist_to_goal_next_and_prob_pairs

        if argmax:
            _, dist_intervene_to_goal_next = closest_distance(
                scan_id, shortest_distances, goal_viewpoints_lst, next_viewpoint
            )
            dist_to_goal_intervenes.append(dist_intervene_to_goal_next)
            _, dist_no_intervene_to_goal_next = closest_distance(
                scan_id,
                shortest_distances,
                goal_viewpoints_lst,
                no_intervention_item["next_pos"],
            )
            dist_to_goal_no_intervenes.append(dist_no_intervene_to_goal_next)

            next_delta_intervene = -(dist_intervene_to_goal_next - dist_stop_to_goal)
            next_delta_no_intervene = -(
                dist_no_intervene_to_goal_next - dist_stop_to_goal
            )
            next_delta_intervenes.append(next_delta_intervene)
            next_delta_no_intervenes.append(next_delta_no_intervene)

            final_delta_intervene = -(dist_intervene_to_goal_final - dist_stop_to_goal)
            final_delta_no_intervene = -(
                dist_no_intervene_to_goal_final - dist_stop_to_goal
            )
            final_delta_intervenes.append(final_delta_intervene)
            final_delta_no_intervenes.append(final_delta_no_intervene)

            roomtype2delta[room_type].append(
                (
                    next_delta_intervene,
                    next_delta_no_intervene,
                    final_delta_intervene,
                    final_delta_no_intervene,
                )
            )

        else:
            d2g_intervene = get_cand_dist_and_prob(item, goal_viewpoints_lst)
            dist_to_goal_intervenes.extend(d2g_intervene)
            d2g_no_intervene = get_cand_dist_and_prob(
                no_intervention_item, goal_viewpoints_lst
            )
            dist_to_goal_no_intervenes.extend(d2g_no_intervene)

            dists, weights = [list(n) for n in zip(*d2g_intervene)]
            d2g_avg_yes = np.average(dists, weights=weights)

            dist_to_goal_next_weighted_avg_intervene.append(
                {
                    "path_id": path_id,
                    "dist2goal": d2g_avg_yes,
                    "scan_id": scan_id,
                    "step_id": step_id,
                }
            )

            dists, weights = [list(n) for n in zip(*d2g_no_intervene)]
            d2g_avg_no = np.average(dists, weights=weights)

            dist_to_goal_next_weighted_avg_no_intervene.append(
                {
                    "path_id": path_id,
                    "dist2goal": d2g_avg_no,
                    "scan_id": scan_id,
                    "step_id": step_id,
                }
            )

            next_delta_intervene = -(d2g_avg_yes - dist_stop_to_goal)
            next_delta_no_intervene = -(d2g_avg_no - dist_stop_to_goal)
            # ? is this structure correct for bootstrap?
            next_delta_intervenes.append(
                {
                    "path_id": path_id,
                    "delta": next_delta_intervene,
                    "scan_id": scan_id,
                    "step_id": step_id,
                }
            )
            next_delta_no_intervenes.append(
                {
                    "path_id": path_id,
                    "delta": next_delta_no_intervene,
                    "scan_id": scan_id,
                    "step_id": step_id,
                }
            )
            next_deltas.append(
                {
                    "path_id": path_id,
                    "scan_id": scan_id,
                    "step_id": step_id,
                    "intervene_delta": next_delta_intervene,
                    "no_intervene_delta": next_delta_no_intervene,
                }
            )
            # * this is argmax result
            final_delta_intervene = -(dist_intervene_to_goal_final - dist_stop_to_goal)
            final_delta_no_intervene = -(
                dist_no_intervene_to_goal_final - dist_stop_to_goal
            )
            final_delta_intervenes.append(final_delta_intervene)
            final_delta_no_intervenes.append(final_delta_no_intervene)

            roomtype2delta[room_type].append(
                (
                    next_delta_intervene,
                    next_delta_no_intervene,
                    final_delta_intervene,
                    final_delta_no_intervene,
                )
            )

        to_boot_strap.append(
            {
                "scan_id": scan_id,
                "path_id": path_id,
                "step_id": step_id,
                "stop2goal": dist_stop_to_goal,
                "intervene_next2goal": dist_to_goal_next_weighted_avg_intervene,
                "no_intervene_next2goal": dist_to_goal_next_weighted_avg_no_intervene,
                "intervene_final2goal": dist_intervene_to_goal_final,
                "no_intervene_final2goal": dist_no_intervene_to_goal_final,
            }
        )

    result_data = {
        "final_delta": final_deltas,
        "next_delta": next_deltas,
        "next_delta_intervene": next_delta_intervenes,
        "next_delta_no_intervene": next_delta_no_intervenes,
        "final_delta_intervene": final_delta_intervenes,
        "final_delta_no_intervene": final_delta_no_intervenes,
        "dist_to_goal_stops": dist_to_goal_stops,
        "dist_to_goal_intervene_next": dist_to_goal_intervenes,
        "dist_to_goal_no_intervene_next": dist_to_goal_no_intervenes,
        "dist_to_goal_no_intervene_final": dist_to_goal_no_intervenes_final,
        "dist_to_goal_intervene_final": dist_to_goal_intervenes_final,
        "roomtype2delta": roomtype2delta,
        "dist_to_goal_next_weighted_avg_intervene": dist_to_goal_next_weighted_avg_intervene,
        "dist_to_goal_next_weighted_avg_no_intervene": dist_to_goal_next_weighted_avg_no_intervene,
    }

    return result_data


def draw_distribution_on_delta_to_goal_room(
    result_data,
    title="data distribution by distance to goal",
    hop="",
    filetype="png",
    save_dir="",
    argmax=False,
    maxValue=None,
    delta=False,
    bootstrap=False,
):
    # *

    """draw the distribution of the data on the distance to the goal"""
    plt.tight_layout()
    colors = my_colors
    plt.style.use(["ggplot", "../data/vlnbehave.mplstyle"])
    # mpl.rc('axes', labelsize=30'
    mpl.rcParams["legend.fontsize"] = 30
    mpl.rcParams["ytick.labelsize"] = 25
    mpl.rcParams["xtick.labelsize"] = 25
    mpl.rcParams["axes.labelsize"] = 25

    fig_name = "distribution_on_dist_to_goal_" + str(hop)

    if delta:
        fig_name = "distribution_on_delta_to_goal" + str(hop)
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        x_deltas = np.arange(
            -5, 5, 0.5
        ).tolist()  # ? this delta distance x axis is too specific is there a better way?
        rr = (-5, 5)
        maxys = []
        barWidth = 0.3
        bar_colors = [colors[0], colors[1]]
        zero_bucket_colors = ["#699EC9", "#EA5050"]
        if not argmax:
            data_to_draw = result_data["next_delta"]
        else:
            final_delta_intervene = result_data["final_delta_intervene"]
            final_delta_no_intervene = result_data["final_delta_no_intervene"]
            data_to_draw = [final_delta_intervene, final_delta_no_intervene]

        bar_pos_sets = [-barWidth / 2, barWidth / 2]

        next_delta = pd.DataFrame(data_to_draw)
        cluster = ["scan_id", "path_id"]
        replace = [True, False]

        intervene_bin_numbers_collection = []
        no_intervene_bin_numbers_collection = []
        boot_sample_collection = []
        intervene_mean_collection = []
        no_intervene_mean_collection = []
        number_of_samples = []

        for i in range(100):
            boot_sample = hierarchy_resample(next_delta, cluster, replace)
            number_of_samples.append(boot_sample.shape[0])
            intervene_next_delta = boot_sample["intervene_delta"]
            intervene_mean_collection.append(np.mean(boot_sample["intervene_delta"]))
            bin_number, bins = np.histogram(
                intervene_next_delta, bins=x_deltas, range=rr, density=True
            )
            intervene_bin_numbers_collection.append(bin_number.tolist())

            no_intervene_next_delta = boot_sample["no_intervene_delta"]
            no_intervene_mean_collection.append(
                np.mean(boot_sample["no_intervene_delta"])
            )
            bin_number, bins = np.histogram(
                no_intervene_next_delta, bins=x_deltas, range=rr, density=True
            )
            no_intervene_bin_numbers_collection.append(bin_number.tolist())

            boot_sample_collection.append(boot_sample)

        delta_names = ["intervene_delta", "no_intervene_delta"]
        for i, (
            bin_numbers_collection,
            mean_collection,
            delta_name,
            color,
            zcolor,
            bar_pos_set,
        ) in enumerate(
            zip(
                [intervene_bin_numbers_collection, no_intervene_bin_numbers_collection],
                zip([intervene_mean_collection, no_intervene_mean_collection]),
                delta_names,
                bar_colors,
                zero_bucket_colors,
                bar_pos_sets,
            )
        ):
            print(f"{delta_name} mean {np.mean(mean_collection)}")
            CI_low = []
            CI_high = []
            err_low = []
            err_high = []
            means = []
            for bins in zip(*bin_numbers_collection):
                # mean = np.mean(bins)
                low, high = np.percentile(bins, [2.5, 97.5])
                CI_low.append(low)
                CI_high.append(high)
                means.append(np.mean(bins))

            CI_low = np.array(CI_low)
            CI_high = np.array(CI_high)
            means = np.array(means)

            bins = x_deltas
            bins = np.round(bins, 1)
            # number_of_zeros = len([n for n in next_delta if n == 0])
            names = list(zip(bins, bins[1:]))

            # create a another bucket so that we could stack the 0
            datas = np.array(means)  # should be a vector of length 19
            # import ipdb; ipdb.set_trace()
            err_low = datas - CI_low  # type: ignore
            err_high = CI_high - datas  # type: ignore
            yerror = np.vstack([err_low, err_high])
            assert yerror.shape[0] == 2

            maxy = np.max([np.max(item) for item in [datas, CI_low, CI_high]])
            x = np.arange(len(names))
            # * add boostrap sampling and CI here.

            ax.bar(
                x + bar_pos_set,
                datas,
                yerr=yerror,
                width=barWidth,
                color=color,
                label="Nonzero Delta",
                capsize=7,
                error_kw=dict(ecolor="black", ls="-", lw=2, capsize=5, capthick=2),
            )

            maxys.append(maxy)
        names = [f"[{s}, {e})" for s, e in names]  # type: ignore
        ax.set_xticks([r for r in range(len(names))], names)
        ax.set_ylabel("Density")
        ax.set_xlabel("Delta Distances to Goal (Relative to Start Position)")

        ax.tick_params(axis="x", rotation=50)
        maxy = np.max(maxys)
        ax.set_ylim(0, np.round(maxy + 0.05, decimals=1))
        # ax.set_aspect('equal')
        ax.grid(True, which="both")
        # generate legend
        import matplotlib.patches as patches

        labels = ["Room Intervention", "No Intervention"]
        legend_patches = [
            patches.Patch(color=colors[j], label=labels[j], alpha=0.5) for j in range(2)
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
        )

    if save_dir:
        save_dir = f"{save_dir}/{fig_name}.pdf"
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        # plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        print(os.path.dirname(save_dir))
        plt.savefig(save_dir, format="pdf", bbox_inches="tight")


def create_exact_x_hop_dataset_with_no_region_loop(
    no_end, result_no_end, x, scan_viewpoint_label, with_stop=False
):
    # *

    scan_viewpoint_label = load_json("./region_data/accurate_scan_viewpoint_label.json")
    scans = list(scan_viewpoint_label.keys())

    # scans = list(scan_region.keys())
    graphs = load_nav_graphs(scans)

    def weight_1(start, end, attr_name):
        return 1

    # build shortest paths without using weight!! such that we only consider the number of hops!
    weight_1_shortest_paths = {}
    for scan, G in graphs.items():  # compute all shortest paths
        weight_1_shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G, weight=weight_1))  # type: ignore

    # use this to get scan id
    instr_id2no_end = {item["instruction_id"]: item for item in no_end}
    inter_id2nhop_region_label = defaultdict(lambda: defaultdict(list))
    region2hop_nums = defaultdict(list)
    for inference in result_no_end:
        instr_id = "_".join(inference["instr_id"].split("_")[1:3])
        scan_id = instr_id2no_end[instr_id]["scan"]

        cand_ids = [
            inf_cand["viewpointId"] for inf_cand in inference["candidates"][0][1]
        ]
        stop_viewpoint_id = inference["candidates"][0][0]

        # ! not used in current version
        cand_ids = cand_ids + [stop_viewpoint_id]
        for i, (target, path) in enumerate(
            weight_1_shortest_paths[scan_id][stop_viewpoint_id].items()
        ):
            assert isinstance(path[0], str)
            target_region_label = scan_viewpoint_label[scan_id][target]
            stop_region_label = scan_viewpoint_label[scan_id][stop_viewpoint_id]
            # We don't want 1. target region is the same as the stop region 2. target region is the same as one of the insight region
            if (
                len(path) <= x + 1
                and target_region_label != stop_region_label
                and target_region_label
            ):  #  and target not in cand_ids:
                region2hop_nums[target_region_label].append(len(path) - 1)
                inter_id2nhop_region_label[instr_id][target_region_label].append(
                    (i, target)
                )

    intervene_dataset = []
    no_intervene_dataset = []

    for instr_id, regions in inter_id2nhop_region_label.items():
        inter_lst = []
        no_inter_lst = []
        for region_label, target in regions.items():
            if region_label not in "zZ":
                intervene_item = copy.deepcopy(instr_id2no_end[instr_id])
                no_intervene_item = copy.deepcopy(instr_id2no_end[instr_id])
                region_name = convert_room_type(room_type2region_name, region_label)

                intervene_item["instruction_id"] = (
                    intervene_item["instruction_id"] + "_" + region_name
                )
                intervene_item["instruction"] = append_room_instruction(
                    intervene_item["instruction"], region_name, with_stop=with_stop
                )
                intervene_item["region_name"] = region_name
                intervene_item["region_type"] = region_label
                intervene_item["goal_cand_indexes"] = target
                intervene_item["stop_viewpoint_id"] = intervene_item["path"][-1]

                no_intervene_item["instruction_id"] += "_" + region_name
                no_intervene_item["region_name"] = region_name
                no_intervene_item["region_type"] = region_label
                no_intervene_item["goal_cand_indexes"] = target
                no_intervene_item["stop_viewpoint_id"] = no_intervene_item["path"][-1]

                inter_lst.append(intervene_item)
                no_inter_lst.append(no_intervene_item)
            else:
                continue
        intervene_dataset.extend(inter_lst)
        no_intervene_dataset.extend(no_inter_lst)

    result_dict = {
        "intervene_dataset": intervene_dataset,
        "no_intervene_dataset": no_intervene_dataset,
        "inter_id2nhop_region_label": inter_id2nhop_region_label,
        "region2hop_nums": region2hop_nums,
    }
    return result_dict

    # print(instr_id, target, region_labels)


def load_hop_data_and_results(hops, data_dir, result_dir, args):
    # *
    hop_data_list = []
    hop_result_list = []

    for hop in hops:
        hop_data = load_jsonl(
            f"{data_dir}/{hop}_hop/rxr_{hop}_hop_intervene_dataset.jsonl"
        ), load_jsonl(f"{data_dir}/{hop}_hop/rxr_{hop}_hop_no_intervene_dataset.jsonl")
        hop_data_list.append(hop_data)

        if "envdrop" in args.model_name:
            hop_result = load_json(
                f"{result_dir}/{hop}_hop/submit_{hop}_hop_intervene_dataset.json"
            ), load_json(
                f"{result_dir}/{hop}_hop/submit_{hop}_hop_no_intervene_dataset.json"
            )
            hop_result_list.append(hop_result)
        else:
            hop_result = load_json(
                f"{result_dir}/{hop}_hop/submit_{hop}_hop_intervene_dataset_0.json"
            ), load_json(
                f"{result_dir}/{hop}_hop/submit_{hop}_hop_no_intervene_dataset_0.json"
            )
            hop_result_list.append(hop_result)

    return hop_data_list, hop_result_list


def generate_hop_results(hops, hop_data_list, hop_result_list, delta=False, **kwargs):
    # *
    # hops = [1, 2,3,4,5, 6, 7, 8]
    hop_nums = []
    dist_to_goal_intervene_collects = []
    dist_to_goal_no_intervene_collects = []
    dist_to_goal_stops_collects = []

    delta_intervene_collects = []
    delta_no_intervene_collects = []
    hop_results = {}
    scan_region = kwargs.get("scanregion", None)

    # for hop in hops:
    for idx, hop in enumerate(hops):
        hop_data = hop_data_list[idx]
        hop_result = hop_result_list[idx]
        hop_processed = process_result_n_hop(
            hop_result, hop_data, argmax=False, **kwargs
        )
        dist_to_goal_stops = hop_processed["dist_to_goal_stops"]
        dist_to_goal_intervenes_next = hop_processed["dist_to_goal_intervene_next"]
        dist_to_goal_no_intervenes_next = hop_processed[
            "dist_to_goal_no_intervene_next"
        ]
        dist_to_goal_intervenes_final = hop_processed["dist_to_goal_intervene_final"]
        dist_to_goal_no_intervenes_final = hop_processed[
            "dist_to_goal_no_intervene_final"
        ]
        next_intervene = hop_processed["dist_to_goal_next_weighted_avg_intervene"]
        next_no_intervene = hop_processed["dist_to_goal_next_weighted_avg_no_intervene"]

        next_delta_intervene, next_delta_no_intervene = (
            hop_processed["next_delta_intervene"],
            hop_processed["next_delta_no_intervene"],
        )
        final_delta_intervene, final_delta_no_intervene = (
            hop_processed["final_delta_intervene"],
            hop_processed["final_delta_no_intervene"],
        )

        if hop == 1:
            hop_nums.extend([hop] * len(next_intervene))
            # next_intervene['hop'] = hop
            next_intervene = [x["dist2goal"] for x in next_intervene]
            dist_to_goal_intervene_collects.extend(next_intervene)
            # next_no_intervene['hop'] = hop
            next_no_intervene = [x["dist2goal"] for x in next_no_intervene]
            dist_to_goal_no_intervene_collects.extend(next_no_intervene)

            delta_intervene_collects.extend(next_delta_intervene)
            delta_no_intervene_collects.extend(next_delta_no_intervene)
        else:
            hop_nums.extend([hop] * len(dist_to_goal_intervenes_final))
            # dist_to_goal_intervenes_final['hop'] = hop
            dist_to_goal_intervenes_final = [
                x["dist2goal"] for x in dist_to_goal_intervenes_final
            ]
            dist_to_goal_intervene_collects.extend(dist_to_goal_intervenes_final)
            # dist_to_goal_no_intervenes_final['hop'] = hop

            dist_to_goal_no_intervenes_final = [
                x["dist2goal"] for x in dist_to_goal_no_intervenes_final
            ]
            dist_to_goal_no_intervene_collects.extend(dist_to_goal_no_intervenes_final)

            delta_intervene_collects.extend(final_delta_intervene)
            delta_no_intervene_collects.extend(final_delta_no_intervene)

        dist_to_goal_stops = [x["dist2goal"] for x in dist_to_goal_stops]
        dist_to_goal_stops_collects.extend(dist_to_goal_stops)
        # draw_distribution_on_dist_to_goal(hop_processed, hop=hop)
        # print(len(hop_nums), len(dist_to_goal_intervene_collects), len(dist_to_goal_no_intervene_collects), len(dist_to_goal_stops_collects))
        if delta:
            hop_results = pd.DataFrame(
                {
                    "hop": hop_nums,
                    "delta_in": delta_intervene_collects,
                    "dist_to_goal_no_intervene": delta_no_intervene_collects,
                }
            )  #'dist_to_goal_stops': dist_to_goal_stops_collects})
        else:
            # hop_results = pd.DataFrame({'hop': hop_nums, 'dist_to_goal_intervene': dist_to_goal_intervene_collects, 'dist_to_goal_no_intervene': dist_to_goal_no_intervene_collects, 'dist_to_goal_stops': dist_to_goal_stops_collects})
            hop_results = {
                "hop": hop_nums,
                "dist_to_goal_intervene": dist_to_goal_intervene_collects,
                "dist_to_goal_no_intervene": dist_to_goal_no_intervene_collects,
                "dist_to_goal_stops": dist_to_goal_stops_collects,
            }
    return hop_results


def get_hboot_samples(n_samples=50, data=None, cluster=None, replace=None):
    # *
    to_sample = get_sample_info(data)
    bootstrap_sample_result = bootstrap_sample(
        to_sample, cluster=cluster, replace=replace, size=n_samples
    )
    return bootstrap_sample_result


# plt.style.use('default')
def instr_id2instruction_id(instr_id):
    # *
    return "_".join(instr_id.split("_")[1:])


def create_ridge_plot(hops, hop_results, data_dir, filename, delta=False, **kwargs):
    # pal = sns.color_palette(palette='Set2')
    # *

    from file_util import my_colors

    colors = my_colors
    # pal = sns.color_palette(palette='Set2') use the max differnce color palette
    plt.rcParams["legend.fontsize"] = 25
    plt.rcParams["font.size"] = 35
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.facecolor"] = (0, 0, 0, 0)
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.labelsize"] = 35
    # g = sns.FacetGrid(hop_results, row='hop', aspect=10, height=0.8, palette=pal)
    g = sns.FacetGrid(hop_results, row="hop", aspect=10, height=2, palette=colors)
    if len(hops) == 1:
        g.figure.set_size_inches(20, 4.5)
    else:
        g.figure.set_size_inches(20, 15)

    # labels = ['Dist to Goal Stops', 'Dist to Goal No Intervene', 'Dist to Goal Intervene']
    labels = kwargs.get("labels", [])
    names = kwargs.get("names", [])
    if delta:
        rg = (-10, 10)
    else:
        rg = (-5, 30)

    for i, name in enumerate(names):
        g.map(
            custom_kdeplot,
            "x",
            name,
            color=colors[i],
            label=labels[i],
            alpha=0.3,
            linewidth=3.0,
        )

    xmin, xmax = rg  # (-5, 30)
    g.map(plt.axhline, y=0, lw=2, clip_on=False, xmin=0, xmax=1, color="k")

    for i, ax in enumerate(g.axes.flat):  # type: ignore
        ax.text(
            -5,
            0.02,
            hops[i],
            fontweight="bold",
            fontsize=35,
            color=ax.lines[-1].get_color(),
        )
    if len(hops) == 1:
        g.axes[0, 0].text(-7.5, 0.2, "Densities", rotation=90)  # type: ignore
    else:
        g.axes[3, 0].text(-7.5, 0, "Densities", rotation=90)  # type: ignore
    g.fig.subplots_adjust(hspace=0)

    # eventually we remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=[], xlim=(xmin, xmax))
    # g.set(xlim=(xmin, xmax))
    g.despine(bottom=True, left=True)
    # g.despine(bottom=True)
    if len(hops) == 1:
        if delta:
            g.set_xlabels(
                "Delta Distances to Goal (Relative to Start Position)", labelpad=15
            )
        else:
            g.set_xlabels("Next Loc to Goal Distance", labelpad=15)
    else:
        if delta:
            g.set_xlabels(
                "Delta Distances to Goal (Relative to Start Position)", labelpad=15
            )
        else:
            g.set_xlabels("Final Loc to Goal Distance", labelpad=15)
    g.set_ylabels("")
    if not delta:
        legend_patches = [
            patches.Patch(color=colors[j], label=labels[j], alpha=0.5) for j in range(3)
        ]
        # legend_patches = [mlines.Line2D([], [], color=colors[j], label=labels[j], alpha=0.9, linewidth=3.5) for j in range(3)]
    else:
        legend_patches = [
            patches.Patch(color=colors[j], label=labels[j], alpha=0.9) for j in range(2)
        ]

    if not delta:
        g.add_legend(
            handles=legend_patches, ncol=3, bbox_to_anchor=(0.31, 1.05), loc="center"
        )
    else:
        g.add_legend(
            handles=legend_patches, ncol=2, bbox_to_anchor=(0.3, 1.05), loc="center"
        )
    # plt.show()
    plt.savefig(f"{data_dir}/{filename}.pdf", format="pdf", bbox_inches="tight")


def custom_kdeplot(x, y, label, color, alpha=0.3, linewidth=3.5):
    # *
    sns.lineplot(x=x, y=y, label=label, color=color, alpha=1.0, linewidth=linewidth)
    plt.fill_between(x, y, alpha=alpha, color=color)

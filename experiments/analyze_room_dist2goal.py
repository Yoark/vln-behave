import os
import json
import argparse
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm
from ipdb import launch_ipdb_on_exception

from rooms.room_utils import (
    generate_hop_results,
    create_ridge_plot,
    load_hop_data_and_results,
)
from rooms.room_utils import get_hboot_samples, instr_id2instruction_id


def get_bootstraped_kde(n_hops, hop_data, hop_result, names, boot_num=10, args={}, **kwargs):
    # n_hops: list holding the number of hops

    rg = kwargs.get("rg", (-1, 1))
    x_vals = np.linspace(rg[0], rg[1], args.kde_num)
    n_hop_result_holder = np.empty((len(n_hops), 3, len(x_vals)))
    for k, n in enumerate(tqdm(n_hops)):
        print(f"{n} hops")
        hdata = hop_data[k]
        # import ipdb; ipdb.set_trace()
        instr_id2intervene_data = {item["instruction_id"]: item for item in hdata[0]}
        instr_id2no_intervene_data = {item["instruction_id"]: item for item in hdata[1]}
        hresult = hop_result[k]
        instr_id2intervene_result = {item["instr_id"]: item for item in hresult[0]}
        instr_id2no_intervene_result = {item["instr_id"]: item for item in hresult[1]}
        # * just get boot ids so which result don't matter
        boot_samples = get_hboot_samples(
            n_samples=boot_num, data=hresult[0], cluster=cluster, replace=replace
        )
        # kde placeholder
        y_vals = np.empty(
            (3, boot_num, len(x_vals))
        )  # 0: dist_to_goal_stops, 1: dist_to_goal_no_intervene, 2: dist_to_goal_intervene
        for i, sample in enumerate(tqdm(boot_samples)):
            sample_intervene_data = [
                instr_id2intervene_data[instr_id2instruction_id(instr_id)]
                for instr_id in sample["instr_id"]
            ]
            sample_no_intervene_data = [
                instr_id2no_intervene_data[instr_id2instruction_id(instr_id)]
                for instr_id in sample["instr_id"]
            ]
            sample_intervene_result = [
                instr_id2intervene_result[instr_id] for instr_id in sample["instr_id"]
            ]
            sample_no_intervene_result = [
                instr_id2no_intervene_result[instr_id]
                for instr_id in sample["instr_id"]
            ]
            sample_data = [sample_intervene_data, sample_no_intervene_data]
            sample_result = [sample_intervene_result, sample_no_intervene_result]

            hop_collect = generate_hop_results(
                [n],
                hop_data_list=[sample_data],
                hop_result_list=[sample_result],
                delta=args.use_delta,
                scanregion=scan_region,
                instr_id2intervene_dataset=instr_id2intervene_data,
                instr_id2no_intervene_result=instr_id2no_intervene_result,
            )

            for j, name in enumerate(names):
                kde = gaussian_kde(hop_collect[name], bw_method="scott")
                y_vals[j, i, :] = kde.evaluate(x_vals)

        y_vals = np.mean(y_vals, axis=1)  # (3, len(x_vals))
        n_hop_result_holder[k, ...] = y_vals  # (len(n_hops), 3, len(x_vals))

    dfs = []
    for (
        i,
        arr,
    ) in enumerate(n_hop_result_holder):
        n_hop_df = pd.DataFrame(arr.T, columns=names)
        n_hop_df["hop"] = n_hops[i]
        n_hop_df["x"] = x_vals
        dfs.append(n_hop_df)
    dfs = pd.concat(dfs, axis=0, ignore_index=True)  #
    return dfs, x_vals


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        args = argparse.ArgumentParser()
        args.add_argument("--data_dir", type=str, default="rooms/data_Nov_10_new")
        args.add_argument("--result_dir", type=str, default="rooms/data_Nov_10_new")
        args.add_argument("--save_dir", type=str, default="results/room/hamt")
        args.add_argument("--model_name", type=str, default="hamt")
        args.add_argument("--use_delta", action="store_true", default=False)
        args.add_argument("--bootstrap_num", type=int, default=10)
        args.add_argument("kde_num", type=int, default=500)
        args = args.parse_args()

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        hops = [2, 3, 4, 5, 6, 7, 8]
        data_dir = args.data_dir
        save_dir = args.save_dir

        save_dir = os.path.join(args.save_dir, args.model_name + "/room")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(
            "/Users/zijiao/home/research/unit-vln/code/region_data/scan_region.json"
        ) as t:
            scan_region = json.load(t)

        result_dir = args.result_dir

        boot_num = args.bootstrap_num
        rg = (-5, 30)
        names = [
            "dist_to_goal_stops",
            "dist_to_goal_no_intervene",
            "dist_to_goal_intervene",
        ]

        cluster = ["scan_id", "path_id"]
        replace = [True, False]
        # load data
        n_hops = [2, 3, 4, 5, 6, 7, 8]
        hop_n_data, hop_n_results = load_hop_data_and_results(
            n_hops, data_dir=data_dir, result_dir=result_dir, args=args
        )
        n_hop_dfs, x_vals = get_bootstraped_kde(
            n_hops,
            hop_data=hop_n_data,
            hop_result=hop_n_results,
            names=names,
            boot_num=boot_num,
            rg=rg,
            args=args
        )
        one_hops = [1]
        hop_1_data, hop_1_result = load_hop_data_and_results(
            one_hops, data_dir=data_dir, result_dir=result_dir, args=args
        )
        one_hop_dfs, x_vals = get_bootstraped_kde(
            one_hops,
            hop_data=hop_1_data,
            hop_result=hop_1_result,
            names=names,
            boot_num=boot_num,
            rg=rg,
            args=args
        )
        # hop_1_result = generate_hop_results(hops=one_hops, hop_data_list=hop_data, hop_result_list=hop_result, delta=args.use_delta)
        # result 9630_29193_5_stairs
        # data 25384_3_living room

        plt.style.use(["ggplot", "../data/vlnbehave.mplstyle"])
        hop_result_name_1 = f"hop_results_{args.model_name}_1_{args.use_delta}_{args.kde_num}_{args.bootstrap_num}"
        hop_result_name_2_8 = f"hop_results_{args.model_name}_2_8_{args.use_delta}_{args.kde_num}_{args.bootstrap_num}"

        one_labels = [
            "Stop to Goal",
            "Next Loc to Goal (No Intervention)",
            "Next Loc to Goal (Intervention)",
        ]
        n_hop_labels = [
            "Stop to Goal",
            "Final Loc to Goal (No Intervention)",
            "Final Loc to Goal (Intervention)",
        ]
        create_ridge_plot(
            n_hops,
            hop_results=n_hop_dfs,
            data_dir=save_dir,
            filename=hop_result_name_2_8,
            delta=args.use_delta,
            cluster=cluster,
            replace=replace,
            scanregion=scan_region,
            boot_num=5,
            names=names,
            labels=n_hop_labels,
        )
        create_ridge_plot(
            one_hops,
            hop_results=one_hop_dfs,
            data_dir=save_dir,
            filename=hop_result_name_1,
            delta=args.use_delta,
            cluster=cluster,
            replace=replace,
            scanregion=scan_region,
            boot_num=5,
            names=names,
            labels=one_labels,
        )

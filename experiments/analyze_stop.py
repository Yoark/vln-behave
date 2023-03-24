import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..utils.file_utils import load_json
from ..utils.direction_utils import bootstrap_sample, get_sample_info, boot_sample_stop_mean, get_mean_CI_for_each_step
from ..utils.direction_utils import draw_dists_with_confidence_tri


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, default='stop/data', help="other options are 'results/envdrop-imagenet' and 'results/envdrop-imagenet")
    args.add_argument('--model_name', type=str, default='hamt')
    # args.add_argument('--intervene', type=str, default='stop')
    args.add_argument('--save_dir', type=str, default='camera-ready', help="this is the root of your save directory")
    args.add_argument('--bootstrap_num', type=int, default=50)
    args = args.parse_args()
    
    data_dir = args.data_dir
    save_dir = os.path.join(args.save_dir , args.model_name, 'stop')
    # data_dir = 'stop/data'
    # save_dir = 'rebuttal/stop/hamt'
    # save_dir = 'camera-ready/hamt/stop'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
                    
    if 'envdrop' in data_dir:
        no_end_result = load_json(
            f'{data_dir}/submit_no_end_paired_with_ahead.json')
        with_end_result = load_json(
            f'./{data_dir}/submit_with_end_paired_with_ahead.json')
        ahead_partial_result = load_json(f'./{data_dir}/submit_ahead_partial.json')
    else:
        no_end_result = load_json(
            f'{data_dir}/submit_no_end_paired_with_ahead_0.json')
        with_end_result = load_json(
            f'./{data_dir}/submit_with_end_paired_with_ahead_0.json')
        ahead_partial_result = load_json(f'./{data_dir}/submit_ahead_partial_0.json')
    
    # return are dataframes 
    print("---start bootstrapping---")
    # * bootstrap all together
    # * hierarchical bootstrapping parameters
    cluster = ['scan_id', 'path_id']
    replace = [True, False]
    to_sample = get_sample_info(no_end_result)
    bootstrap_sample_result = bootstrap_sample(to_sample, cluster, replace, size=50)

    instr_id2no_end = {item['instr_id']: item for item in no_end_result}
    instr_id2with_end = {item['instr_id']: item for item in with_end_result}
    instr_id2ahead_partial = {item['instr_id']: item for item in ahead_partial_result}
    #-------------------------start

    no_end_collects, with_end_collects, ahead_partial_collects = [], [], []
    for sample in tqdm(bootstrap_sample_result):
        no_end_collect = boot_sample_stop_mean(sample, instr_id2no_end)
        with_end_collect = boot_sample_stop_mean(sample, instr_id2with_end)
        ahead_partial_collect = boot_sample_stop_mean(sample, instr_id2ahead_partial)
        no_end_collects.append(no_end_collect)
        with_end_collects.append(with_end_collect)
        ahead_partial_collects.append(ahead_partial_collect)

    no_end_boot = get_mean_CI_for_each_step(no_end_collects)
    with_end_boot = get_mean_CI_for_each_step(with_end_collects)
    ahead_partial_boot = get_mean_CI_for_each_step(ahead_partial_collects)
    #---------------------------end
    # size = 500
    print("---end bootstrapping---")
    print(f"no_end{no_end_boot['mean_over_all_steps'].iloc[0]}")
    print(f"with_end{with_end_boot['mean_over_all_steps'].iloc[0]}")
    print(f"ahead_partial{ahead_partial_boot['mean_over_all_steps'].iloc[0]}")

    plt.style.use(['ggplot', '../data/vlnbehave.mplstyle'])
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    # mpl.rcParams['font.size'] = 25
    # mpl.rcParams['legend.fontsize'] = 30
    # mpl.rcParams['ytick.labelsize'] = 30
    # mpl.rcParams['xtick.labelsize'] = 30
    # mpl.rcParams['axes.labelsize'] = 100
    draw_dists_with_confidence_tri(no_end_boot, with_end_boot, ahead_partial_boot, name1='Implicit Stop Instruction', name2='Explicit Stop Instruction',
                                name3='One-step Ahead Prior', ax=ax, curve=False, xlabel='Trajectory Length', ylabel='Average Stop Probability', savedir=f'./{save_dir}/stop.pdf')

   
import pathlib as path
import numpy as np
import copy
from collections import defaultdict
# from ..stats import bootstrapping
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
import pandas as pd

import string
from tqdm import trange

from ..utils.file_utils import save_json


# /Users/zijiao/home/research/data/RxR/pose_traces
pose_trace_dir = path.Path('/Users/zijiao/home/research/data/RxR/pose_traces')
split = 'rxr_val_unseen'
rxr_data_path = path.Path('/Users/zijiao/home/research/data/RxR/visualizations/rxr_data')

def boot_sample_stop_mean(sample, instr_id2result):
    # * process the sample
    sampled = [instr_id2result[instr_id] for instr_id in sample['instr_id']]
    pd_sampled = create_scan_id_intervene_label_stop_prob(sampled)
    overall_mean = pd_sampled['stop_prob'].mean()
    boot_result = get_stop_mean_by_step(pd_sampled)
    return pd_sampled, overall_mean, boot_result

def get_mean_CI_for_each_step(boot_collects):
    # * get the mean and CI for each step
    boot_result = pd.concat([item[2] for item in boot_collects], axis=0)
    CI_low = boot_result.sort_values('step').groupby('step')['mean'].quantile(0.025)
    CI_high = boot_result.sort_values('step').groupby('step')['mean'].quantile(0.975)
    mean = boot_result.sort_values('step').groupby('step')['mean'].mean()
    overall_mean = np.mean([item[1] for item in boot_collects])
    pd_result = pd.DataFrame(data={'mean': mean, 'CI_low': CI_low, 'CI_high': CI_high, 'step': mean.index, 'mean_over_all_steps': overall_mean})
    return pd_result

def hierarchy_resample(data, cluster, replace):
    # *
    """ a hierarchical bootstrap function for correlated data

    Args:
        data (change this to pandas dataframe): 
        cluster (list): hierarchical order from left to right
        replace (list): whether to sample with replacement at each hierarchical level
    """
    # exit early
    if data.shape[0] == 1 or not any(replace):
        return (data)
    
    unique_cluster = data.drop_duplicates(cluster[0])[cluster[0]]
    # print(unique_cluster.head())
    # print(unique_cluster.shape)
    cls = unique_cluster.sample(replace=replace[0], frac=1.0)
  
    
  # subset on the sampled clustering factors
    sub = cls.apply(lambda b: data[data[cluster[0]]==b])
    
  # sample lower levels of hierarchy (if any)
    if len(cluster) > 1:
        sub = sub.apply(lambda b: hierarchy_resample(b, cluster[1:], replace[1:]))
    
    
  # join and return samples
    sub = sub.tolist()
    return pd.concat(sub, axis=0)

def get_stop_by_step(stop_samples, cluster, replace, size=500, intervene=None):
    # this function do the bootstrapping for each step
    step_nums = sorted(stop_samples['step_id'].unique())
    step_means = []
    CI_low = []
    CI_high = []
    steps = []
    intervenes=[]
    
    for step in step_nums:
        means = []
        step_subset = stop_samples[stop_samples['step_id'] == step]
        for i in range(size):
            boot = hierarchy_resample(step_subset, cluster, replace)
            means.append(boot['stop_prob'].mean())

        low, high = np.percentile(means, [2.5, 97.5])

        step_means.append(np.mean(means))
        CI_low.append(low)
        CI_high.append(high)
        steps.append(step)
        intervenes.append(intervene)
    # return pd.DataFrame({'step': steps, 'mean': step_means, 'CI_low': CI_low, 'CI_high': CI_high, 'intervention_yes_no': intervenes})
    return pd.DataFrame({'step': steps, 'mean': step_means, 'CI_low': CI_low, 'CI_high': CI_high, 'intervention_yes_no': intervenes})

def get_stop_mean_by_step(stop_samples):
    #* compute the mean of stop probability for each step
    step_nums = sorted(stop_samples['step_id'].unique())
    step_means = []
    
    for step in step_nums:
        step_subset = stop_samples[stop_samples['step_id'] == step]
        mean = step_subset['stop_prob'].mean()
        step_means.append(mean)

    return pd.DataFrame({'step': step_nums, 'mean': step_means})


def get_viewpoint_time_pair(sample):
    # * load pose trace data, and match viewpoints in path with time

    instruction_id = sample['instruction_id']
    pose_name = f'{instruction_id:06}_guide_pose_trace.npz'
    pose_trace = np.load(pose_trace_dir/split/pose_name)
    
    viewpoint_time_index = []
    for viewpoint in sample['path']:
        try:
            index = pose_trace['pano'].tolist().index(viewpoint)
            viewpoint_time_index.append((viewpoint, pose_trace['time'][index]))
        except:
            continue
    return viewpoint_time_index

def remove_long_samples(data, max_len=20):
    # *
    new_col = []
    for item in data:
        if len(item['path']) > max_len:
            continue
        else:
            new_col.append(item)
    return new_col



def add_period(data):
    # *
    copy_data = copy.deepcopy(data)
    for item in copy_data:
        instruction = item['instruction']
        if instruction[-1] not in string.punctuation:
            # print(instruction)
            instruction += '.'
        elif instruction[-1] == '.':
            continue
        elif instruction[-1] != '.' and instruction[-1] in string.punctuation:
            instruction = instruction[:-1] + '.'        
        item['instruction'] = instruction
    return copy_data

def split_instruction(sample, viewpoint_time_index, step_ahead=0):
    # *
    """
    sample: original data instance
    viewpoint_time_index: generated viewpoint time pair (to match with instruction)
    if offset is true, we take full instruction, and one step ahead of partial path
    
    Return: list of (partial path, partial instruction)
    """
    splited_instruction = []
    partial_instruction = []
    idx = 1+step_ahead # start ind for partial instructions
    for timed in sample['timed_instruction']:
        try:
            if np.isnan(timed['end_time']):
                # print(1)
                partial_instruction.append(timed['word']) # to catch nan
            else:
                if timed['end_time'] <= viewpoint_time_index[idx][1]:
                    partial_instruction.append(timed['word'])
                else:
                    stop_viewpoint = viewpoint_time_index[idx][0]
                    # stop_viewpoint_idx is adjusted as some viewpoint in path don't have a paired time
                    stop_viewpoint_idx = sample['path'].index(stop_viewpoint)
                    to_add = ' '.join(copy.deepcopy(partial_instruction))
                    splited_instruction.append({
                        'partial_instruction': to_add,
                        # step_ahead is set to 1
                        'partial_path':[vp for vp in sample['path'][:stop_viewpoint_idx+1-step_ahead]],
                        'viewpoint_time': viewpoint_time_index,
                    })

                    partial_instruction.append(timed['word'])
                    idx += 1
                    if idx == len(viewpoint_time_index)                                                                                                                                                                                                                                                                                                                                                                  :
                        break          
        except:
            partial_instruction.append(timed['word']) # to catch nothing but word
    return splited_instruction


def generate_splited_training_data(original_sample, remove_last=False, step_ahead=0, discard=1):
    # * use pose_trace to split instruction-trajectory pair
    viewpoint_time_index = get_viewpoint_time_pair(original_sample)
    splited_instruction = split_instruction(original_sample, viewpoint_time_index, step_ahead=step_ahead)
    train_samples = []
    lessthantwo = 0 
    # print(len(splited_instruction), len(original_sample['path']))
    for i, split in enumerate(splited_instruction):
        new = {
            "path_id": original_sample['path_id'],
            "split": original_sample['split'],
            "scan": original_sample['scan'],
            "heading": original_sample['heading'],
            "path": split['partial_path'],
            "instruction_id": str(original_sample['instruction_id'])+'_'+str(len(split['partial_path'])), # check what this relates to
            "language": original_sample['language'],
            "instruction": split['partial_instruction'],
            "edit_distance": original_sample['edit_distance'],
        }
        if remove_last and i == len(splited_instruction)-discard:
            break
        else:
            train_samples.append(new)
    if len(splited_instruction) < 2:
        lessthantwo = 1
        
    return train_samples, lessthantwo

def filter_direction_instances(results, direction=''):
    # *
    """
    filter out examples that don't have both left and right options
    1. goal direction has at least 1 viewpoint
    2. less than 2 are excluded
    3. two other neighboring outside the target direction
    """
    lefted = [False] * len(results)
    outside_counter = [0] * len(results)
    def outside_checker(direction, deg):
        # *
        if direction == 'left' and not (-90 < deg <= -15):
            return 1
        if direction == 'right' and not (15 < deg <= 90):
            return 1
        if direction == 'forward' and not (-15 < deg <= 15):
            return 1
        if direction == 'backward' and not (-180 < deg <= -165 or 165 < deg <= 180): # check this
            return 1
        if direction == 'back left' and not (-165 < deg <= -90):
            return 1
        if direction == 'back right' and not (90 < deg <= 165):
            return 1
        return 0

    for i, item in enumerate(results):
        # add this filter here
        if len(item['candidates'][0][1]) <=2:
            continue
        else:
            for cand in item['candidates'][0][1]:
                deg = rad2deg_mod(cand['heading'])
                is_outside = outside_checker(direction, deg)
                outside_counter[i] += is_outside
                if not is_outside:
                    lefted[i] = True
                    break # ! this is added

    left_and_right_sample = [item for i, item in enumerate(results) if lefted[i] and outside_counter[i] >= 2]
    left_and_right_ids = [i for i, _ in enumerate(results) if lefted[i] and outside_counter[i] >= 2]
    return left_and_right_sample, left_and_right_ids

def get_heading_and_probs_pairs_all_directions(directional_instances):
    # *
    """
    for each item, get a new key,value pair: 'directions: prob': {'0-10': 0.5, '10-20': 0.2 ... }, 
    but notice here, the stop action's prob is IGNORED!
    """
    dir_copy = directional_instances
    # first get the correct degrees and get corresponding probs
    all_pairs = []
    all_stops = []
    for item in dir_copy:
        heading_prob_pairs = []
        for i, (cand, prob) in enumerate(zip(item['candidates'][0][1], item['final_logits'][0])): # only considering the first after supervision for now
            head_to_process = np.rad2deg(cand['heading'])
            # normalized heading
            if 0 <= head_to_process < 180:
                head_to_process = head_to_process
            if -180 <= head_to_process < 0:
                head_to_process = head_to_process
            if 180 <= head_to_process <= 360:
                head_to_process = head_to_process % -360
            if -360 <= head_to_process <= -180:
                head_to_process = head_to_process % 360
            
            heading_prob_pairs.append((head_to_process, prob))
        all_stops.append(item['final_logits'][0][len(item['candidates'][0][1])])
        all_pairs.extend(heading_prob_pairs)
    
    return dir_copy, all_pairs, all_stops

    
def rad2deg_mod(rad):
    # *
    """
    Convert rad2deg, mod the result so it is within (-180, 180), support input from (-pi, pi)
    This is for VLN heading axis.
    """
    head_to_process = np.rad2deg(rad)
    if 0 <= head_to_process < 180:
        head_to_process = head_to_process
    if -180 <= head_to_process < 0:
        head_to_process = head_to_process
    if 180 <= head_to_process <= 360:
        head_to_process = head_to_process % -360
    if -360 <= head_to_process <= -180:
        head_to_process = head_to_process % 360
    return head_to_process

def binning_direction_pairs(all_heading_probs_pairs, binsize=30):
    # *
    ranges = list(np.arange(-180, 181, binsize))
    direction_pairs = defaultdict(list)
    for item in all_heading_probs_pairs:
        for start, end in zip(ranges, ranges[1:]):
            if start<=item[0]< end:
                direction_pairs[start].append(item[1])
                break #added thid break
    return direction_pairs

def prepare_direction_and_draw(no_intervene_results, intervene_results, polar=False, xlabel='step #', ylabel='SR', title='dist', name1='', name2='', ax=None, curve=True, cl=True, **kwargs):
    # * prepare and draw fig 
    no_end_mean, no_end_low, no_end_high, no_end_bins = no_intervene_results['dist'].to_numpy(), no_intervene_results['CI_low_dist'].to_numpy(), no_intervene_results['CI_high_dist'].to_numpy(), no_intervene_results.index.to_numpy(dtype=np.int32)
    with_end_mean, with_end_low, with_end_high, with_end_bins = intervene_results['dist'].to_numpy(), intervene_results['CI_low_dist'].to_numpy(), intervene_results['CI_high_dist'].to_numpy(), intervene_results.index.to_numpy(dtype=np.int32)

    x = (no_end_bins, with_end_bins)
    y = no_end_mean
    z = with_end_mean

    yerr_low = y[np.newaxis, :] - no_end_low[np.newaxis, :]
    yerr_high = no_end_high[np.newaxis, :] - y[np.newaxis, :]

    zerr_low = z[np.newaxis, :] - with_end_low[np.newaxis, :]
    zerr_high = with_end_high[np.newaxis,:] - z[np.newaxis, :]
    
    yerr = np.concatenate((yerr_low, yerr_high), axis=0)
    zerr = np.concatenate((zerr_low, zerr_high), axis=0)
    draw_direction(ax, x, y, z, yerr, zerr, polar=polar, xlabel=xlabel, ylabel=ylabel, title=title, name1=name1, name2=name2, curve=curve, cl=cl, **kwargs)

def create_scan_id_intervene_label_stop_prob(data, intervene_label=None):
    #* collect info for STOP
    sample_triplet = []
    for item in data:
        scan_id = item['candidates'][0][1][0]['scanId']
        sample = {
            "scan_id": scan_id,
            "path_id": item['instr_id'].split('_')[0],
            "step_id": int(item['instr_id'].split('_')[-1]),
            "intervention_yes_no": intervene_label,
            "stop_prob": item['stop_prob'][0],
        }
        sample_triplet.append(sample)
    return pd.DataFrame(sample_triplet)

def draw_dists_with_confidence_tri(no_intervene_results, intervene_results, ahead_partial, xlabel='step #', ylabel='SR', title='dist', name1='', name2='', name3='', ax=None, curve=True, cl=True, savedir=''):
    # * prepare data and draw
    no_end_mean, no_end_low, no_end_high, no_end_bins = no_intervene_results['mean'].to_numpy(), no_intervene_results['CI_low'].to_numpy(), no_intervene_results['CI_high'].to_numpy(), no_intervene_results['step'].to_numpy(dtype=np.int32)
    with_end_mean, with_end_low, with_end_high, with_end_bins = intervene_results['mean'].to_numpy(), intervene_results['CI_low'].to_numpy(), intervene_results['CI_high'].to_numpy(), intervene_results['step'].to_numpy(dtype=np.int32)
    ahead_partial_mean, ahead_partial_low, ahead_partial_high, ahead_partial_bins = ahead_partial['mean'].to_numpy(), ahead_partial['CI_low'].to_numpy(), ahead_partial['CI_high'].to_numpy(), ahead_partial['step'].to_numpy(dtype=np.int32)

    x = (no_end_bins, with_end_bins, ahead_partial_bins)
    y = no_end_mean
    z = with_end_mean
    w = ahead_partial_mean

    yerr_low = y[np.newaxis, :] - no_end_low[np.newaxis, :]
    yerr_high = no_end_high[np.newaxis, :] - y[np.newaxis, :]

    zerr_low = z[np.newaxis, :] - with_end_low[np.newaxis, :]
    zerr_high = with_end_high[np.newaxis,:] - z[np.newaxis, :]
    
    werr_low = w[np.newaxis, :] - ahead_partial_low[np.newaxis, :]
    werr_high = ahead_partial_high[np.newaxis, :] - w[np.newaxis, :]


    yerr = np.concatenate((yerr_low, yerr_high), axis=0)
    zerr = np.concatenate((zerr_low, zerr_high), axis=0)
    werr = np.concatenate((werr_low, werr_high), axis=0)

    get_stop_rate_by_traj_progress_tri(ax, x, y, z, w,  yerr, zerr, werr, xlabel=xlabel, ylabel=ylabel, title=title, name1=name1, name2=name2, name3=name3, curve=curve, cl=cl, savedir=savedir)
    
def get_stop_rate_by_traj_progress_tri(ax, x,y,z,w, yerr, zerr, werr, xlabel='', ylabel='', title='', name1='', name2='', name3='', curve=True, cl=True, savedir=''):
    # * draw stop figure
    # plt.style.use(['ggplot', './image_style/paper2.mplstyle'])
    # plt.style.use(['ggplot', './image_style/paper2.mplstyle'])
    # mpl.rcParams['legend.fontsize'] = 30 
    # mpl.rcParams['ytick.labelsize'] = 30
    # mpl.rcParams['xtick.labelsize'] = 30
    # mpl.rcParams['axes.labelsize'] = 100
    # plt.savefig('tst.png', format='png', bbox_inches='tight')

    plt.tight_layout()
    width = 0.3 
    labels = x[0] if len(x[0]) > len(x[1]) else x[1]
    labels = labels if len(labels) > len(x[2]) else x[2]
    from file_util import my_colors
    colors = my_colors
    if cl:
        rect_no_end = ax.bar(x[0]-3 *width/2, y, width, label=name1, yerr=yerr,error_kw=dict(lw=2, capsize=5, capthick=2), color=colors[0])
    else:
        rect_no_end = ax.bar(x[0]-3* width/2, y, width, label=name1)
    if curve:
        x_0_new = np.linspace(labels[0], labels[-1], num=50)
        bspline = interpolate.make_interp_spline(labels, y)
        y_0_new = bspline(x_0_new)
        ax.plot(x_0_new, y_0_new, 'b-')

    if cl:
        rect_with_end = ax.bar(x[1]-width/2, z, width, label=name2, yerr=zerr, error_kw=dict(lw=2, capsize=5, capthick=2), color=colors[1]) #yerr shape (2, N)
    else:
        rect_with_end = ax.bar(x[1] -width/2, z, width, label=name2)
    if curve:
        x_1_new = np.linspace(labels[0], labels[-1], num=50)
        bspline1 = interpolate.make_interp_spline(labels, z)
        y_1_new = bspline1(x_1_new)
        ax.plot(x_1_new, y_1_new, 'g-')
    
    if cl:
        f_partial = ax.bar(x[2]+width/2, w, width, label=name3, yerr=werr, error_kw=dict(lw=2, capsize=5, capthick=2), color=colors[2]) #yerr shape (2, N)
    else:
        f_partial = ax.bar(x[2]+width/2, w, width, label=name3)
    if curve:
        x_2_new = np.linspace(labels[0], labels[-1], num=50)
        bspline2 = interpolate.make_interp_spline(labels, w)
        y_2_new = bspline2(x_2_new)
        ax.plot(x_2_new, y_2_new, 'g-')
    ax.set_ylabel(ylabel)
    ax.set_xticks(labels, labels)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim(0, 1)  
    ax.legend(loc = 'center', bbox_to_anchor=(0.5, 1.10), ncol=len(x))   
    plt.savefig(savedir, format='pdf', bbox_inches='tight')

def draw_direction(ax, x,y,z, yerr, zerr, xlabel='', ylabel='', title='', name1='', name2='', curve=True, cl=True, polar=False, **kwargs):
    # * draw polar figure for direction

    if not polar:
        # width = 0.3  # * length for stop experiment
        width = 4  #* width for direction experiment
        labels = x[0] if len(x[0]) > len(x[1]) else x[1]
        if cl:
            rect_no_end = ax.bar(x[0]-width/2, y, width, label=name1, yerr=yerr,error_kw=dict(lw=2, capsize=5, capthick=2))
        else:
            rect_no_end = ax.bar(x[0]-width/2, y, width, label=name1)
        if curve:
            x_0_new = np.linspace(labels[0], labels[-1], num=50)
            bspline = interpolate.make_interp_spline(labels, y)
            y_0_new = bspline(x_0_new)
            ax.plot(x_0_new, y_0_new, 'b-')
        if cl:
            rect_with_end = ax.bar(x[1]+width/2, z, width, label=name2, yerr=zerr, error_kw=dict(lw=2, capsize=5, capthick=2)) #yerr shape (2, N)
        else:
            rect_with_end = ax.bar(x[1]+width/2, z, width, label=name2)
        if curve:
            x_1_new = np.linspace(labels[0], labels[-1], num=50)
            bspline1 = interpolate.make_interp_spline(labels, z)
            y_1_new = bspline1(x_1_new)
            ax.plot(x_1_new, y_1_new, 'g-')
        
        ax.set_ylabel(ylabel)
        ax.set_xticks(labels, labels)
        ax.set_xlabel(xlabel)

        
        ax.tick_params(axis='x', rotation=50)
        ymax = max(y+z + zerr.flatten().tolist() + yerr.flatten().tolist())
        ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    elif polar:
        from fig_help.fig_util import angularDensityProb
        values = np.empty((len(x[0]), 2))
        cis = np.empty((2, 2, len(x[0])))
        values[:, 0] = y
        values[:, 1] = z
        cis[0, ...] = yerr
        cis[1, ...] = zerr
        ang_input = (x[0], values, [name1, name2], cis)
        angularDensityProb(ang_input, edge=0.3, **kwargs)
       
    
def append_instruction_to_noend(no_end, instruction='', templates=[]):
    # * 
    import random
    appended = []
    for item in no_end:
        new_item = copy.deepcopy(item)
        template= random.choice(templates)
        template += '.'
        # import ipdb; ipdb.set_trace()
        new_item['instruction'] += ' ' + template
        appended.append(new_item)
    return appended                   


def boot_draw_total(bootstrap_sample_result, instr_id2no_intervene, instr_id2intervene, name1, name2, direction='', polar=False, **kwargs):
    # * process bootstrap result and draw fig for direction

    ax1 = None
    if not polar:
        plt.figure(figsize=(15,10))
        ax1  = plt.subplot(111, polar=polar)
    binsize = kwargs.get('binsize', 10)
    model_name = kwargs.get('model_name', 'hamt')
    # save_path = kwargs['save_path']
    save_path = kwargs.get('save_path', '')
    intervene_dists, no_intervene_dists = [], []
    intervene_stops, no_intervene_stops = [], []

    # Process each bootstrap sample
    for sample in tqdm(bootstrap_sample_result):
        intervene_dist, inter_stop = process_direction_result(sample['instr_id'], instr_id2intervene, bin_size=binsize, direction=direction)
        no_intervene_dist, no_inter_stop = process_direction_result(sample['instr_id'], instr_id2no_intervene, bin_size=binsize, direction=direction)
        intervene_dists.append(intervene_dist)
        no_intervene_dists.append(no_intervene_dist)
        intervene_stops.append(inter_stop)
        no_intervene_stops.append(no_inter_stop)
    
# Compute the CI
    CI_low_intervene = pd.concat(intervene_dists, axis=0).sort_values('heading').groupby('heading').quantile(0.025)
    CI_high_intervene = pd.concat(intervene_dists, axis=0).sort_values('heading').groupby('heading').quantile(0.975)
    CI_low_intervene_stop = np.quantile(intervene_stops, 0.025)
    CI_high_intervene_stop = np.quantile(intervene_stops, 0.975)

    CI_low_no_intervene = pd.concat(no_intervene_dists, axis=0).sort_values('heading').groupby('heading').quantile(0.025)
    CI_high_no_intervene = pd.concat(no_intervene_dists, axis=0).sort_values('heading').groupby('heading').quantile(0.975)
    CI_low_no_intervene_stop = np.quantile(no_intervene_stops, 0.025)
    CI_high_no_intervene_stop = np.quantile(no_intervene_stops, 0.975)
    # Compute dist  
    boot_overall = pd.concat(bootstrap_sample_result, axis=0)

    # intervene_overall_dist, intervene_stop_prob = process_direction_result(boot_overall['instr_id'], instr_id2intervene, bin_size=10, direction=direction)
    # no_inter_overall_dist, no_intervene_stop_prob = process_direction_result(boot_overall['instr_id'], instr_id2no_intervene, bin_size=10, direction=direction)
    # intervene_overall_dist = intervene_overall_dist.sort_values('heading').set_index('heading')
    # no_inter_overall_dist = no_inter_overall_dist.sort_values('heading').set_index('heading')

    mean_intervene = pd.concat(intervene_dists, axis=0).sort_values('heading').groupby('heading').mean()
    mean_no_intervene = pd.concat(no_intervene_dists, axis=0).sort_values('heading').groupby('heading').mean()
    # compute mean of stop prob by averaging stop prob of each bootstrap sample
    intervene_stop_overall = np.mean(intervene_stops)
    no_intervene_stop_overall = np.mean(no_intervene_stops)

    # combine all the data into a dataframe
    intervene_collection_dframe = pd.concat([mean_intervene, CI_low_intervene, CI_high_intervene], axis=1)
    intervene_collection_dframe.columns = ['dist', 'CI_low_dist', 'CI_high_dist']
    no_inter_collection_dframe = pd.concat([mean_no_intervene, CI_low_no_intervene, CI_high_no_intervene], axis=1)
    no_inter_collection_dframe.columns = ['dist', 'CI_low_dist', 'CI_high_dist']

    to_record = {
        'direction': direction,
        'model_name': model_name,
        'sample_size': len(boot_overall),
        'sample_repeat_num': len(bootstrap_sample_result),
        'intervene': {
            'stop' : intervene_stop_overall,
            # 'dist': intervene_overall_dist.to_dict('index'),
            # 'CI_low_dist' : CI_low_intervene,
            # 'CI_high_dist': CI_high_intervene,
            'dist_related': intervene_collection_dframe.to_dict('index'),
            'CI_low_stop': CI_low_intervene_stop,
            'CI_high_stop': CI_high_intervene_stop
        },
        'no_intervene': {
            'stop' : no_intervene_stop_overall,
            # 'dist': no_inter_overall_dist,
            # 'CI_low_dist' : CI_low_no_intervene,
            # 'CI_high_dist': CI_high_no_intervene,
            'dist_related': no_inter_collection_dframe.to_dict('index'),
            'CI_low_stop': CI_low_no_intervene_stop,
            'CI_high_stop': CI_high_no_intervene_stop
        }
    }
    # save the computed result.
    save_file = os.path.join(save_path, f'{model_name}_{direction}.json')
    save_json(save_file, to_record)


    prepare_direction_and_draw(no_inter_collection_dframe, intervene_collection_dframe ,name1=name1, name2=name2, xlabel='Next Heading', ylabel='Density', title='prob vs deg_change', ax=ax1, cl=False, 
                               curve=False, polar=polar, **kwargs) 

# from direction.data_process import hierarchy_resample

def bootstrap_sample(samples, cluster, replace, func=None, size=100):
    # *
    """
    Do hierarchical bootstrap sampling on the samples. 
    Return bootstrap samples
    """
    boot_collection = []
    for i in trange(size):
        boot_sample = hierarchy_resample(samples, cluster, replace)
        boot_collection.append(boot_sample) # collect boot samples for later use
    return boot_collection

# from direction.data_process import get_heading_and_probs_pairs_all_directions, filter_direction_instances, binning_direction_pairs
# from collections import defaultdict

def process_direction_result(sampled_ids, instr_id2data, bin_size=10, direction=''):
    # *
    """
    Compute the probability distribution of the sampled results on directions. 
    The distribution is discrete with respect to a bin size. 

    sampled_ids: list of sampled result ids
    instr_id2data: dict of instr_id to data
    """
    selected_sampled = [instr_id2data[ins_id] for ins_id in sampled_ids]
    
    # ? Should we filter it here?
    filtered_sample, _  = filter_direction_instances(selected_sampled,direction=direction)


    _, all_pairs, stop_probs = get_heading_and_probs_pairs_all_directions(filtered_sample) # for speed reason, might change this
    binned_prob_dict = binning_direction_pairs(all_pairs, bin_size)

    # Compute the distribution:
    total_prob_exclude_stop = np.sum([np.sum(binned_prob_dict[k]) for k in binned_prob_dict])
    total_prob_include_stop = total_prob_exclude_stop + np.sum(stop_probs)

    # store this to do significance test later
    stop_prob = np.sum(stop_probs) / total_prob_include_stop 

    prob_bined = defaultdict(list)
    for k in binned_prob_dict:
        prob_bined[k] = np.sum(binned_prob_dict[k]) / total_prob_exclude_stop

    prob_bined_dataframe = pd.DataFrame(prob_bined.items(), columns=['heading', 'prob'])
        
    return prob_bined_dataframe, stop_prob

def get_sample_info(data):
    # *collect info to bootstrap
    scan_ids, path_ids, step_ids, instr_ids = [], [], [], []

    for item in data:
        path_id = item['instr_id'].split('_')[0]
        step_id = item['instr_id'].split('_')[2]

        scan_ids.append(item['candidates'][0][1][0]['scanId'])
        path_ids.append(path_id)
        step_ids.append(step_id)
        instr_ids.append(item['instr_id'])
    return pd.DataFrame({'scan_id': scan_ids, 'path_id': path_ids, 'step_id': step_ids, 'instr_id': instr_ids})
import random
import os
import pathlib as path
import argparse

from tqdm import tqdm
from ..utils.direction_utils import (
    generate_splited_training_data,
    remove_long_samples,
    add_period,
)
from ..utils.file_utils import read_gz_jsonlines, save_jsonl


# pose_trace_dir = path.Path("/Users/zijiao/home/research/data/RxR/pose_traces")
# rxr_data_path = path.Path("/Users/zijiao/home/research/RxR/visualizations/rxr_data")


def add_stop(dataset):
    # * add a random stop instruction at the end of each instruction
    for item in dataset:
        # if item['path_id'] not in bad_samples:
        instruction = item["instruction"]
        try:
            if instruction[-1] == ".":
                # print(instruction)
                pass
            else:
                instruction = instruction + "."
        except:
            print(item)
        instruction = (
            instruction + " " + random.choice(end_sentences)
        )  # end_sentences[0]
        item["instruction"] = instruction
    return dataset


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--baseline_dir", type=str, default="baselines")
    args.add_argument("--stop_dir", type=str, default="stop")
    args.add_argument("--pose_trace_dir", type=str, default="")
    args.add_argument("--rxr_data_path", type=str, default="")
    args.add_argument()

    args = args.parse_args
    split = "rxr_val_unseen"
    bad_samples = [4721, 9015]  # bad path ids
    
    
    pose_trace_dir = path.Path(args.pose_trace_dir)
    rxr_data_path = path.Path(args.rxr_data_path)
    test = read_gz_jsonlines(rxr_data_path / "rxr_val_unseen_guide.jsonl.gz")
    test_filtered = [item for item in test if item["language"] == "en-US"]

    no_end_dataset = []
    total = 0
    for item in tqdm(test_filtered):
        if item["path_id"] not in bad_samples:
            generated, ss = generate_splited_training_data(
                item, remove_last=True, discard=1
            )
            total += ss
            no_end_dataset.extend(generated)

    dataset_1ahead = []
    total = 0
    for item in tqdm(test_filtered):
        if item["path_id"] not in bad_samples:
            generated, ss = generate_splited_training_data(
                item, remove_last=True, step_ahead=1, discard=1
            )
            total += ss
            dataset_1ahead.extend(generated)

    ids = [item["instruction_id"] for item in dataset_1ahead]
    commmon_ids = []
    for item in no_end_dataset:
        if item["instruction_id"] in ids:
            commmon_ids.append(item["instruction_id"])
    commmon_ids = set(commmon_ids)

    new_no_end = [
        item for item in no_end_dataset if item["instruction_id"] in commmon_ids
    ]
    new_no_end = remove_long_samples(new_no_end)
    new_no_end = add_period(new_no_end)

    new_ahead_partial = [
        item for item in dataset_1ahead if item["instruction_id"] in commmon_ids
    ]
    new_ahead_partial = remove_long_samples(new_ahead_partial)
    new_ahead_partial = add_period(new_ahead_partial)

    # save_data_dir = './baselines'
    # save_data_dir = './baselines/more_templates'
    save_data_dir = args.baseline_dir

    # stop_dir = './stop/more_templates'
    stop_dir = args.stop_dir

    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    if not os.path.exists(stop_dir):
        os.makedirs(stop_dir)

    save_data_dir = save_data_dir + "/{}"
    stop_dir = stop_dir + "/{}"

    save_jsonl(save_data_dir.format("rxr_no_end_paired_with_ahead.jsonl"), new_no_end)
    save_jsonl(save_data_dir.format("rxr_ahead_partial.jsonl"), new_ahead_partial)

    # use this to reproduce the result in paper.
    # end_sentences = [
    #     'This is your destination.',
    #     'This is your end point.',
    #     'You reached your destination.',
    #     'You are done.',
    # ]

    end_sentences = [
        "This is your destination.",
        "This is your end point.",
        "You reached your destination.",
        "You are done.",
        "Stop.",
        "You have arrived.",
        "You've reached the end.",
        "This is where you stop.",
    ]
    # add stop instruction
    with_end = add_stop(new_no_end)
    save_jsonl(stop_dir.format("rxr_with_end_paired_with_ahead.jsonl"), with_end)
    print(
        "No end: ",
        len(new_no_end),
        "1 ahead: ",
        len(new_ahead_partial),
        "with end: ",
        len(with_end),
    )
    print()

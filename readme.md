<!--ts-->
<!--te-->

# Project Description

**This is the accompanying code base for the paper "Behavioral Analysis of Vision-and-Language Navigation
Agents".**

**Authors:** [Zijiao Yang](https://yoark.github.io/), [Arjun Majumdar](https://arjunmajum.github.io/), [Stefan Lee](https://web.engr.oregonstate.edu/~leestef/)

**Abstract:**

_To be successful, Vision-and-Language Navigation (VLN) agents must be able to ground instructions to actions based on their surroundings. In this work, we develop a methodology to study agent behavior on a skill-specific basis -- examining how well existing agents ground instructions about stopping, turning, and moving towards specified objects or rooms. Our approach is based on generating skill-specific interventions and measuring changes in agent predictions. We present a detailed case study analyzing the behavior of a recent agent and then compare multiple agents in terms of skill-specific competency scores. This analysis suggests that biases from training have lasting effects on agent behavior and that existing models are able to ground simple referring expressions. Our comparisons between models show that skill-specific scores correlate with improvements in overall VLN task performance._

**Bibtex**

```
@inproceedings{yang2023vlnbehave,
    title={Behavioral Analysis of Vision-and-Language Navigation Agents},
    author={Zijiao Yang and Arjun Majumdar and Stefan Lee},
    booktitle={Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}
```

The code mainly contain two parts:

1. [Integrate logging code into VLN agents.](#data-log) (mannully for now)
2. [Conduct data processing and analyze agents performance on STOP, TURN, DIRECTION, ROOM skills.](#data-analysis)

<details>
    <summary> Directory Structure </summary>
<pre>
.
├── LICENSE
├── data
│   ├── __init__.py
│   ├── create_direction_data.py
│   ├── create_object_data.py
│   ├── create_region_data.py
│   ├── create_room_data.py
│   ├── create_stop_data.py
│   └── vlnbehave.mplstyle
├── experiments
│   ├── analyze_direction.py
│   ├── analyze_object.py
│   ├── analyze_room_delta.py
│   ├── analyze_room_dist2goal.py
│   ├── analyze_stop.py
│   └── rxr_traj_stats.py
├── lmer
│   ├── lmer_analysis
│   │   ├── direction.r
│   │   ├── object.r
│   │   ├── room.r
│   │   └── stop.r
│   ├── lmer_direction.py
│   ├── lmer_object.py
│   ├── lmer_room.py
│   ├── lmer_stop.py
│   └── n_hop_dist2goal.bash
├── readme.md
├── scores
│   ├── direction_score.py
│   ├── object_score.py
│   ├── room_score.py
│   └── stop_score.py
└── utils
    ├── __init__.py
    ├── direction_utils.py
    ├── file_utils.py
    ├── object_utils.py
    └── room_utils.py
</pre>
</details>

# Datasets

| Dataset Name   | Description                                                                                                                           | Link                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| RxR pose trace | The data describing alignments between agent's pose, location, and instruction; **note we use follower pose trace **                  | [RxR](https://github.com/google-research-datasets/RxR)                    |
| RxR dataset    | We use `train, and val_unseen split`                                                                                                  | same as above                                                             |
| REVERIE        | It provides object annotation for matterport dataset, we use the object annotation that is within 3m of the observer. i.e. REVERIE_3m | [REVERIE](https://github.com/YuankaiQi/REVERIE/tree/master/tasks/REVERIE) |
| Matterport     | The dataset capturing houses from various aspect. _We use their region annotations mostly_                                            | [Matterport3D](https://niessner.github.io/Matterport/)                    |

_Please cite the corresponding dataset papers if you use the data._

<a id="data-log"></a>

# Integrate Data Log into Your Agent

See [logging/readme.md](logging/readme.md) for details.

<a id="data-analysis"></a>

# Data Creation and Experiment Scripts

Make sure all the data is downloaded/created using `data/` before running the experiments.

## Stop

**To create truncation data, and data for stop analysis**

```
python create_stop_data.py \
    --stop_dir stop_data_save_dir \
    --baseline_dir truncation_data_save_dir \
    --pose_trace_dir pose_trace_dir \
    --rxr_data_path rxr_data_path \
```

**To analyze agent's sensitivity to stop instructions**

```
python analyze_stop.py \
    --data_dir model_result_dir \
    --model_name model_name \
    --save_dir save_dir \
    --bootstrap_num 500
```

## Direction

**To create data for direction analysis**

```
python create_direction_data.py \
    --baseline_dir truncation_data_save_dir \
    --direction_dir direction_data_save_dir \
```

**To analyze agent's sensitivity to direction instructions**

```
python analyze_direction.py \
    --model_name model_name \
    --result_path model_result_dir \
    --baseline_dir model_result_on_truncation_data \
    --save_dir save_dir \
    --bootstrap_num 100
```

## Object

**To create data for object analysis**

```
python create_object_data.py \
    --object_path dir_contain_obj_annotations \
    --baseline_dir truncation_data_save_dir \
    --save_dir save_dir
```

**To analyze agent's sensitivity to object seeking instructions**

```
python analyze_object.py \
    --results_path model_result_dir \
    --save_path save_dir \
    --model_name model_name \
    --input_path  --bootstrap_num 100
```

## Room

**Create region data**

```
python create_region_data.py \
    --region_dir region_dir \
    --save_dir save_dir
```

**To create data for room analysis**

```
python create_room_data.py \
    --save_dir save_dir \
    --baseline_dir truncation_data_save_dir \
    --region_dir region_dir \
    --result_dir truncation_result_dir
```

**To analyze agent's sensitivity to room seeking instructions (1 hop only)**

```
python analyze_room.py \
    --hop 1 \
    --model_name model_name \
    --result_dir model_result_dir \
    --save_dir save_dir
```

**To analyze distance to goal room type (for 1 -- 8 hops)**

```
python draw_dist_to_goal_room.py \
    --model_name model_name \
    --data_dir room_data_dir \
    --save_dir save_dir \
    --result_dir model_result_dir
```

# Linear Mixed Effect Model

## data preparation

`./lmer`

## model fitting and anova

`./lmer/lmer_analysis`

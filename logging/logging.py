# Note the code in this file is not runnable, it is just for demonstration purpose.
# This file demonstrates the logging added in your agent class/file.
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

# replace this with true args
argparser = argparse.ArgumentParser()
args = argparser.parse_args()

# We need to add a few entry points to the recorded info
obs = []  # the observation, replace this
traj = [
    {
        "instr_id": ob["instr_id"],
        "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
        "stop_prob": [],
        "final_logits": [],
        "nav_type": [],
        "heading": "",
        "candidates": [],
        "prev_stop_probs": [],
        "viewIndex": "",
        # 'current_pos': None, # the position right after supervision
        "next_pos": None,  # argmax
        "final_viewpoint_id": None,
        # 'final_pos': None,
        # 'goal_region_loc':None
    }
    for ob in obs
]


# move this to agent class
def teacher_action_inference(obs, ended):
    """
    Extract teacher actions into variable, the label is used to control when to end guided navigation
    :param obs: The observation.
    :param ended: Whether the action seq is ended
    :return:
    """
    a = np.zeros(len(obs), dtype=np.int64)
    is_teachering = [None] * len(obs)
    for i, ob in enumerate(obs):
        if ended[i]:  # Just ignore this index
            a[i] = args.ignoreid
            is_teachering[i] = "stopped"
        else:
            for k, candidate in enumerate(ob["candidate"]):
                if candidate["viewpointId"] == ob["teacher"]:  # Next view point
                    a[i] = k
                    is_teachering[i] = "yes"
                    break
            else:  # Stop here
                assert (
                    ob["teacher"] == ob["viewpoint"]
                )  # The teacher action should be "STAY HERE"
                a[i] = len(ob["candidate"])
                is_teachering[i] = "no"
    return torch.from_numpy(a).cuda(), is_teachering


def get_action(self, obs, logit, target=None, ended=None):
    # anything type2 is related to force the agent to a location, and continue from there
    if not self.args.ablate_names or self.args.test_type == "type1":
        # These are the usual model logit options

        if self.feedback == "teacher":
            ...
        elif self.feedback == "argmax":
            ...
        elif self.feedback == "sample":
            ...
        else:
            print(self.feedback)
            sys.exit("Invalid feedback option")
    elif self.args.ablate_names and self.args.test_type == "type2":
        a_t = self._get_action_type2(obs, logit, target, ended)
    return a_t


def _get_action_type2(self, obs, logit, target, ended):
    a_t = torch.zeros(len(obs), dtype=torch.int64)
    _, a_t_argmax = logit.max(1)
    a_t_argmax = a_t_argmax.cpu()
    # get labels for current navigation status: still teacher forcing? or freely navigate
    target, is_teachering = self._teacher_action_inference(obs, ended)
    for i, (teachering, a) in enumerate(zip(is_teachering, target)):
        if teachering == "yes":
            a_t[i] = a
            # log stop probs during teacher forcing
            if self.args.store_prev_stops:
                prev_stop_prob = torch.softmax(logit[i].detach(), dim=0)[
                    ob_cand_lens[i] - 1
                ].item()
                traj[i]["prev_stop_probs"].append((t, prev_stop_prob))
        elif teachering == "no":
            a_t[i] = a_t_argmax[i]
            # this should be the first prediction after supervision
            if not stopped[i]:
                stop_prob = torch.softmax(logit[i].detach(), dim=0)[
                    ob_cand_lens[i] - 1
                ].item()
                stopped[i] = 1
                # log stop prob after right after teacher forcing
                traj[i]["stop_prob"].append(stop_prob)
            if self.args.store_final_logits:  # could be more than one logged
                final_logits = torch.softmax(logit[i].detach(), dim=0).tolist()
                # recording logits for every step after teacher forcing as well as candidates at each step.
                traj[i]["final_logits"].append(final_logits)
                log_cands = []
                for cand in obs[i]["candidate"]:
                    new_cand = {}
                    for k, v in cand.items():
                        if k != "feature":
                            new_cand[k] = v
                    log_cands.append(new_cand)
                traj[i]["candidates"].append((obs[i]["viewpoint"], log_cands))
                traj[i]["nav_type"].append(
                    ob_nav_types[i].tolist()
                )  # why 38? since the pano feature and cand feature are appended, don't matter for me

                # only record the heading right after supervision
                if not traj[i]["heading"]:
                    traj[i]["heading"] = obs[i]["heading"]
                # only record the final viewpoint id right after supervision
                if not traj[i]["viewIndex"]:
                    traj[i]["viewIndex"] = obs[i]["viewIndex"]
                # only record the final viewpoint id right after supervision
                if not traj[i]["next_pos"]:
                    if (
                        a_t[i] == -1
                        or a_t[i] == (ob_cand_lens[i] - 1)
                        or ended[i]
                        or a_t[i] == self.args.ignoreid
                    ):
                        next_viewpoint = obs[i]["viewpoint"]
                    else:
                        next_viewpoint = obs[i]["candidate"][a_t[i]]["viewpointId"]
                    traj[i]["next_pos"] = next_viewpoint

        elif teachering == "stopped":
            a_t[i] = self.args.ignoreid
    return a_t


# log the final viewpoint id after forceing the agent to end of sub-path. This is mainly for validation purpose
assert all([len(item["stop_prob"]) == 1 for item in traj])
for i, item in enumerate(traj):
    item["final_viewpoint_id"] = item["path"][-1]

import argparse
import os

from utils.direction_utils import append_instruction_to_noend
from utils.file_utils import load_jsonl, save_jsonl

# find more templates in commmented code below

if __name__ == "__main__":
    # use more templates
    args = argparse.ArgumentParser()
    args.add_argument("--baseline_dir", type=str, default="baselines")
    args.add_argument("--direction_dir", type=str, default="direction")
    args = args.parse_args()
    # use more templates
    data_dir = args.baseline_dir + "/{}"

    no_end = load_jsonl(data_dir.format("rxr_no_end_paired_with_ahead.jsonl"))

    save_dir = args.direction_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + "/{}"
    # 6 directions
    forward_templates = [
        "Move forward",
        "Go forward",
        "Move ahead",
        "Walk ahead",
        "Go ahead",
    ]

    # backward_no_turn_around_templates= ["Move back", "Walk back", "Go back"]
    backward_turn_around_templates = [
        "Turn around and " + fo.lower() for fo in forward_templates
    ]
    # change the string to uppercase

    left_templates = [
        "Turn to your left",
        "Take a left",
        "Go to your left",
        "Go left",
        "Turn to the left",
        "Make a left",
        "Take left",
    ]
    left_and_forward_templates = [
        le + " and " + fo.lower() for le in left_templates for fo in forward_templates
    ]

    right_templates = [
        "Turn to your right",
        "Take a right",
        "Go to your right",
        "Go right",
        "Turn to the right",
        "Make a right",
        "Take right",
    ]
    right_and_forward_templates = [
        ri + " and " + fo.lower() for ri in right_templates for fo in forward_templates
    ]

    # back_left_templates= ["Turn around to your left"]
    # back_right_templates= ["Turn around to your right"]
    # back_left_forward_templates = [le + " and " + fo.lower() for le in back_left_templates for fo in forward_templates]
    # back_right_forward_templates = [ri + " and " + fo.lower() for ri in back_right_templates for fo in forward_templates]

    back_left_combined_templates = [
        "Turn around and " + ri.lower() for ri in right_templates
    ]
    back_right_combined_templates = [
        "Turn around and " + le.lower() for le in left_templates
    ]
    back_left_forward_combined_templates = [
        le + ", " + fo.lower()
        for le in back_left_combined_templates
        for fo in forward_templates
    ]
    back_right_forward_combined_templates = [
        ri + ", " + fo.lower()
        for ri in back_right_combined_templates
        for fo in forward_templates
    ]

    templates = {
        "forward": forward_templates,
        "backward_with_turn_around": backward_turn_around_templates,
        # 'backward_no_turn_around': backward_no_turn_around_templates,
        "left": left_and_forward_templates,
        "right": right_and_forward_templates,
        # 'back_left_no_combine': back_left_forward_templates,
        "back_left_combine": back_left_forward_combined_templates,
        # 'back_right_no_combine': back_right_forward_templates,
        "back_right_combine": back_right_forward_combined_templates,
    }

    for direction in templates:
        intervention_data = append_instruction_to_noend(
            no_end, templates=templates[direction]
        )
        save_jsonl(
            save_dir.format("rxr_new_{}_intervention.jsonl".format(direction)),
            intervention_data,
        )
        print("Saved {} intervention data!".format(direction))

    # walk_forward = append_instruction_to_noend(new_no_end, 'Walk forward.')

    # save_jsonl('direction/data/rxr_new_walk_forward.jsonl', walk_forward)
    # backward
    # walk_backward = append_instruction_to_noend(new_no_end, 'Turn around and walk forward.')

    # save_jsonl('direction/data/rxr_new_walk_backward.jsonl', walk_backward)
    # # walk back left
    # turn_back_left = append_instruction_to_noend(new_no_end, 'Turn around and go to your right.')
    # save_jsonl('direction/data/rxr_new_turn_back_left.jsonl', turn_back_left)

    # # walk back right
    # turn_back_right = append_instruction_to_noend(new_no_end, 'Turn around and go to your left.')
    # save_jsonl('direction/data/rxr_new_turn_back_right.jsonl', turn_back_right)

    # # walk left
    # turn_left_ahead = append_instruction_to_noend(new_no_end, 'Turn left and walk forward.')
    # save_jsonl('direction/data/rxr_new_turn_left.jsonl', turn_left_ahead)

    # # walk right
    # turn_right_ahead = append_instruction_to_noend(new_no_end, 'Turn right and walk forward.')
    # save_jsonl('direction/data/rxr_new_turn_right.jsonl', turn_right_ahead)

    # walk_backward = append_instruction_to_noend(new_no_end, 'Turn around and move forward.')
    # ---------- turn around experiment ------------
    # walk_backward = append_instruction_to_noend(new_no_end, 'Turn around.')

    # save_jsonl('direction/data/rxr_new_walk_backward_dec_9_turn_around.jsonl', walk_backward)
    # ---------- turn around experiment ------------
    # save_jsonl('direction/data/rxr_new_walk_backward_dec_9.jsonl', walk_backward)
    # # walk back left

    # turn_back_left = append_instruction_to_noend(new_no_end, 'Turn around to your left and walk forward.')
    # save_jsonl('direction/data/rxr_new_turn_back_left_dec_9.jsonl', turn_back_left)

    # # walk back right
    # turn_back_right = append_instruction_to_noend(new_no_end, 'Turn around to your right and walk forward.')
    # save_jsonl('direction/data/rxr_new_turn_back_right_dec_9.jsonl', turn_back_right)


# file_names = ['new_back_left_combine_intervention new_back_left_no_combine_intervention new_back_right_combine_intervention new_back_right_no_combine_intervention new_backward_no_turn_around_intervention new_backward_with_turn_around_intervention new_forward_intervention new_left_intervention new_right_intervention']

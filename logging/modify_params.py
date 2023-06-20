# These are some modifications to your params setting.
import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("--root_dir", type=str, default="/usr/src/data/unitVLN/datasets")
parser.add_argument(
    "--dataset",
    type=str,
    default="r2r",
    choices=["r2r", "r4r", "r2r_back", "r2r_last", "rxr"],
)
# * give dataset names here
parser.add_argument(
    "--ablate_names", nargs="*", default=None, help="enter a list of ablation datasets"
)
parser.add_argument(
    "--test_type",
    type=str,
    choices=["type1", "type2"],
    default="",
    help="type1 is full instruction no teacher forcing, type2 is parital instructions and with teacher forcing to certain viewpoint,"
    'this should be used together with "--ablate_names"',
)
parser.add_argument("--store_final_logits", default=False, action="store_true")
parser.add_argument("--store_prev_stops", default=False, action="store_true")

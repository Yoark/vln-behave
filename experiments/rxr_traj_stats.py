import numpy as np
import pathlib as path
import matplotlib.pyplot as plt

from ..utils.object_utils import diff_with_direction
from ..utils.file_utils import load_json, read_gz_jsonlines, my_colors

result_train = load_json("baselines/submit_train_filtered_0.json")
angle_diffs = []
forward_bin = 0
total = 0
for item in result_train:
    for i in range(len(item["trajectory"]) - 1):
        total += 1
        diff = np.abs(
            diff_with_direction(item["trajectory"][i][1], item["trajectory"][i + 1][1])
        )
        if abs(diff) < 15:
            forward_bin += 1
        angle_diffs.append(diff)


rxr_data_path = path.Path("/Users/zijiao/home/research/data/RxR/")
train = read_gz_jsonlines(rxr_data_path / "rxr_train_guide.jsonl.gz")
train_filtered = [item for item in train if item["language"] == "en-US"]
train_instructions = [item["instruction"] for item in train_filtered]
for ins in train_instructions:
    if "walk towards the bathroom" in ins:
        print(ins)
        break
train_lengths = [len(item["path"]) for item in train_filtered]
colors = my_colors
x_deltas = np.arange(
    0, 181, 15
).tolist()  # ? this delta distance x axis is too specific is there a better way?
bin_number, bins = np.histogram(angle_diffs, bins=x_deltas, range=(0, 180))
bins = np.round(bins, 1)

names = list(zip(bins, bins[1:]))
datas = bin_number / len(angle_diffs)

x = np.arange(len(names))
barWidth = 0.3
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(20, 10))
ax.bar(x, datas, width=barWidth, color=colors[1], label="", capsize=7)
len_b = len(x_deltas)
ax.set_ylabel("Density")
ax.set_xlabel("Absolute Angular Change Distribution")

ax.tick_params(axis="x", rotation=50)
ax.grid(True, which="both")

names = [f"[{a}, {b})" for a, b in names]
ax.set_xticks(ticks=[r for r in range(len(names))], labels=[str(i) for i in names])
width = 0.5
ax2.hist(train_lengths, np.arange(2, 25) - width / 2, width=width, color=colors[0])

ax2.set_xticks(ticks=np.arange(2, 25))
ax2.set_xlabel("Path length")
ax2.set_ylabel("Count")
plt.tight_layout()
plt.savefig(
    "./baselines/abs_angular_change_train.pdf", bbox_inches="tight", format="pdf"
)

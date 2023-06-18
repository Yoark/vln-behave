import argparse
import pathlib

from utils.file_utils import save_json


def read_region_from_file(filename):
    with open(filename) as f:
        test = []
        viewpoint_region_index = dict()
        region_index2region_label = dict()
        for row in f:
            # print(row[0])
            if row[0] == "R":
                row = row.strip().split()
                # print(row[0], row[5])
                region = {
                    "room_type": row[5],
                    "loc_xyz": list(map(float, [row[6], row[7], row[8]])),
                    "level_index": int(row[2]),
                    "bbox": list(
                        map(
                            float, [row[9], row[10], row[11], row[12], row[13], row[14]]
                        )
                    ),
                    "height": float(row[15]),
                }
                region_index2region_label[row[1]] = row[5]
                test.append(region)
            if row[0] == "P":
                row = row.strip().split()  # type: ignore
                viewpoint_region_index[row[1]] = row[3]
                # if row[3] == '-1':
                # viewpoint_region_index[row[1]] = row[3]
                # print(filename, row)
    return test, viewpoint_region_index, region_index2region_label


if __name__ == "__main__":
    scan_viewpoint_label = dict()
    args = argparse.ArgumentParser()
    args.add_argument(
        "--region_dir", type=str, default=None, help="the region data dir"
    )
    args.add_argument("--save_dir", type=str, default=None, help="the save path")
    args = args.parse_args()

    house_dir = pathlib.Path(args.region_dir)
    # house_dir = pathlib.Path('/Users/zijiao/home/research/data/matterport3d/houses/')
    for gob in house_dir.glob("*.house"):
        _, v2i, i2r = read_region_from_file(gob)
        v2r = dict()
        for v, i in v2i.items():
            if i == "-1":
                v2r[v] = None
            else:
                v2r[v] = i2r[i]
        # v2r = {v: i2r[i] for v, i in v2i.items()}
        scan_viewpoint_label[gob.stem] = v2r
    save_json(
        f"{args.save_dir}/accurate_scan_viewpoint_label.json", scan_viewpoint_label
    )

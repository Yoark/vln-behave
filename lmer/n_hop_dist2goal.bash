#!/bin/bash
# catch params from cmd
hop=1
mode="dist"
result_path="results/envdrop-imagenet"
model_name="envdrop-imagenet"

while [ $# -gt 0 ]; do
    case "$1" in
        --hop=*)
            hop="${1#*=}"
            ;;
        --mode=*)
            mode="${1#*=}"
            ;;
        --result_path=*)
            result_path="${1#*=}"
            ;;
        --model_name=*)
            model_name="${1#*=}"
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
    esac
    shift
done

for i in {2..8}
do
    # python lmer/lmer_room.py --hop $i --mode dist --result_path results/envdrop-imagenet --model_name envdrop-imagenet
    python lmer/lmer_room.py --hop=$i --mode=$mode --result_path=$result_path --model_name=$model_name
done
#!/bin/bash
set -e
model_name="gpt2"
data_type="zip2_3"
dataset_test="./data/${data_type}_gt.csv"
dataset_test_full="./data/${data_type}_gt.csv"
default_temperature=0.5
temperature="${2:-$default_temperature}"
name_t="${temperature//./}"
# Set the default CUDA device(s)
DEFAULT_CUDA_VISIBLE_DEVICES=0
# Check if a specific CUDA device is given as a command-line argument
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
# Export the CUDA_VISIBLE_DEVICES variable
export CUDA_VISIBLE_DEVICES
echo "temperature=${temperature}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# tags for Pre-trained dataset
data_deltas=("02_4096" "01_4096")

epoch=10000
repeat=4096


for data_delta in ${data_deltas[@]};do
  dataset_train="./data/${data_type}_${data_delta}.csv"
  current_time=$(date +"%Y%m%d_%H%M%S")
  output_basedir="./results/rlstar_${data_type}_t_${name_t}_${data_delta}_${current_time}"
  output_dir="${output_basedir}/iter_result_0_${data_delta}"
  mkdir -p ${output_basedir}

  for i in $(seq 1 10); do
    python3 RLstar_trainer.py  \
          --model_name=${model_name} \
          --dataset_train=${dataset_train} \
          --dataset_eval=${dataset_test_full} \
          --logging_steps=1000 \
          --output_dir=${output_dir} \
          --epoch=${epoch} | tee "${output_basedir}/train_${i}.log"

    python3 RLstar_evaluator.py \
          --model_name=${model_name} \
          --ckpt_path=${output_dir} \
          --dataset_path_full=${dataset_test_full}  \
          --dataset_path=${dataset_train}  | tee "${output_basedir}/eval_train_${i}.log"

    dataset_train_2="${output_basedir}/iter_data_4_${i}.csv"
    python3 RLstar_evaluator.py \
          --model_name=${model_name} \
          --ckpt_path=${output_dir} \
          --dataset_path=${dataset_test}  \
          --dataset_path_full=${dataset_test_full}  \
          --temperature=${temperature} \
          --repeat=${repeat} \
          --output_path=${dataset_train_2} | tee "${output_basedir}/eval_test_${i}.log"


    dataset_train="${dataset_train_2}"
    output_dir="${output_basedir}/iter_result_${i}_${data_delta}"


    if [ -f "${output_basedir}/complete.txt" ]; then
        echo "Stopping early as the complete condition is met."
        break
    elif [ -f "${output_basedir}/incomplete.txt" ]; then
        echo "Not completed."
    fi

  done

done


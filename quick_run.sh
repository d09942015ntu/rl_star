#!/bin/bash
set -e


mkdir -p results

#---------------------------- Show Theoretical Values -------------------------#

python3 RLstar_theory.py

#---------------------------- Generate pre-train Data -------------------------#

base=2
# ground-truth data
python dataset_generator_zip.py \
    --file-train "data/zip${base}_3_gt.csv" \
    --base ${base}

y2=64
# all delta are non-zero
deltas=(0.2 0.1)
for delta in ${deltas[@]};do
  name_delta="${delta//./}"
  python3 dataset_generator_pretrain.py --data_path="./data/zip2_3_gt.csv" --data_path_out="./data/zip2_3_${name_delta}_${y2}.csv" --delta=${delta} --repeat_nongt=${y2}
done

# one of the delta is zero
delta_zs=(1 2 3)
for dz in ${delta_zs[@]};do
  delta=0.1
  name_delta="${delta//./}"
  python3 dataset_generator_pretrain.py --data_path="./data/zip2_3_gt.csv" --data_path_out="./data/zip2_3_${name_delta}_${dz}_${y2}.csv" --delta=${delta} --repeat_nongt=${y2} --delta_zero ${dz}
done

#---------------------------- Visualize dataset -------------------------#

python3 vis_dataset_prob.py --data_path_gt=./data/zip2_3_gt.csv --data_path=./data/zip2_3_01_64.csv

#---------------------------- RLSTaR Algorithm -------------------------#
model_name="gpt2"
data_type="zip2_3"
dataset_test="./data/${data_type}_gt.csv"
dataset_test_full="./data/${data_type}_gt.csv"
default_temperature=0.5
temperature="${2:-$default_temperature}"
name_t="${temperature//./}"
DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
echo "temperature=${temperature}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

data_deltas=("02_64" "01_64")
epoch=2
repeat=64

for data_delta in ${data_deltas[@]};do
  dataset_train="./data/${data_type}_${data_delta}.csv"
  current_time=$(date +"%Y%m%d_%H%M%S")
  output_basedir="./results/rlstar_${data_type}_t_${name_t}_${data_delta}_${current_time}"
  output_dir="${output_basedir}/iter_result_0_${data_delta}"
  mkdir -p ${output_basedir}

  for i in $(seq 1 3); do
    python3 RLstar_trainer.py  \
          --model_name=${model_name} \
          --dataset_train=${dataset_train} \
          --dataset_eval=${dataset_test_full} \
          --logging_steps=50 \
          --output_dir=${output_dir} \
          --batch_size=32 \
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

#----------------------------Visualization of Experiment Result-------------------------#

python3 vis_rlstar_accuracy.py --data_path_gt=./data/zip2_3_gt.csv --data_path=./data/zip2_3_01_64.csv

echo "Done!"
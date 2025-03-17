#!/bin/bash

base=2

python dataset_generator_zip.py \
    --file-train "data/zip${base}_3_gt.csv" \
    --base ${base}

y2=4096

deltas=(0.2 0.1)
for delta in ${deltas[@]};do
  name_delta="${delta//./}"
  python3 dataset_generator_pretrain.py --data_path="./data/zip2_3_gt.csv" --data_path_out="./data/zip2_3_${name_delta}_${y2}.csv" --delta=${delta} --repeat_nongt=${y2}
done

delta_zs=(1 2 3)
for dz in ${delta_zs[@]};do
  delta=0.1
  name_delta="${delta//./}"
  python3 dataset_generator_pretrain.py --data_path="./data/zip2_3_gt.csv" --data_path_out="./data/zip2_3_${name_delta}_${dz}_${y2}.csv" --delta=${delta} --repeat_nongt=${y2} --delta_zero ${dz}
done




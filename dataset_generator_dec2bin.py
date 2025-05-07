import argparse
import csv
import math
import shutil

import numpy as np


def convert_to_base(num, base, n):
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36.")

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    negative = num < 0
    num = abs(num)
    result = []
    intermediate = []

    for _ in range(n):
        remainder = num % base
        result.append(digits[remainder])
        num //= base
        intermediate.append(num)

    if negative:
        result.append('-')

    return ''.join(result), intermediate


def add_dataset(file_train, file_eval, train_ratio=0.5, eval_ratio=0.5, base=2, n=5):
    dataset_train = [[f"s{i}" for i in range(n + 1)]]
    dataset_eval = [[f"s{i}" for i in range(n + 1)]]
    rng = np.random.RandomState(0)

    for a in range(base ** n):

        bin_a, dec_a = convert_to_base(a, base, n)

        fill_str = math.ceil(math.log(base ** n) / math.log(10))
        # Generate different steps

        s0 = "[x]" + ''.join([f'[{x}]' for x in str(a).zfill(fill_str)])
        s_all = [s0]
        for i in range(0, n):
            si = '[x]' + ''.join([f'[{x}]' for x in str(dec_a[i]).zfill(fill_str)]) + '[y]' + ''.join(
                [f'[{x}]' for x in reversed(str(bin_a[:i + 1]))])
            s_all.append(si)

        # Append to dataset
        rand_number = rng.random()
        if rand_number <= train_ratio:
            dataset_train.append(s_all)
        elif rand_number > train_ratio and rand_number <= train_ratio + eval_ratio:
            dataset_eval.append(s_all)
        else:
            pass

    # Save to CSV file 
    with open(file_train, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataset_train)
    print(f"Dataset saved to {file_train}")
    if len(dataset_eval) > 1:
        with open(file_eval, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(dataset_eval)
        print(f"Dataset saved to {file_eval}")


def main():
    parser = argparse.ArgumentParser(description='Generate datasets with specific configurations.')
    parser.add_argument('--file-train', type=str, default="data/zip2_T_3_1_1.csv",
                        help='Path to the training data file.')
    parser.add_argument('--file-eval', type=str, default="data/zip2_E_3_1_1.csv",
                        help='Path to the evaluation data file.')
    parser.add_argument('--train-ratio', type=float, default=1,
                        help='Ratio of training data.')
    parser.add_argument('--eval-ratio', type=float, default=0,
                        help='Ratio of evaluation data.')
    parser.add_argument('--base', type=int,
                        default=2, help='Base format of the datasets.')
    parser.add_argument('--n', type=int,
                        default=5, help='Base format of the datasets.')

    args = parser.parse_args()

    add_dataset(file_train=args.file_train, file_eval=args.file_eval,
                train_ratio=args.train_ratio, eval_ratio=args.eval_ratio, base=args.base, n=args.n)

    shutil.copy("data/token_zip.json", args.file_train.replace(".csv", "_token.json"))


if __name__ == '__main__':
    main()

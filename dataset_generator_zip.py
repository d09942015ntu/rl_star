import argparse
import csv
import os

import numpy as np
import shutil


def convert_to_base(num, base):
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36.")

    if num == 0:
        return "0"

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    negative = num < 0
    num = abs(num)
    result = []

    while num > 0:
        remainder = num % base
        result.append(digits[remainder])
        num //= base

    if negative:
        result.append('-')

    return ''.join(reversed(result))


def add_dataset(file_train, file_eval, train_ratio=0.5, eval_ratio=0.5, base=2):
    n = 3
    dataset_train = [[f"s{i}" for i in range(n + 1)]]
    dataset_eval = [[f"s{i}" for i in range(n + 1)]]
    rng = np.random.RandomState(0)

    for a in range(base ** n):  # 被加數(000~111)
        for b in range(base ** n):  # 加數(000~111)

            bin_a = convert_to_base(a, base).zfill(n)
            bin_b = convert_to_base(b, base).zfill(n)

            # Generate different steps

            s0 = f"[x]{''.join([f'[{a}]' for a in bin_a])}[.]{''.join([f'[{b}]' for b in bin_b])}"
            s_all = [s0]
            for i in reversed(range(0, n)):
                si = f"[x]{''.join([f'[{a}]' for a in bin_a[:i]])}[.]{''.join([f'[{b}]' for b in bin_b[:i]])}[y]{'[.]'.join([f'[{c[0]}][{c[1]}]' for c in zip(bin_a[i:], bin_b[i:])])}"
                s_all.append(si)

            smi = f"[x]{''.join([f'[{a}]' for a in bin_a[:i]])}[.]{''.join([f'[{b}]' for b in bin_b[:i]])}[y]{'[.]'.join([f'[{c[0]}][{c[1]}]' for c in zip(bin_a[i:], bin_b[i:])])}"
            # sm2 = f"[z][{carrying[0]}][y]{''.join([f'[{c}]' for c in ans])}"
            # sm1 = f"[y][{carrying[0]}]{''.join([f'[{c}]' for c in ans])}"
            # s_all.append(sm2)
            # s_all.append(sm1)

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
    parser.add_argument('--file-train', type=str, default="data/zip2_3_gt.csv",
                        help='Path to the ground-truth data.')
    parser.add_argument('--file-eval', type=str, default="data/zip2_3_eval.csv",
                        help='Path to the hold-out data.')
    parser.add_argument('--train-ratio', type=float, default=1,
                        help='Ratio of training data.')
    parser.add_argument('--eval-ratio', type=float, default=0,
                        help='Ratio of hold out ratio.')
    parser.add_argument('--base', type=int,
                        default=2, help='Base format of the datasets.')

    args = parser.parse_args()

    add_dataset(file_train=args.file_train, file_eval=args.file_eval,
                train_ratio=args.train_ratio, eval_ratio=args.eval_ratio, base=args.base)

    shutil.copy("data/token_zip.json", args.file_train.replace(".csv", "_token.json"))


if __name__ == '__main__':
    main()

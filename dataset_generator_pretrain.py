import argparse
import shutil

import csv
import numpy as np
import pandas as pd


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Load a CSV file and display its contents.')

    # Add a positional argument for the CSV file
    parser.add_argument('--data_path', type=str, default='./data/zip2_3_gt_1.csv', help='Path to the CSV file')
    parser.add_argument('--data_path_out', type=str, default='./data/zip2_3_A_02_64.csv', help='Path to the CSV file')
    parser.add_argument('--repeat_nongt', type=int, default=64, help='Repetition of sampling trajectories')
    parser.add_argument('--delta', type=float, default=0.2, help='Value of delta')
    parser.add_argument('--delta_zero', type=int, default=-1, help='Step of zero delta ,-1 means all delta are non-zero')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the CSV file into a DataFrame
    df = pd.read_csv(args.data_path)

    # Iterate through each row in the DataFrame
    transition = {}
    outputs = [list(df.keys())]
    for index, row in df.iterrows():
        row_list = row.values.tolist()
        for i in range(len(row_list)):
            s_current = row_list[i]
            transition[s_current] = []
            if i < len(row_list) - 1:
                s_next_gt = row_list[i + 1]
                s_nexts = list(set(df[f's{i + 1}'].tolist()))
                M = len(s_nexts)
                # print(M)
                prob = 1 / M
                alpha = prob
                beta = prob
                if args.delta is not None:
                    delta = args.delta
                    if i + 1 == args.delta_zero:
                        delta = 0
                    beta = prob - delta / (M - 1)
                    alpha = prob + delta
                print(f"index={index},M={M},prob={prob},alpha={alpha},beta={beta}")
                for s_next in s_nexts:
                    if s_next == s_next_gt:
                        transition[s_current].append((s_next, alpha))
                    else:
                        transition[s_current].append((s_next, beta))


    rng = np.random.RandomState(0)
    for s0 in df['s0'].tolist():
        for i in range(args.repeat_nongt):
            if i % 100 == 0:
                print(f"repeat:{i}/{args.repeat_nongt}")
            s_current = s0
            s_nexts_weights = transition[s0]
            traj = [s_current]

            while len(s_nexts_weights) > 0:
                # print(s_nexts_weights)
                s_nexts, weights = zip(*s_nexts_weights)
                s_current = rng.choice(s_nexts, p=weights)
                traj.append(s_current)
                s_nexts_weights = transition[s_current]
            outputs.append(traj)

    with open(args.data_path_out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(outputs)
    print(f"Dataset saved to {args.data_path_out}")

    shutil.copy(args.data_path.replace(".csv", "_token.json"), args.data_path_out.replace(".csv", "_token.json"))


if __name__ == "__main__":
    main()

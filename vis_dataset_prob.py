import argparse
from collections import defaultdict
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Load a CSV file and display its contents.')

    # Add a positional argument for the CSV file
    parser.add_argument('--data_path_gt', type=str, default='./data/zip2_3_gt.csv', help='Path to the CSV file')
    parser.add_argument('--data_path', type=str, default='./data/zip2_3_01_64.csv', help='Path to the CSV file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the CSV file into a DataFrame
    df_gt = pd.read_csv(args.data_path_gt)
    df_data = pd.read_csv(args.data_path)

    dict_prob_all={}

    for ikey, jkey in zip(df_gt.keys(), df_gt.keys()[1:]):
        dict_trans = {}
        dict_prob = {}
        for index, row in df_gt.iterrows():
            sn0 = row[ikey]
            dict_trans[sn0] = defaultdict(int)
            dict_prob[sn0] = defaultdict(int)
        for index, row in df_data.iterrows():
            if index % 1000 == 0:
                print(f"trans=({ikey},{jkey})iter={index}/{len(df_data)}")
            sn0 = row[ikey]
            sn1 = row[jkey]
            if sn0 not in dict_trans.keys():
                continue
            dict_trans[sn0][sn1] += 1

        for index, row in df_gt.iterrows():
            sn0 = row[ikey]
            normalize = sum(dict_trans[sn0].values())
            for sn1 in dict_trans[sn0].keys():
                dict_prob[sn0][sn1] = dict_trans[sn0][sn1] / normalize

        dict_s0_idx = dict([(x[1],x[0]) for x in enumerate(df_gt[ikey]) ])
        dict_s1_idx = dict([(x[1],x[0]) for x in enumerate(df_gt[jkey]) ])
        trans_matrix = np.zeros((len(dict_s0_idx),len(dict_s1_idx)))
        for s0 in dict_prob.keys():
            for s1 in dict_prob[s0].keys():
                trans_matrix[dict_s0_idx[s0]][dict_s1_idx[s1]] = dict_prob[s0][s1]
        vis_matrix(trans_matrix, args.data_path.replace(".csv",f"_trans_{ikey}_{jkey}.png"))
        dict_prob_all["%s" % (ikey)] = {"prob":dict_prob,"trans_matrix":trans_matrix.tolist()}
    json.dump(dict_prob_all, open(args.data_path.replace(".csv","_trans.json"),"w") ,indent=2)

def vis_matrix(trans_matrix, fname):
    # Assume trans_matrix, dict_s0_idx, dict_s1_idx are already defined

    plt.rcParams.update({'font.size': 24})  # Adjust the number to your desired font size


    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    # Plot heatmap using imshow
    plt.imshow(trans_matrix, cmap='viridis', aspect='auto')

    # Add a color bar
    plt.colorbar(label='Probability')

    plt.tight_layout()
    plt.savefig(fname)

    json.dump(trans_matrix.tolist(), open(fname.replace(".png",".json"),"w") ,indent=2)



if __name__ == '__main__':
    main()
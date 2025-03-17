import os
from collections import OrderedDict
import json

import matplotlib.pyplot as plt




def delta_t_1(delta, M):
    return ((M - 2) * delta ** 3 + delta ** 2 + delta) / ((M - 1) * delta ** 3 + 1)

def J_P_t(alpha,beta, M):
    return alpha**3+3*(M-1)*alpha*(beta**2)+(M-1)*(M-2)*(beta)**3

def get_vab(delta, M):
    va = (1 + (M - 1) * delta) / M
    vb = (1 - delta) / M
    return va, vb



def plot_simulation_result_values(delta, M, out_name):
    delta_0 = delta
    va, vb = get_vab(delta, M)
    iis = [0]
    deltas = [delta]
    jpt1 = J_P_t(va, vb, M)
    jpts = [jpt1]
    for i in range(1, 10):
        delta = delta_t_1(delta, M)
        va, vb = get_vab(delta, M)
        iis.append(i)
        deltas.append(delta)
        jpt1 = J_P_t(va, vb, M)
        jpts.append(jpt1)
    plt.clf()
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size
    plt.plot(iis, jpts, label='accuracy2', marker='x', alpha=0.5)
    plt.xlabel('$t$')
    plt.ylabel('Values')
    plt.title(f'Plot of accuracy: M={M}')
    plt.legend()

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    os.makedirs("results",exist_ok=True)
    json.dump({'iteration': iis, 'accuracy': jpts},
              open(f'results/json_{out_name}_{M}_{str(delta_0).replace(".", "")}.json', "w"), indent=2)
    plt.savefig(f'results/png_{out_name}_{M}_{str(delta_0).replace(".", "")}.png')

if __name__ == '__main__':
    M = 64
    poly_MN = OrderedDict()
    print(f"M={M}")
    for delta in [0.2, 0.1]:
        plot_simulation_result_values(delta, M,  "theory")

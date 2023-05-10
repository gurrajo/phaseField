import numpy as np
import re
import DMN
import matplotlib.pylab as plt
import time
import random


def read_dataset(file_name):
    with open(f'data/{file_name}.txt', 'r') as file:
        in_data = file.readlines()
    data = np.ndarray((len(in_data), 28))
    for i, s in enumerate(in_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        nums = [float(val) for val in s_out]
        data[i] = nums
    return data


def extract_D_mat(data, ind):
    D_1 = data[ind, 1:10]
    D_2 = data[ind, 10:19]
    D_correct = data[ind, 19:28]
    D1 = np.zeros((3, 3))
    D2 = np.zeros((3, 3))
    DC = np.zeros((3, 3))
    k = 0
    for i in range(3):
        for j in range(3):
            D1[i, j] = D_1[k]
            D2[i, j] = D_2[k]
            DC[i, j] = D_correct[k]
            k += 1

    return D1, D2, DC


def run_train_sample(epoch, M):
    N_s = 1000
    D1 = np.zeros((3, 3))
    D2 = np.zeros((3, 3))
    DC = np.zeros((3, 3))
    nn = DMN.Network(N, D1, D2, DC)
    cost = []

    for i in range(epoch):
        np.random.shuffle(data)
        k = 0
        for i in range(int(N_s/M)):
            for j in range(M):
                (D1, D2, DC) = extract_D_mat(data, k)
                nn.update_phases(D1, D2, DC)
                nn.forward_pass()
                nn.calc_cost()
                nn.backwards_prop()
                k += 1
            nn.learn_step()
            cost.append(np.sum(nn.C)/M)
            nn.C = []
            print(cost[i])
    return cost, nn


def tot_cost(mse):
    C_0 = 1/(2*len(mse))*np.sum(mse)
    return C_0


start_time = time.time()
data = read_dataset("data_set")
N = 8
mini_batch = 20

cost, nn = run_train_sample(1, mini_batch)
print(time.time() - start_time)
plt.plot(range(len(cost)), cost)
plt.show()
print(1)

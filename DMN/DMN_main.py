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


def run_train_sample(epoch, M, nn=False):
    N_s = 1000
    if not nn:
        D1 = np.zeros((3, 3))
        D2 = np.zeros((3, 3))
        DC = np.zeros((3, 3))
        nn = DMN.Network(N, D1, D2, DC)
        print("new network")
    else:
        print("old network")
    cost = []
    z_0 = np.zeros((int(epoch*N_s/M), 1))
    theta_0 = np.zeros((int(epoch * N_s / M), 1))
    z_5 = np.zeros((int(epoch*N_s/M), 1))
    m = 0
    epoch_cost = np.zeros((epoch, 1))
    for i in range(epoch):
        np.random.shuffle(data)
        k = 0
        print("Epoch " + str(i))
        batch_cost = 0
        for l in range(int(N_s/M)):
            for j in range(M):
                (D1, D2, DC) = extract_D_mat(data, k)
                nn.update_phases(D1, D2, DC)
                nn.forward_pass()
                nn.calc_cost()
                nn.backwards_prop()
                k += 1
            nn.learn_step()
            z_0[m] = nn.layers[0][0].z
            theta_0[m] = nn.layers[0][0].theta
            z_5[m] = nn.layers[0][5].z
            cost.append(np.sum(nn.C)/M)
            batch_cost += np.sum(nn.C)/M
            nn.C = []
            print(cost[-1])
            m += 1
        epoch_cost[i] = batch_cost/(N_s/M)
    plt.plot(np.linspace(0, N_s*epoch/M, epoch), epoch_cost)
    plt.plot(range(len(theta_0)), theta_0)
    plt.plot(range(len(z_0)), z_0)
    plt.plot(range(len(z_5)), z_5)
    plt.legend(['Cost', 'Theta0', 'Z0', 'Z5'])
    plt.show()
    return nn, epoch_cost, z_0, z_5


def tot_cost(mse):
    C_0 = 1/(2*len(mse))*np.sum(mse)
    return C_0


start_time = time.time()
data = read_dataset("data_set")
N = 8
mini_batch = 100

nn, epoc_cost, z0, z_5 = run_train_sample(10, mini_batch)
print(time.time() - start_time)

print(1)

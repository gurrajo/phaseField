import numpy as np
import re
import DMN
import matplotlib.pylab as plt
import time
import random
import multiprocessing as mp


def write_dmn(dmn):
    data = []
    for layer in dmn.layers:
        for node in layer:
            data.append(f"{str(node.theta[0])} ")
        data.append("\n")
    for node in dmn.input_layer:
        data.append(f"{str(node.z[0])} ")
    with open(f'data/DMN_1.txt', 'w') as file:
        file.writelines(data)


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


def create_dmn_from_save(dmn_file):
    """
    Create a nn object from saved weights
    :param dmn_file: data file with weights from text file
    :return: nn object
    """
    nn = 0
    return nn

def run_train_sample(epoch, M, nn=False):
    N_s = 800
    if not nn:
        D1 = np.zeros((3, 3))
        D2 = np.zeros((3, 3))
        DC = np.zeros((3, 3))
        nn = DMN.Network(N, D1, D2, DC)
        print("new network")
    else:
        print("old network")
    cost = []
    theta_0 = np.zeros((int(epoch * N_s / M), 1))
    zs = np.zeros((int(epoch*N_s/M), 2**(N-1)))
    m = 0
    epoch_cost = np.zeros((epoch, 1))
    inter_plot = True  # interactive plotting on
    if inter_plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        plt.xlabel("epochs")
    start_time = time.time()
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
            zs[m, :] = nn.zs
            theta_0[m] = nn.layers[0][0].theta
            cost.append(np.sum(nn.C)/M)
            batch_cost += np.sum(nn.C)/M
            nn.C = []
            if inter_plot:
                ax.plot(range(m), zs[0:m, :])
            print(cost[-1])
            m += 1
        epoch_cost[i] = batch_cost/(N_s/M)
        if inter_plot:
            ax2.plot(range(i), epoch_cost[0:i])
            fig.canvas.draw()
            fig.canvas.flush_events()
    print("runtime: " + str(time.time() - start_time) + " seconds")
    if not inter_plot:
        plt.plot(np.linspace(0, N_s*epoch/M, epoch), epoch_cost)
        plt.plot(range(len(theta_0)), theta_0)
        plt.plot(range(len(zs)), zs)
        plt.legend(['Cost', 'Theta0'])
        plt.show()

    return nn, epoch_cost, zs


def run_validation(nn, valid_set):
    """
    :param nn: the DMN
    :param valid_set: data_set for validation.
    :return: validation cost. (Without cost function addition)
    """
    N_s = 200
    nn.C = []
    for i in range(N_s):
        (D1, D2, DC) = extract_D_mat(valid_set, i)
        nn.update_phases(D1, D2, DC)
        nn.forward_pass()
        nn.calc_cost()
    cost = np.sum(nn.C)/N_s
    return cost


data = read_dataset("data_set")
N = 7
mini_batch = 80

nn, epoc_cost, zs = run_train_sample(30, mini_batch)
write_dmn(nn)
valid_data = read_dataset("validation")
valid_cost = run_validation(nn, valid_data)
print("validation cost: " + str(valid_cost))
print(1)

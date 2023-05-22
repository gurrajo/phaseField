import numpy as np
import re
import DMN
import matplotlib.pyplot as plt
import time
import random


def write_dmn(dmn, ind):
    data = []
    for layer in dmn.layers:
        for node in layer:
            data.append(f"{str(node.theta)} ")
        data.append("\n")
    for node in dmn.input_layer:
        data.append(f"{str(node.z)} ")
    with open(f'data/DMN_{ind}.txt', 'w') as file:
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
    with open(f'data/{dmn_file}.txt', 'r') as file:
        in_data = file.readlines()
    N = len(in_data)
    D1 = np.zeros((3, 3))
    D2 = np.zeros((3, 3))
    DC = np.zeros((3, 3))
    nn = DMN.Network(N, D1, D2, DC)
    for i, layer in enumerate(nn.layers):
        thetas = in_data[i].split(" ")
        for j, node in enumerate(layer):
            node.theta = float(thetas[j])
    zs = in_data[-1].split(" ")
    for i, node in enumerate(nn.input_layer):
        node.z = float(zs[i])
        node.w = np.max([node.z, 0])
    return nn


def run_train_sample(epoch, M, ind, nn=False, inter_plot=True, N=False):
    """
    Train a DMN network for a set of epochs
    :param epoch: amount of epochs, one training session over dataset
    :param M: Minibatch size. Amount of samples in a minibatch
    :param nn: A Network object from DMN.If False, create new object
    :param inter_plot: interactive plotting on or off.
    :return: Trained network object. Epoch averaged cost. z activations for each training step.
    """
    N_s = 800
    if not nn:
        D1 = np.zeros((3, 3))
        D2 = np.zeros((3, 3))
        DC = np.zeros((3, 3))
        nn = DMN.Network(N, D1, D2, DC)
        print("new network")
    else:
        N = nn.N
        print("old network")
    cost = []
    theta_0 = np.zeros((int(epoch * N_s / M), 1))
    zs = np.zeros((int(epoch*N_s/M), 2**(N-1)))
    thetas = np.zeros((int(epoch*N_s/M), 2**(N-2)))
    m = 0
    epoch_cost = np.zeros((epoch, 1))
    if inter_plot:
        plt.ion()
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
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
            theta_0[m] = nn.layers[-1][0].theta
            thetas[m, :] = [node.theta for node in nn.layers[0]]
            cost.append(np.sum(nn.C)/M)
            batch_cost += np.sum(nn.C)/M
            nn.C = []
            if inter_plot:
                axs[0].clear()
                axs[2].clear()
                axs[0].plot(range(m), zs[0:m, :])
                axs[2].plot(range(m), thetas[0:m, :])
            print(cost[-1])
            m += 1
        epoch_cost[i] = batch_cost/(N_s/M)
        if inter_plot:
            axs[1].clear()
            axs[0].set_title("z parameters")
            axs[0].set_xlabel("learning steps")
            axs[0].set_ylabel("activation")
            axs[1].set_title("Cost function")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Error")
            axs[2].set_title(r"First layer $\theta$")
            axs[2].set_xlabel("Learning steps")
            axs[2].set_ylabel("Rotation angle")
            axs[1].plot(range(i), epoch_cost[0:i])
            fig.canvas.draw()
            fig.canvas.flush_events()
    print("runtime: " + str(time.time() - start_time) + " seconds")
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].set_title(fr"z parameters $\eta$ = {nn.layers[0][0].eta_z}")
    axs[0].set_xlabel("learning steps")
    axs[0].set_ylabel("activation")
    axs[1].set_title("Cost function")
    axs[1].set_xlabel("Learnings steps")
    axs[1].set_ylabel("Error")
    axs[2].set_title(fr"$\theta$ First layer, $\eta$ = {nn.layers[0][0].eta_theta} ")
    axs[2].set_xlabel("Learning steps")
    axs[2].set_ylabel("Rotation angle")
    axs[1].plot(range(len(cost)), cost)
    axs[2].plot(range(len(thetas)), thetas)
    axs[0].plot(range(len(zs)), zs)
    plt.show()
    plt.savefig(f"train_sample_{ind}.svg")

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
new = True
if new:
    N = 7
    mini_batch = 100
    ind = 160
    nn, epoc_cost, zs = run_train_sample(2, mini_batch, ind, N=N, inter_plot=True)
    write_dmn(nn, ind)
else:
    mini_batch = 50
    ind = 151
    nn_old = create_dmn_from_save(f"DMN_{ind}")
    #nn_old.update_learn_rate(0.01, 0.0002)
    nn_old.lam = 0.04
    nn, epoc_cost, zs = run_train_sample(100, mini_batch, ind, nn=nn_old, inter_plot=True)
    write_dmn(nn, ind+1)

valid_data = read_dataset("validation")
valid_cost = run_validation(nn, valid_data)
print("validation cost: " + str(valid_cost))
print(1)


import numpy as np
import re
import DMN
import DMN_2
import matplotlib.pyplot as plt
import time


def file_reader(filename):
    f = open(filename, 'r')
    Lines = f.readlines()
    line = []
    for Line in Lines:
        nums = Line.split("#")
        points = []
        for num in nums:
            if num == "\n":
                line.append([points])
                break
            point = num.replace("[", '')
            point = point.replace("]",'')
            point = point.split(",")
            points.append([float(point[0]), float(point[1])])
    f.close()
    return line


def write_dmn(dmn, ind):
    data = []
    for node in dmn.input_layer:
        data.append(f"{str(node.theta)} ")
    data.append("\n")
    for layer in dmn.layers:
        for node in layer:
            data.append(f"{str(node.theta)} ")
        data.append("\n")
    for node in dmn.input_layer:
        data.append(f"{str(node.w)} ")
    data.append("\n")
    for layer in dmn.layers:
        for node in layer:
            data.append(f"{str(node.w)} ")
        data.append("\n")
    with open(f'data/DMN_{ind}.txt', 'w') as file:
        file.writelines(data)


def write_data(epoc_cost, zs, thetas, ind):
    data = []
    for i,ep_cost in enumerate(epoc_cost):
        data.append(f"{ep_cost} z:{zs[i,:]} t:{thetas[i,:]}\n")
    with open(f'data/Outdata_{ind}.txt', 'w') as file:
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


def make_dataset_sym(data):
    sym_data = []
    for i in range(200):
        data_line = data[i,:]
        d_corr = data[i, 19:28]
        d_21 = (d_corr[1] + d_corr[3])/2
        d_31 = (d_corr[2] + d_corr[6])/2
        d_32 = (d_corr[5] + d_corr[7])/2
        d_corr[1] = d_21
        d_corr[3] = d_21
        d_corr[2] = d_31
        d_corr[6] = d_31
        d_corr[5] = d_32
        d_corr[7] = d_32
        data_line[19:28] = d_corr
        data_string = ' '.join(str(x) for x in data_line)
        sym_data.append(f"{data_string}\n")
    with open(f'data/Symdata2.txt', 'w') as file:
        file.writelines(sym_data)


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
    N = int(len(in_data)/2)
    D1 = np.zeros((3, 3))
    D2 = np.zeros((3, 3))
    DC = np.zeros((3, 3))
    nn = DMN_2.Network(N, D1, D2, DC)
    theta_in = in_data[0].split(" ")
    for i, node in enumerate(nn.input_layer):
        node.theta = float(theta_in[i])
    for i, layer in enumerate(nn.layers):
        thetas = in_data[i+1].split(" ")
        for j, node in enumerate(layer):
            node.theta = float(thetas[j])
        ind = i
    zs = in_data[ind + 2].split(" ")
    for i, node in enumerate(nn.input_layer):
        node.w = float(zs[i])
    for j, layer in enumerate(nn.layers):
        ws = in_data[ind+3+j].split(" ")
        for k, node in enumerate(layer):
            node.w = float(ws[k])


    return nn


def run_train_sample(epoch, M, ind, nn=False, inter_plot=True, N=False, update_lam=True):
    """
    Train a DMN network for a set of epochs
    :param epoch: amount of epochs, one training session over dataset
    :param M: Minibatch size. Amount of samples in a minibatch
    :param nn: A Network object from DMN.If False, create new object
    :param inter_plot: interactive plotting on or off.
    :return: Trained network object. Epoch averaged cost. z activations for each training step.
    """
    N_s = 200
    if not nn:
        D1 = np.zeros((3, 3))
        D2 = np.zeros((3, 3))
        DC = np.zeros((3, 3))
        nn = DMN_2.Network(N, D1, D2, DC)
        print("new network")
    else:
        N = nn.N
        nn.ws = [node.w for node in nn.input_layer]
        print("old network")
    cost = []
    theta_0 = np.zeros((int(epoch * N_s / M), 1))
    ws = np.zeros((int(epoch*N_s/M), 2))
    cs = np.zeros((int(epoch * N_s / M), 2 ** (N - 1)))
    thetas = np.zeros((int(epoch*N_s/M), 2**(N-1)))
    m = 0
    epoch_cost = np.zeros((epoch, 1))
    epoch_cost_0 = np.zeros((epoch, 1))
    epoch_ws = np.zeros((epoch, 2**(N-1)))
    epoch_thetas = np.zeros((epoch, 2**(N-1)))
    if inter_plot:
        plt.ion()
        fig, axs = plt.subplots(4, 1, constrained_layout=True)
        fig.suptitle(fr"N_s = {N_s}")
        fig.set_size_inches(17, 12, forward=True)
        axs[1].set_yscale("log")
    start_time = time.time()
    for i in range(epoch):
        if update_lam:
            if i == 200:
                nn.lam = nn.lam*3/4
            if i == 500:
                nn.lam = nn.lam*3/4
            if i == 1000:
                nn.lam = nn.lam/2
        np.random.shuffle(data)
        k = 0
        print("Epoch " + str(i))
        if i == 100:
            print(1)
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
            ws[m, :] = [node.w for node in nn.layers[-2]]
            cs[m, :] = [node.learn_w for node in nn.input_layer]
            theta_0[m] = nn.layers[-1][0].theta
            thetas[m, :] = [node.theta for node in nn.input_layer]
            cost.append(np.sum(nn.C)/(2*M))
            batch_cost += np.sum(nn.C)/(2*M)
            nn.C = []
            if inter_plot:
                axs[0].clear()
                axs[2].clear()
                axs[3].clear()

                axs[0].plot(range(m), ws[0:m, :])
                axs[2].plot(range(m), thetas[0:m, :])
                axs[3].plot(range(m), cs[0:m, :])
            print(cost[-1])
            m += 1

        epoch_cost[i] = batch_cost/(N_s/M)
        epoch_ws[i, :] = nn.ws
        epoch_thetas[i, :] = [node.theta for node in nn.input_layer]
        if inter_plot:
            axs[1].clear()
            axs[1].set_yscale("log")
            axs[0].set_title(f"z, deactivated nodes: {nn.ws.count(0)}/{len(nn.ws)}")
            axs[0].set_xlabel("learning steps")
            axs[0].set_ylabel("activation")
            axs[1].set_title("Cost function")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Error")
            axs[2].set_title(r"First layer $\theta$")
            axs[2].set_xlabel("Learning steps")
            axs[2].set_ylabel("Rotation angle")
            axs[3].set_title(fr"$z$ learn rate")
            axs[3].set_xlabel("Learning steps")
            axs[3].set_ylabel(fr"$\eta$")
            axs[1].plot(range(i), epoch_cost[0:i])
            axs[1].plot(range(i), epoch_cost_0[0:i])
            fig.canvas.draw()
            fig.canvas.flush_events()
    plt.savefig(f"train_sample_{ind}_run.svg")
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
    axs[0].plot(range(len(ws)), ws)
    plt.show()
    plt.savefig(f"train_sample_{ind}_ext.svg")

    return nn, epoch_cost_0, epoch_ws, epoch_thetas


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
    cost = np.sum(nn.C0)/(2*N_s)
    return cost


data = read_dataset("Symdata2")
new = True

if new:
    N = 6
    mini_batch = 25
    ind = 70
    nn, epoc_cost, epoch_ws, epoch_thetas = run_train_sample(100, mini_batch, ind, N=N, inter_plot=True, update_lam=False)
    write_dmn(nn, ind)
    write_data(epoc_cost, epoch_ws, epoch_thetas, ind)
else:
    mini_batch = 25
    ind = 61
    nn_old = create_dmn_from_save(f"DMN_{ind}")
    nn, epoc_cost, epoch_zs, epoch_thetas = run_train_sample(100, mini_batch, ind+1, nn=nn_old, inter_plot=True, update_lam=False)
    write_dmn(nn, ind+1)
    write_data(epoc_cost, epoch_zs, epoch_thetas, ind+1)

valid_data = read_dataset("validation")
valid_cost = run_validation(nn, valid_data)
print("validation cost: " + str(valid_cost))
print(1)

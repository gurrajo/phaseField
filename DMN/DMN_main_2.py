import numpy as np
import re
import DMN_2
import matplotlib.pyplot as plt
import time
import os


def elast_bounds(D_1, D_2):
    """
    calculate lower Reuss value and upper Voigt value for the representative elasticity parameters
    """
    vol_frac = 0.5
    D_voigt = np.zeros((3, 3))
    D_voigt[0, 0] = 1 / (vol_frac / D_1[0, 0] + (1 - vol_frac) / D_2[0, 0])
    D_voigt[0, 1] = 1 / (vol_frac / D_1[0, 1] + (1 - vol_frac) / D_2[0, 1])
    D_voigt[1, 0] = D_voigt[0, 1]
    D_voigt[1, 1] = 1 / (vol_frac / D_1[1, 1] + (1 - vol_frac) / D_2[1, 1])
    D_voigt[2, 2] = 1 / (vol_frac / D_1[2, 2] + (1 - vol_frac) / D_2[2, 2])

    C_voigt = np.zeros((1, 4))
    C_voigt[0, 0] = 1 / D_voigt[0, 0]
    C_voigt[0, 1] = 1 / D_voigt[1, 1]
    C_voigt[0, 3] = 1 / (2 * D_voigt[2, 2])
    C_voigt[0, 2] = -C_voigt[0, 0] * D_voigt[1, 0]

    D_reuss = np.zeros((3, 3))
    D_reuss[0, 0] = vol_frac * D_1[0, 0] + (1 - vol_frac) * D_2[0, 0]
    D_reuss[0, 1] = vol_frac * D_1[0, 1] + (1 - vol_frac) * D_2[0, 1]
    D_reuss[1, 0] = D_reuss[0, 1]
    D_reuss[1, 1] = vol_frac * D_1[1, 1] + (1 - vol_frac) * D_2[1, 1]
    D_reuss[2, 2] = vol_frac * D_1[2, 2] + (1 - vol_frac) * D_2[2, 2]

    C_reuss = np.zeros((1, 4))
    C_reuss[0, 0] = 1 / D_reuss[0, 0]
    C_reuss[0, 1] = 1 / D_reuss[1, 1]
    C_reuss[0, 3] = 1 / (2 * D_reuss[2, 2])
    C_reuss[0, 2] = -C_reuss[0, 0] * D_reuss[1, 0]
    return C_reuss, C_voigt, D_reuss, D_voigt


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
    for node in dmn.input_layer:
        data.append(f"{str(node.bias)} ")
    data.append("\n")
    for layer in dmn.layers:
        for node in layer:
            data.append(f"{str(node.bias)} ")
        data.append("\n")
    with open(f'data/DMN_{ind}.txt', 'w') as file:
        file.writelines(data)


def write_data(epoc_cost, zs, ind):
    data = []
    for i,ep_cost in enumerate(epoc_cost):
        data.append(f"{ep_cost} z:{zs[i,:]}\n")
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
    N = int(len(in_data)/3)
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
    bias = in_data[ind + 4 + j].split(" ")
    for i, node in enumerate(nn.input_layer):
        node.bias = float(bias[i])
    for m, layer in enumerate(nn.layers):
        bias = in_data[ind + 5 + j + m].split(" ")
        for k, node in enumerate(layer):
            node.bias = float(bias[k])
    return nn


def run_train_sample(epoch, M, ind, N_s, data_file, nn=False, inter_plot=True, N=False, update_lam=True):
    """
    Train a DMN network for a set of epochs
    :param epoch: amount of epochs, one training session over dataset
    :param M: Minibatch size. Amount of samples in a minibatch
    :param nn: A Network object from DMN.If False, create new object
    :param inter_plot: interactive plotting on or off.
    :return: Trained network object. Epoch averaged cost. z activations for each training step.
    """
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
    error = []
    w_list = []
    th_list = []
    bias_list = []
    th_list.append(np.zeros((int(epoch * N_s / M), 2 ** (N - 1))))
    w_list.append(np.zeros((int(epoch*N_s/M), 2 ** (N - 1))))
    bias_list.append(np.zeros((int(epoch*N_s/M), 2 ** (N - 1))))
    for i, layer in enumerate(nn.layers):
        w_list.append(np.zeros((int(epoch*N_s/M), 2 ** (N - 2 - i))))
        th_list.append(np.zeros((int(epoch*N_s/M), 2 ** (N - 2 - i))))
        bias_list.append(np.zeros((int(epoch*N_s/M), 2 ** (N - 2 - i))))

    m = 0
    epoch_cost = np.zeros((epoch, 1))
    epoch_error = np.zeros((epoch, 1))
    epoch_ws = np.zeros((epoch, 2**(N-1)))
    if inter_plot:
        plt.ion()
        fig, axs = plt.subplots(nn.N, 3, constrained_layout=True)
        fig.suptitle(fr"dataset: {data_file}")
        fig.set_size_inches(16, 11, forward=True)
        axs[nn.N - 1, 0].set_yscale("log")
        axs[nn.N-1, 1].set_yscale("log")
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
        batch_cost = 0
        batch_error = 0
        for l in range(int(N_s/M)):
            for j in range(M):
                (D1, D2, DC) = extract_D_mat(data, k)
                nn.update_phases(D1, D2, DC)
                nn.forward_pass()
                nn.calc_cost()
                nn.backwards_prop()
                k += 1
            nn.learn_step()

            # --Store values for plotting--
            bias_list[0][m,:] = [node.bias for node in nn.input_layer]
            w_list[0][m, :] = [node.w for node in nn.input_layer]
            th_list[0][m, :] = [node.theta for node in nn.input_layer]
            for p, layer in enumerate(nn.layers):
                w_list[p + 1][m, :] = [node.w for node in layer]
                th_list[p + 1][m, :] = [node.theta for node in layer]
                bias_list[p + 1][m, :] = [node.bias for node in layer]
            cost.append(np.sum(nn.C)/(2*M))
            batch_cost += np.sum(nn.C)/(2*M)
            batch_error += np.sum(nn.error) /M
            print(np.sum(nn.error)/M)
            nn.error = []
            nn.C = []
            if inter_plot:
                axs[0, 0].clear()
                axs[0, 1].clear()
                axs[0, 2].clear()
                axs[0, 0].plot(range(m), w_list[0][0:m, :])
                axs[0, 1].plot(range(m), th_list[0][0:m, :])
                axs[0, 2].plot(range(m), bias_list[0][0:m, :])
                for l in range(nn.N - 2):
                    axs[l + 1, 0].clear()
                    axs[l + 1, 1].clear()
                    axs[l + 1, 2].clear()
                    axs[l + 1, 0].plot(range(m), w_list[l+1][0:m, :])
                    axs[l + 1, 1].plot(range(m), th_list[l + 1][0:m, :])
                    axs[l + 1, 2].plot(range(m), bias_list[l + 1][0:m, :])
                axs[nn.N - 1, 2].plot(range(m), bias_list[nn.N - 1][0:m, :])
            m += 1

        epoch_error[i] = M*batch_error/N_s
        epoch_cost[i] = batch_cost/(N_s/M)
        if np.mod(i,100) == 0:
            write_dmn(nn, f"{ind}_{i/100}")

        if inter_plot:
            axs[0,0].set_title(f"weights")
            axs[0,0].set_xlabel("learning steps")
            axs[0,0].set_ylabel("activation")

            axs[0,1].set_title(f"thetas")
            axs[0,1].set_xlabel("learning steps")
            axs[0,1].set_ylabel("rot angles")
            for j in range(nn.N - 2):
                axs[j+1,1].set_title(f"thetas")
                axs[j+1, 1].set_xlabel("learning steps")
                axs[j+1, 1].set_ylabel("rot angles")

                axs[j+1, 0].set_title(f"weights")
                axs[j+1, 0].set_xlabel("learning steps")
                axs[j+1, 0].set_ylabel("activation")
            # for j in range(nn.N - 2):
            #     axs[j].set_title(f"weights")
            #     axs[j].set_xlabel("learning steps")
            #     axs[j].set_ylabel("activation")
            axs[nn.N-1,0].clear()
            axs[nn.N-1,0].set_yscale("log")
            axs[nn.N-1,0].set_title("Error")
            axs[nn.N-1,0].set_xlabel("Epochs")
            axs[nn.N-1,0].plot(range(i), epoch_error[0:i])

            axs[nn.N-1,1].clear()
            axs[nn.N-1,1].set_yscale("log")
            axs[nn.N-1,1].set_title("Cost")
            axs[nn.N-1,1].set_xlabel("Epochs")
            axs[nn.N-1,1].plot(range(i), epoch_cost[0:i])
            fig.canvas.draw()
            fig.canvas.flush_events()
    plt.savefig(f"train_sample_{ind}_run.svg")
    print("runtime: " + str(time.time() - start_time) + " seconds")
    return nn, epoch_cost, epoch_ws


def showcase_temp_variation(dmn):
    T = np.linspace(20, 150, 100)
    E_1 = 6.2225*10**4 - 75*(T + 273.15)
    nu_1 = 0.36
    G_1 = E_1/(2*(1+nu_1))
    E_2 = 9*10**4
    nu_2 = 0.35
    G_2 = E_2 / (2 * (1 + nu_2))
    #  isotropic assumption
    E_out = np.zeros((100, 1))
    E_reuss = np.zeros((100, 1))
    E_voigt = np.zeros((100, 1))
    for i, t in enumerate(T):
        D_1 = np.array([[1/E_1[i], -nu_1/E_1[i], 0], [-nu_1/E_1[i], 1/E_1[i], 0], [0, 0, 1/G_1[i]]])
        D_2 = np.array([[1/E_2, -nu_2/E_2, 0], [-nu_2/E_2, 1/E_2, 0], [0, 0, 1/G_2]])
        C_reuss, C_voigt, D_reuss, D_voigt = elast_bounds(D_1, D_2)
        dmn.update_phases(D_1, D_2, np.zeros((3, 3)))
        dmn.forward_pass()
        D_out = dmn.get_comp()
        E_out[i] = 1/D_out[0, 0]
        E_voigt[i] = C_voigt[0, 0]
        E_reuss[i] = C_reuss[0, 0]
    fig = plt.figure()
    plt.plot(T, E_1)
    plt.plot(T, E_2*np.ones((100, 1)))
    plt.plot(T, E_out)
    plt.plot(T, E_voigt)
    plt.plot(T, E_reuss)
    plt.title("Temperature varying phase 1")
    plt.xlabel(r"Temperature $c^0$")
    plt.ylabel("E [MPa]")
    plt.legend(["Phase 1", "Phase 2", "Homogenized material", "Voigt", "Reuss"])
    plt.savefig("temp_varying.svg")
    return E_out


def validate_multiple(folder, data_set):
    error = np.zeros((31, 1))
    for i in range(31):
        dmn = create_dmn_from_save(f"{folder}/{i}")
        error[i], error_v, error_r = run_validation(dmn, data_set)
    fig = plt.figure()
    plt.plot(range(31), error)
    plt.title("Error during training")
    plt.xlabel(r"100 Epochs")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("validation_error_morph_1.svg")


def run_validation(nn, valid_set):
    """
    :param nn: the DMN
    :param valid_set: data_set for validation.
    :return: validation cost. (Without cost function addition)
    """
    N_s = len(valid_set)
    nn.C = []
    nn.error = []
    voigt_error = []
    reuss_error = []
    comp_correct = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    for i in range(N_s):
        (D1, D2, DC) = extract_D_mat(valid_set, i)
        C_reuss, C_voigt, D_reuss, D_voigt = elast_bounds(D1, D2)

        reuss_error.append((np.linalg.norm(comp_correct * (D_reuss - DC), 'fro')) / (np.linalg.norm(comp_correct * DC, 'fro')))
        voigt_error.append((np.linalg.norm(comp_correct * (D_voigt - DC), 'fro')) / (np.linalg.norm(comp_correct * DC, 'fro')))
        nn.update_phases(D1, D2, DC)
        nn.forward_pass()
        nn.calc_cost()
    error = np.sum(nn.error)/N_s
    error_v = np.sum(voigt_error)/N_s
    error_r = np.sum(reuss_error)/N_s
    return error, error_v, error_r

#data_file = "data_comb"
data_file = "Symdata2"
#data_file = "data_comb_5"
data = read_dataset(data_file)
new = False
#nn_old = create_dmn_from_save("DMN_222_14.0")
#valid_cost, error_v, error_r = run_validation(nn_old, data)
#showcase_temp_variation(nn_old)
print(1)

if new:
    N_s = 150
    N = 8
    mini_batch = 30
    ind = 1050
    nn, epoc_cost, epoch_ws = run_train_sample(500, mini_batch, ind, N_s, data_file, N=N, inter_plot=True, update_lam=False )
    write_dmn(nn, ind)
    write_data(epoc_cost, epoch_ws, ind)
else:
    N_s = 200
    mini_batch = 20
    ind = 1404
    nn_old = create_dmn_from_save(f"DMN_{ind}")
    nn, epoc_cost, epoch_zs = run_train_sample(1000, mini_batch, ind+1, N_s, data_file, nn=nn_old, inter_plot=True, update_lam=False)
    write_dmn(nn, ind+1)
    write_data(epoc_cost, epoch_zs, ind+1)

valid_data = read_dataset("validation")
valid_cost, error_v, error_r = run_validation(nn, valid_data)
print("validation cost: " + str(valid_cost))
print(1)

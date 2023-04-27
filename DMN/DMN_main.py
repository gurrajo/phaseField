import numpy as np
import re
import DMN


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


def run_train_sample(nn, N_s):
    D_bar = []
    mse = []
    for i in range(N_s):
        (D1, D2, DC) = extract_D_mat(data, i)
        nn = DMN.Network(N, D1, D2, DC)
        nn.forward_pass()
        D_bar.append(nn.get_comp())
        mse.append(nn.mse)
    return D_bar, mse


def tot_cost(mse):
    C_0 = 1/(2*len(mse))*np.sum(mse)
    return C_0


data = read_dataset("data_set")
N = 5
N_s = 1000
D1 = np.zeros((3, 3))
D2 = np.zeros((3, 3))
DC = np.zeros((3, 3))
dmn_test = DMN.Network(N, D1, D2, DC)
D_bar, mse = run_train_sample(dmn_test, N_s)
(D1, D2, DC) = extract_D_mat(data, 0)
C_0 = tot_cost(mse)

print(1)


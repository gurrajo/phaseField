import numpy as np


def hyper_cube(n, d):
    rng = np.random.default_rng()
    samples = rng.uniform(size=(n, d))
    perms = np.tile(np.arange(1, n + 1), (d, 1))
    for i in range(d):
        rng.shuffle(perms[i, :])
    perms = perms.T
    samples = (perms - samples) / n
    return samples


def comp_mats(sample):
    E_1 = np.zeros((3,3))
    E_2 = np.zeros((3,3))

    uni_11 = sample[0]*2 - 1
    uni_11_2 = sample[1]*2 - 1
    uni_44 = sample[2]*8 - 4

    uni_nu_1 = sample[3]*0.4 + 0.3
    uni_nu_2 = sample[4]*0.4 + 0.3

    E_1[0,0] = 10**(uni_11/-2)
    E_1[1,1] = 1/E_1[0,0]
    E_2[0,0] = np.sqrt(10**(uni_44)/(10**(uni_11_2)))
    E_2[1,1] = E_2[0,0]*10**uni_11_2

    nu_1 = uni_nu_1*np.sqrt(E_1[1,1]/E_1[0,0])
    nu_2 = uni_nu_2*np.sqrt(E_2[1,1]/E_2[0,0])
    return E_1,E_2,nu_1,nu_2


n = 1000
d = 5
hyperCube = hyper_cube(n,d)
E_1 = []
E_2 = []
nu_1 = []
nu_2 = []

for i, samples in enumerate(hyperCube):
    E_1_temp, E_2_temp, nu_1_temp, nu_2_temp = comp_mats(samples)
    E_1.append(E_1_temp)
    E_2.append(E_2_temp)
    nu_1.append(nu_1_temp)
    nu_2.append(nu_2_temp)

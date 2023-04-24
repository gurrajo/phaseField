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

    uni_G_1 = sample[5]*0.25 + 0.25
    uni_G_2 = sample[6]*0.25 + 0.25

    E_1[0,0] = np.sqrt(1/10**(uni_11))
    E_1[1,1] = 1/E_1[0,0]
    E_2[1,1] = np.sqrt((10**(uni_11_2))*(10**uni_44))
    E_2[0,0] = (10**uni_44)/E_2[1,1]

    nu_1 = uni_nu_1*np.sqrt(E_1[1,1]/E_1[0,0])
    nu_2 = uni_nu_2*np.sqrt(E_2[1,1]/E_2[0,0])

    G_1 = uni_G_1*np.sqrt(E_1[1,1]*E_1[0,0])

    G_2 = uni_G_2 * np.sqrt(E_2[1, 1] * E_2[0, 0])

    D_1 = np.zeros((3,3))
    D_1[0,0] = 1/E_1[0,0]
    D_1[1,1] = 1/E_1[1,1]
    D_1[0,1] = -nu_1/E_1[1,1]
    D_1[1,0] = D_1[0,1]
    D_1[2,2] = 1/(2*G_1)

    D_2 = np.zeros((3,3))
    D_2[0,0] = 1/E_2[0,0]
    D_2[1,1] = 1/E_2[1,1]
    D_2[0,1] = -nu_2/E_2[1,1]
    D_2[1,0] = D_2[0,1]
    D_2[2,2] = 1/(2*G_2)

    return D_1, D_2


def phaseinator():
    n = 1000
    d = 7
    hyperCube = hyper_cube(n,d)
    D_1 = []
    D_2 = []

    for i, samples in enumerate(hyperCube):
        D_1_temp, D_2_temp = comp_mats(samples)
        D_1.append(D_1_temp)
        D_2.append(D_2_temp)

    return D_1, D_2

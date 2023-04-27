import numpy as np


class Branch:
    """
    represents a branch in the deep-material network.
    The branch is the connection between two nodes
    """
    def __init__(self, child_1, child_2, f, theta, inp):
        self.input = inp
        self.ch_1 = child_1
        self.ch_2 = child_2
        self.f_1 = f
        self.f_2 = 1 - self.f_1
        self.theta = theta
        self.D_r = np.zeros((3, 3))  # output compliance before rotation
        self.D_bar = np.zeros((3, 3))  # output compliance after rotation

    def homogen(self):
        if self.input:
            self.D_1 = self.ch_1
            self.D_2 = self.ch_2
        else:
            self.D_1 = self.ch_1.D_bar
            self.D_2 = self.ch_2.D_bar

        gamma = self.f_1*self.D_2[0, 0] + self.f_2*self.D_1[0, 0]
        self.D_r[0, 0] = 1/gamma*(self.D_1[0, 0]*self.D_2[0, 0])
        self.D_r[0, 1] = 1/gamma*(self.f_1*(self.D_1[0, 1]*self.D_2[0, 0]) + self.f_2*self.D_1[0, 0]*self.D_2[0, 1])
        self.D_r[1, 0] = self.D_r[0, 1]
        self.D_r[0, 2] = 1/gamma*(self.f_1*self.D_1[0, 2]*self.D_2[0, 0] + self.f_2 * self.D_1[0, 0]*self.D_2[0, 2])
        self.D_r[2, 0] = self.D_r[0, 2]
        self.D_r[1, 1] = self.f_1*self.D_1[1, 1] + self.f_2*self.D_2[1, 1] - 1/gamma*self.f_1*self.f_2*(self.D_1[0,1] - self.D_2[0,1])**2
        self.D_r[1, 2] = self.f_1*self.D_1[1, 2] + self.f_2*self.D_2[1, 2] - 1/gamma*self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])*(self.D_1[0,1] - self.D_2[0,1])
        self.D_r[2, 1] = self.D_r[1, 2]
        self.D_r[2, 2] = self.f_1*self.D_1[2, 2] + self.f_2*self.D_2[2, 2] - 1/gamma*self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])**2

    def rot_mat(self, theta):
        R = np.zeros((3, 3))
        R[0, :] = [np.cos(theta)**2, np.sin(theta)**2, np.sqrt(2)*np.sin(theta)*np.cos(theta)]
        R[1, :] = [np.sin(theta) ** 2, np.cos(theta) ** 2, -np.sqrt(2) * np.sin(theta) * np.cos(theta)]
        R[2, :] = [-np.sqrt(2) * np.sin(theta) * np.cos(theta), np.sqrt(2) * np.sin(theta) * np.cos(theta), np.cos(theta)**2 - np.sin(theta)**2]
        return R

    def rotate_comp(self):
        self.D_bar = np.matmul(np.matmul(self.rot_mat(-self.theta), self.D_r), self.rot_mat(self.theta))


class Network:
    """
    Represents the network structure.
    Contains N layers
    """
    def __init__(self, N, D_1, D_2, D_correct):
        self.N = N
        self.D_1 = D_1  # phase 1 compliance (all samples)
        self.D_2 = D_2  # phase 2 compliance (all samples)
        self.D_correct = D_correct  # correct compliance after homogenization
        self.input_layer = []
        for j in range(2 ** (N - 1)):
            if np.mod(j, 2) == 0:
                # Phase 1 input nodes
                self.input_layer.append(Branch(D_1, D_1, 1, 0, True))
            else:
                # ´Phase 2 input nodes
                self.input_layer.append(Branch(D_2, D_2, 1, 0, True))
        self.layers = []
        self.layers.append(self.input_layer)
        for i in range(1, N):
            self.layers.append(self.fill_layer(i))

    def fill_layer(self, i):
        rng = np.random.default_rng()
        prev_layer = self.layers[i-1]
        new_layer = []
        for i in range(int(len(prev_layer)/2)):
            samples = rng.uniform(size=(2, 1))
            new_layer.append(Branch(prev_layer[i*2], prev_layer[i*2 + 1], samples[0], samples[1]*2*np.pi, False))
        return new_layer

    def get_comp(self):
        return self.layers[-1][0].D_bar

    def forward_pass(self):
        for i in range(self.N):
            for parent in self.layers[i]:
                parent.homogen()
                parent.rotate_comp()
        self.calc_cost()

    def update_phases(self, D_1, D_2):
        for j in range(2 ** (self.N - 1)):
            if np.mod(j, 2) == 0:
                # Phase 1 input nodes
                self.layers[0][j].ch_1 = D_1
                self.layers[0][j].ch_2 = D_1
            else:
                # ´Phase 2 input nodes
                self.layers[0][j].ch_1 = D_2
                self.layers[0][j].ch_2 = D_2

    def calc_cost(self):
        D_bar = self.get_comp()
        self.mse = np.linalg.norm((self.D_correct - D_bar), 'fro')**2/np.linalg.norm(self.D_correct, 'fro') ** 2



import numpy as np


class Branch:
    """
    represents a branch in the deep-material network.
    The branch is the connection between two child nodes and one parent node
    """
    def __init__(self, child_1, child_2, theta, inp, w):
        self.alpha = np.zeros((3, 3))
        self.dC_dZj = []
        self.dC_dTheta = []
        self.c_theta = 0.1  # RMSprop parameter
        self.c_z = 0.1  # RMSprop parameter
        self.learn_z = 0
        self.learn_theta = 0
        self.eta_z = 0.001
        self.eta_theta = 0.001  # learning rates
        self.ch_1 = child_1
        self.ch_2 = child_2
        self.theta = theta
        self.D_r = np.zeros((3, 3))  # output compliance before rotation
        self.D_bar = np.zeros((3, 3))  # output compliance after rotation
        self.delta = np.zeros((3, 3))
        self.z = w
        self.w = w
        if not inp:
            self.f_1 = self.ch_1.w/(self.ch_1.w + self.ch_2.w)
            self.f_2 = 1 - self.f_1
        else:
            self.f_1 = 1
            self.f_2 = 0

    def homogen(self):

        if (self.ch_1.w + self.ch_2.w) == 0:
            self.f_1 = 0
        else:
            self.f_1 = self.ch_1.w / (self.ch_1.w + self.ch_2.w)
        self.D_1 = self.ch_1.D_bar
        self.D_2 = self.ch_2.D_bar

        self.f_2 = 1 - self.f_1
        gamma = self.f_1*self.D_2[0, 0] + self.f_2*self.D_1[0, 0]
        self.D_r[0, 0] = (self.D_1[0, 0]*self.D_2[0, 0])/gamma
        self.D_r[0, 1] = (self.f_1*self.D_1[0, 1]*self.D_2[0, 0] + self.f_2*self.D_1[0, 0]*self.D_2[0, 1])/gamma
        self.D_r[1, 0] = self.D_r[0, 1]
        self.D_r[0, 2] = 1/gamma*(self.f_1*self.D_1[0, 2]*self.D_2[0, 0] + self.f_2 * self.D_1[0, 0]*self.D_2[0, 2])
        self.D_r[2, 0] = self.D_r[0, 2]
        self.D_r[1, 1] = self.f_1*self.D_1[1, 1] + self.f_2*self.D_2[1, 1] - self.f_1*self.f_2*(self.D_1[0,1] - self.D_2[0,1])**2/gamma
        self.D_r[1, 2] = self.f_1*self.D_1[1, 2] + self.f_2*self.D_2[1, 2] - self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])*(self.D_1[0,1] - self.D_2[0,1])/gamma
        self.D_r[2, 1] = self.D_r[1, 2]
        self.D_r[2, 2] = self.f_1*self.D_1[2, 2] + self.f_2*self.D_2[2, 2] - self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])**2/gamma

        self.gamma = gamma

    def rot_mat(self, theta):
        R = np.zeros((3, 3))
        R[0, :] = [np.cos(theta)**2, np.sin(theta)**2, np.sqrt(2)*np.sin(theta)*np.cos(theta)]
        R[1, :] = [np.sin(theta) ** 2, np.cos(theta) ** 2, -np.sqrt(2) * np.sin(theta) * np.cos(theta)]
        R[2, :] = [-np.sqrt(2) * np.sin(theta) * np.cos(theta), np.sqrt(2) * np.sin(theta) * np.cos(theta), np.cos(theta)**2 - np.sin(theta)**2]
        return R

    def rot_mat_prime(self, theta):
        R = np.zeros((3, 3))
        R[0, :] = [-np.sin(2*theta), np.sin(2*theta), np.sqrt(2)*np.cos(2*theta)]
        R[1, :] = [np.sin(2*theta), -np.sin(2*theta), -np.sqrt(2)*np.cos(2*theta)]
        R[2, :] = [-np.sqrt(2) * np.cos(theta*2), np.sqrt(2) * np.cos(theta*2), -2*np.sin(2*theta)]
        return R

    def rotate_comp(self):
        D_bar = np.matmul(np.matmul(self.rot_mat(-self.theta), self.D_r), self.rot_mat(self.theta))
        self.D_bar = D_bar

    def gradients(self):
        # derivative of D comps with respect to D_r
        D_d_Dr = np.zeros((3, 3, 3, 3))
        R_1 = self.rot_mat(-self.theta)
        R_2 = self.rot_mat(self.theta)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        D_d_Dr[i,j,k,l] = R_1[i,k]*R_2[l,j]
        self.D_d_Dr = D_d_Dr

        # derivatives of D_r comps with respect to first child D1

        Dr_d_D1 = np.zeros((3, 3, 3, 3))
        temp = np.zeros((3,))
        temp[:] = [self.f_1 * self.D_2[0, 0] ** 2 / self.gamma, self.f_2 * (-self.D_r[0, 1] + self.D_2[0, 1]),
                   self.f_2 * (-self.D_r[0, 2] + self.D_2[0, 2])]
        Dr_d_D1[:, 0, 0, 0] = 1 / self.gamma * temp
        temp[:] = [self.f_2 * (-self.D_r[0, 1] + self.D_2[0, 1]),
                   self.f_1 * self.f_2 ** 2 * (self.D_1[0, 1] - self.D_2[0, 1]) ** 2 / self.gamma,
                   self.f_1 * self.f_2 ** 2 * (self.D_1[0, 2] - self.D_2[0, 2]) * (
                               self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma]
        Dr_d_D1[:, 1, 0, 0] = 1 / self.gamma * temp
        temp[:] = [self.f_2 * (-self.D_r[0, 2] + self.D_2[0, 2]),
                   self.f_1 * (self.f_2 ** 2) * (self.D_1[0, 2] - self.D_2[0, 2]) * (
                               self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma,
                   self.f_1 * (self.f_2 ** 2) * ((self.D_1[0, 2] - self.D_2[0, 2]) ** 2) / self.gamma]
        Dr_d_D1[:, 2, 0, 0] = 1 / self.gamma * temp

        Dr_d_D1[:, 0, 0, 1] = [0, self.f_1 * self.D_2[0, 0] / self.gamma, 0]
        Dr_d_D1[:, 1, 0, 1] = [self.f_1 * self.D_2[0, 0] / self.gamma,
                               -2 * self.f_1 * self.f_2 * (self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma,
                               -self.f_1 * self.f_2 * (self.D_1[0, 2] - self.D_2[0, 2]) / self.gamma]
        Dr_d_D1[:, 2, 0, 1] = [0, -self.f_1 * self.f_2 * (self.D_1[0, 2] - self.D_2[0, 2]) / self.gamma, 0]

        Dr_d_D1[:, :, 1, 0] = Dr_d_D1[:, :, 0, 1]

        Dr_d_D1[:, 0, 0, 2] = [0, 0, self.f_1 * self.D_2[0, 0] / self.gamma]
        Dr_d_D1[:, 1, 0, 2] = [0, 0, -self.f_1 * self.f_2 * (self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma]
        Dr_d_D1[:, 2, 0, 2] = [self.f_1 * self.D_2[0, 0] / self.gamma,
                               -self.f_1 * self.f_2 * (self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma,
                               -2 * self.f_1 * self.f_2 * (self.D_1[0, 2] - self.D_2[0, 2]) / self.gamma]

        Dr_d_D1[:, :, 2, 0] = Dr_d_D1[:, :, 0, 2]

        Dr_d_D1[:, :, 1, 1] = [[0, 0, 0], [0, self.f_1, 0], [0, 0, 0]]
        Dr_d_D1[:, :, 1, 2] = [[0, 0, 0], [0, 0, self.f_1], [0, self.f_1, 0]]
        Dr_d_D1[:, :, 2, 1] = Dr_d_D1[:, :, 1, 2]
        Dr_d_D1[:, :, 2, 2] = [[0, 0, 0], [0, 0, 0], [0, 0, self.f_1]]

        temp = np.zeros((3,))
        Dr_d_D2 = np.zeros((3, 3, 3, 3))
        temp[:] = [self.f_2 * self.D_1[0, 0] ** 2 / self.gamma,
                   self.f_1*self.f_2*self.D_1[0,0]*(self.D_1[0,1] - self.D_2[0,1])/self.gamma,
                   self.f_1*self.f_2*self.D_1[0,0]*(self.D_1[0,2] - self.D_2[0,2])/self.gamma]
        Dr_d_D2[:, 0, 0, 0] = 1 / self.gamma * temp
        temp[:] = [self.f_1*self.f_2*self.D_1[0,0]*(self.D_1[0,1] - self.D_2[0,1])/self.gamma,
                   self.f_2 * self.f_1 ** 2 * (self.D_1[0, 1] - self.D_2[0, 1]) ** 2 / self.gamma,
                   self.f_2 * self.f_1 ** 2 * (self.D_1[0, 2] - self.D_2[0, 2]) * (
                               self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma]
        Dr_d_D2[:, 1, 0, 0] = 1 / self.gamma * temp
        temp[:] = [self.f_1*self.f_2*self.D_1[0,0]*(self.D_1[0,2] - self.D_2[0,2])/self.gamma,
                   self.f_2 * self.f_1 ** 2 * (self.D_1[0, 2] - self.D_2[0, 2]) * (
                               self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma,
                   self.f_2 * self.f_1 ** 2 * (self.D_1[0, 2] - self.D_2[0, 2]) ** 2 / self.gamma]
        Dr_d_D2[:, 2, 0, 0] = 1 / self.gamma * temp

        Dr_d_D2[:, 0, 0, 1] = [0, self.f_2 * self.D_1[0, 0] / self.gamma, 0]
        Dr_d_D2[:, 1, 0, 1] = [self.f_2 * self.D_1[0, 0] / self.gamma,
                               2 * self.f_2 * self.f_1 * (self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma,
                               self.f_2 * self.f_1 * (self.D_1[0, 2] - self.D_2[0, 2]) / self.gamma]
        Dr_d_D2[:, 2, 0, 1] = [0, self.f_1 * self.f_2 * (self.D_1[0, 2] - self.D_2[0, 2]) / self.gamma, 0]
        Dr_d_D2[:, :, 1, 0] = Dr_d_D2[:, :, 0, 1]
        Dr_d_D2[:, 0, 0, 2] = [0, 0, self.f_2 * self.D_1[0, 0] / self.gamma]
        Dr_d_D2[:, 1, 0, 2] = [0, 0, self.f_2 * self.f_1 * (self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma]
        Dr_d_D2[:, 2, 0, 2] = [self.f_2 * self.D_1[0, 0] / self.gamma,
                               self.f_2 * self.f_1 * (self.D_1[0, 1] - self.D_2[0, 1]) / self.gamma,
                               2 * self.f_2 * self.f_1 * (self.D_1[0, 2] - self.D_2[0, 2]) / self.gamma]
        Dr_d_D2[:, :, 2, 0] = Dr_d_D2[:, :, 0, 2]
        Dr_d_D2[:, :, 1, 1] = [[0, 0, 0], [0, self.f_2, 0], [0, 0, 0]]
        Dr_d_D2[:, :, 1, 2] = [[0, 0, 0], [0, 0, self.f_2], [0, self.f_2, 0]]
        Dr_d_D2[:, :, 2, 1] = Dr_d_D2[:, :, 1, 2]
        Dr_d_D2[:, :, 2, 2] = [[0, 0, 0], [0, 0, 0], [0, 0, self.f_2]]

        self.D_d_theta = np.matmul(np.matmul(-self.rot_mat_prime(-self.theta), self.D_r), self.rot_mat(self.theta)) + np.matmul(np.matmul(self.rot_mat(-self.theta), self.D_r), self.rot_mat_prime(self.theta))
        self.Dr_d_D1 = Dr_d_D1
        self.Dr_d_D2 = Dr_d_D2
        self.D_d_Dr = D_d_Dr

    def theta_grad(self):
        self.D_d_theta = -np.matmul(np.matmul(self.rot_mat_prime(-self.theta), self.D_r), self.rot_mat(self.theta)) + np.matmul(np.matmul(self.rot_mat(-self.theta), self.D_r), self.rot_mat_prime(self.theta))
        D_d_Dr = np.zeros((3, 3, 3, 3))
        R_1 = self.rot_mat(-self.theta)
        R_2 = self.rot_mat(self.theta)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        D_d_Dr[i, j, k, l] = R_1[i, k] * R_2[l, j]
        self.D_d_Dr = D_d_Dr
        self.D_d_Dr = D_d_Dr

    def update_theta(self):
        gamma = 0.985
        self.c_theta = self.c_theta * gamma + (1 - gamma) * ((np.mean(self.dC_dTheta)) ** 2)
        learn = self.eta_theta / (np.sqrt(self.c_theta) + 1E-6)
        self.learn_theta = np.min([learn, 0.4])
        self.theta -= self.learn_theta * np.median(self.dC_dTheta)
        if self.theta > np.pi/2:
            self.theta -= np.pi
        elif self.theta < -np.pi/2:
            self.theta += np.pi
        self.dC_dTheta = []

    def update_z(self):
        """
        Update weight for branch node. use RELu for input layer only.
        :return:
        """
        if self.z <= 0:
            self.z = 0
        else:
            gamma = 0.985
            self.c_z = self.c_z*gamma + (1-gamma)*(np.mean(self.dC_dZj))**2
            learn = self.eta_z/(np.sqrt(self.c_z) + 1E-6)
            self.learn_z = np.min([learn, 0.1])
            self.z -= self.learn_z*(np.mean(self.dC_dZj))
            if self.z <= 0:
                self.z = 0
        self.w = self.z
        self.dC_dZj = []

    def calc_delta(self, child_1):
        delta_new = np.zeros((3, 3))
        if child_1:  # if current node is child 1 of the parent node
            Dr_d_D = self.Dr_d_D1
        else:
            Dr_d_D = self.Dr_d_D2
        for j in range(3):
            for k in range(3):
                delta_new[j, k] = np.sum(self.alpha*Dr_d_D[:, :, j, k])

        return delta_new


class Network:
    """
    Represents the network structure.
    Contains N layers
    """
    def __init__(self, N, D_1, D_2, D_correct):
        self.C = []
        self.C0 = []
        self.N = N  # network depth
        self.D_1 = D_1  # phase 1 compliance (all samples)
        self.D_2 = D_2  # phase 2 compliance (all samples)
        self.D_correct = D_correct  # correct compliance after homogenization
        self.input_layer = []
        rng = np.random.default_rng()
        #zs = np.linspace(0.2, 0.8, 2**(N-1))
        for j in range(2 ** (N - 1)):
            samples = rng.uniform(size=(2, 1))
            if np.mod(j, 2) == 0:
                # Phase 1 input nodes
                in_node = Branch(D_1, D_1, (samples[0][0]*np.pi - np.pi/2),True, samples[1][0]*0.6 + 0.2)
                in_node.D_r = D_1
            else:
                # Phase 2 input nodes
                in_node = Branch(D_2, D_2, (samples[0][0]*np.pi - np.pi/2),True, samples[1][0]*0.6 + 0.2)
                in_node.D_r = D_2
            self.input_layer.append(in_node)

        self.layers = []
        self.zs = [np.max([node.z, 0]) for node in self.input_layer]
        for i in range(0, N-1):
            self.layers.append(self.fill_layer(i))

    def fill_layer(self, i):
        if i == 0:
            prev_layer = self.input_layer
        else:
            prev_layer = self.layers[i-1]
        new_layer = []
        rng = np.random.default_rng()
        for j in range(int(len(prev_layer)/2)):
            samples = rng.uniform(size=(2, 1))
            new_layer.append(Branch(prev_layer[j*2], prev_layer[j*2 + 1], (samples[0][0]*np.pi - np.pi/2), False, 0.5))
        return new_layer

    def get_comp(self):
        return self.layers[-1][0].D_bar

    def forward_pass(self):
        for node in self.input_layer:
            node.rotate_comp()
        for layer in self.layers:
            for node in layer:
                node.homogen()
                node.rotate_comp()
                node.gradients()

    def update_phases(self, D_1, D_2, DC):
        self.D_correct = DC
        for j in range(2 ** (self.N - 1)):
            if np.mod(j, 2) == 0:
                # Phase 1 input nodes
                self.input_layer[j].D_r = D_1
            else:
                # Phase 2 input nodes
                self.input_layer[j].D_r = D_2

    def update_learn_rate(self, new_eta_t, new_eta_z):
        for layer in self.layers:
            for node in layer:
                node.eta_theta = new_eta_t
                node.eta_z = new_eta_z

    def calc_cost(self):
        D_bar = self.get_comp()
        self.del_C = (D_bar - self.D_correct)/((np.linalg.norm(self.D_correct, 'fro'))**2)  # cost gradient
        self.C.append(((np.linalg.norm(self.D_correct - D_bar, 'fro'))**2)/((np.linalg.norm(self.D_correct, 'fro'))**2))

    def backwards_prop(self):
        for i, layer in enumerate(reversed(self.layers)):
            m = 0
            for k, node in enumerate(layer):
                if i == 0:
                    # output layer
                    delta_new = self.del_C
                else:
                    parent_node = prev_layer[m]
                    if np.mod(k, 2) == 0:
                        delta_new = parent_node.calc_delta(True)
                    else:
                        delta_new = parent_node.calc_delta(False)
                        m += 1
                node.delta = delta_new
                alpha = np.zeros((3, 3))
                for n in range(3):
                    for l in range(3):
                        alpha[n, l] = np.sum(delta_new * node.D_d_Dr[:, :, n, l])
                node.alpha = alpha
                dC_d_theta = np.sum(delta_new*node.D_d_theta)
                node.dC_dTheta.append(dC_d_theta)
                #dC_dZj = np.sum(alpha)
                #node.dC_dZj.append(dC_dZj)
            prev_layer = layer
        for j, node in enumerate(self.input_layer):
            if node.z <= 0:
                node.dC_dZj = 0
                node.dC_dTheta.append(0)
            else:
                parent_ind = int((j - np.mod(j, 2)) / 2)
                parent_node = self.layers[0][parent_ind]
                if np.mod(j, 2) == 0:
                    delta_new = parent_node.calc_delta(True)
                else:
                    delta_new = parent_node.calc_delta(False)
                node.theta_grad()
                alpha = np.zeros((3, 3))
                for n in range(3):
                    for l in range(3):
                        alpha[n, l] = np.sum(delta_new * node.D_d_Dr[:, :, n, l])
                dC_dZj = np.sum(delta_new)
                node.dC_dZj.append(dC_dZj)

                dC_d_theta = np.sum(delta_new * node.D_d_theta)
                node.dC_dTheta.append(dC_d_theta)

    def learn_step(self):
        for i, node in enumerate(self.input_layer):
            node.update_z()
            node.update_theta()
        for layer in self.layers:
            for node in layer:
                #node.update_z()
                node.update_theta()
        self.zs = [node.z for node in self.input_layer]


def component_vec(matrix):
    vec = np.zeros((1, 6))
    k = 0
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 0:
                continue
            elif i == 2 and j == 0:
                continue
            elif i == 2 and j == 1:
                continue
            vec[0, k] = matrix[i, j]
            k += 1
    return vec

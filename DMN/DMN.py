import numpy as np


class Branch:
    """
    represents a branch in the deep-material network.
    The branch is the connection between two child nodes and one parent node
    """
    def __init__(self, child_1, child_2, theta, inp, z):
        self.dC_dZj = []
        self.dC_dTheta = []
        self.eta_z = 0.1
        self.eta_theta = 0.1  # learning rates
        self.input = inp
        self.ch_1 = child_1
        self.ch_2 = child_2
        self.theta = theta
        self.D_r = np.zeros((3, 3))  # output compliance before rotation
        self.D_bar = np.zeros((3, 3))  # output compliance after rotation
        self.delta = np.zeros((3, 3))
        if not inp:
            self.w = self.ch_1.w + self.ch_2.w
            self.f_1 = self.ch_1.w/self.w
        else:
            self.z = z
            self.w = np.max([self.z, 0])
            self.w = np.min([self.w, 1])
            self.f_1 = self.w
        self.f_2 = 1 - self.f_1

    def homogen(self):
        if self.input:
            self.D_1 = self.ch_1.D_bar
            self.D_2 = self.ch_2.D_bar
            self.w = np.max([self.z, 0])
            self.w = np.min([self.w, 1])
            self.f_1 = self.w
        else:
            self.w = self.ch_1.w + self.ch_2.w
            self.f_1 = self.ch_1.w / self.w
            self.D_1 = self.ch_1.D_bar
            self.D_2 = self.ch_2.D_bar

        self.f_2 = 1 - self.f_1
        gamma = self.f_1*self.D_2[0, 0] + self.f_2*self.D_1[0, 0]
        self.D_r[0, 0] = 1/gamma*(self.D_1[0, 0]*self.D_2[0, 0])
        self.D_r[0, 1] = 1/gamma*(self.f_1*self.D_1[0, 1]*self.D_2[0, 0] + self.f_2*self.D_1[0, 0]*self.D_2[0, 1])
        self.D_r[1, 0] = self.D_r[0, 1]
        self.D_r[0, 2] = 1/gamma*(self.f_1*self.D_1[0, 2]*self.D_2[0, 0] + self.f_2 * self.D_1[0, 0]*self.D_2[0, 2])
        self.D_r[2, 0] = self.D_r[0, 2]
        self.D_r[1, 1] = self.f_1*self.D_1[1, 1] + self.f_2*self.D_2[1, 1] - 1/gamma*self.f_1*self.f_2*(self.D_1[0,1] - self.D_2[0,1])**2
        self.D_r[1, 2] = self.f_1*self.D_1[1, 2] + self.f_2*self.D_2[1, 2] - 1/gamma*self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])*(self.D_1[0,1] - self.D_2[0,1])
        self.D_r[2, 1] = self.D_r[1, 2]
        self.D_r[2, 2] = self.f_1*self.D_1[2, 2] + self.f_2*self.D_2[2, 2] - 1/gamma*self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])**2

        self.D_r_vec = component_vec(self.D_r)
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
        self.D_bar_vec = component_vec(D_bar)

    def gradients(self):
        # derivative of D comps with respect to D_r
        D_d_Dr = np.zeros((3,3,3,3))
        R_1 = self.rot_mat(-self.theta)
        R_2 = self.rot_mat(self.theta)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        D_d_Dr[i,j,k,l] = R_1[i,k]*R_2[l,j]
        self.D_d_Dr = D_d_Dr

        # derivatives of D_r comps with respect to first child D1
        Dr_d_D1 = np.zeros((3,3,3,3))
        temp = np.zeros((3,))
        temp[:] = [self.f_1 * self.D_2[0, 0] ** 2 / self.gamma, self.f_2 * (-self.D_r[0, 1] + self.D_2[0, 1]),self.f_2 * (-self.D_r[0, 2] + self.D_2[0, 2])]
        Dr_d_D1[:,0,0,0] = 1/self.gamma*temp
        temp[:] =[self.f_2 * (-self.D_r[0, 1] + self.D_2[0, 1]),
                                               self.f_1*self.f_2**2*(self.D_1[0,1]-self.D_2[0,1])**2/self.gamma,
                                               self.f_1*self.f_2**2*(self.D_1[0,2]-self.D_2[0,2])*(self.D_1[0,1]-self.D_2[0,1])/self.gamma]
        Dr_d_D1[:, 1, 0, 0] = 1 / self.gamma *temp
        temp[:] =[self.f_2*(-self.D_r[0,2] + self.D_2[0,2]),
                                        self.f_1*self.f_2**2*(self.D_1[0,2]-self.D_2[0,2])*(self.D_1[0,1]-self.D_2[0,1])/self.gamma,
                                        self.f_1*self.f_2**2*(self.D_1[0,2]-self.D_2[0,2])**2/self.gamma]
        Dr_d_D1[:, 2, 0, 0] = 1 / self.gamma* temp

        Dr_d_D1[:,0,0,1] =[0,self.f_1*self.D_2[0,0]/self.gamma,0]
        Dr_d_D1[:, 1, 0, 1] =[self.f_1*self.D_2[0,0]/self.gamma,
                             -2*self.f_1*self.f_2*(self.D_1[0,1] - self.D_2[0,1])/self.gamma,
                             -self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])/self.gamma]
        Dr_d_D1[:, 2, 0, 1] = [0,-self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])/self.gamma,0]

        Dr_d_D1[:,:,1,0] = Dr_d_D1[:,:,0,1]

        Dr_d_D1[:,0,0,2] = [0,0,self.f_1*self.D_2[0,0]/self.gamma]
        Dr_d_D1[:, 1, 0, 2] = [0, 0, -self.f_1*self.f_2*(self.D_1[0,1] - self.D_2[0,1])/self.gamma]
        Dr_d_D1[:, 2, 0, 2] = [self.f_1*self.D_2[0,0]/self.gamma,
                              -self.f_1*self.f_2*(self.D_1[0,1] - self.D_2[0,1])/self.gamma,
                              -2*self.f_1*self.f_2*(self.D_1[0,2] - self.D_2[0,2])/self.gamma]

        Dr_d_D1[:, :, 2, 0] = Dr_d_D1[:, :, 0, 2]

        Dr_d_D1[:,:,1,1] = [[0,0,0],[0,self.f_1,0],[0,0,0]]
        Dr_d_D1[:, :, 1, 2] = [[0, 0, 0], [0, 0, self.f_1], [0, self.f_1, 0]]
        Dr_d_D1[:, :, 2, 1] = Dr_d_D1[:, :, 1, 2]
        Dr_d_D1[:, :, 2, 2] = [[0, 0, 0], [0, 0, 0], [0, 0, self.f_1]]

        Dr_d_D2 = np.zeros((3,3,3,3))
        temp[:] = [self.f_1*self.D_1[0,0]**2/self.gamma,
                                        self.f_2*(-self.D_r[0,1] + self.D_1[0,1]),
                                        self.f_2*(-self.D_r[0,2] + self.D_1[0,2])]
        Dr_d_D2[:,0,0,0] = 1/self.gamma*temp
        temp[:] = [self.f_2 * (-self.D_r[0, 1] + self.D_1[0, 1]),
                                               self.f_1*self.f_2**2*(self.D_2[0,1]-self.D_1[0,1])**2/self.gamma,
                                               self.f_1*self.f_2**2*(self.D_2[0,2]-self.D_1[0,2])*(self.D_2[0,1]-self.D_1[0,1])/self.gamma]
        Dr_d_D2[:, 1, 0, 0] = 1 / self.gamma * temp
        temp[:] = [self.f_2*(-self.D_r[0,2] + self.D_1[0,2]),
                                        self.f_1*self.f_2**2*(self.D_2[0,2]-self.D_1[0,2])*(self.D_2[0,1]-self.D_1[0,1])/self.gamma,
                                        self.f_1*self.f_2**2*(self.D_2[0,2]-self.D_1[0,2])**2/self.gamma]
        Dr_d_D2[:,2,0,0] = 1/self.gamma*temp

        Dr_d_D2[:,0,0,1] =[0,self.f_1*self.D_1[0,0]/self.gamma,0]
        Dr_d_D2[:, 1, 0, 1] =[self.f_1*self.D_1[0,0]/self.gamma,
                             -2*self.f_1*self.f_2*(self.D_2[0,1] - self.D_1[0,1])/self.gamma,
                             -self.f_1*self.f_2*(self.D_2[0,2] - self.D_1[0,2])/self.gamma]
        Dr_d_D2[:, 2, 0, 1] = [0,-self.f_1*self.f_2*(self.D_2[0,2] - self.D_1[0,2])/self.gamma,0]
        Dr_d_D2[:, :, 1, 0] = Dr_d_D2[:, :, 0, 1]
        Dr_d_D2[:,0,0,2] =[0,0,self.f_1*self.D_1[0,0]/self.gamma]
        Dr_d_D2[:, 1, 0, 2] =[0, 0, -self.f_1*self.f_2*(self.D_2[0,1] - self.D_1[0,1])/self.gamma]
        Dr_d_D2[:, 2, 0, 2] =[self.f_1*self.D_1[0,0]/self.gamma,
                              -self.f_1*self.f_2*(self.D_2[0,1] - self.D_1[0,1])/self.gamma,
                              -2*self.f_1*self.f_2*(self.D_2[0,2] - self.D_1[0,2])/self.gamma]
        Dr_d_D2[:, :, 2, 0] = Dr_d_D2[:, :, 0, 2]
        Dr_d_D2[:,:,1,1] = [[0,0,0],[0,self.f_1,0],[0,0,0]]
        Dr_d_D2[:, :, 1, 2] = [[0, 0, 0], [0, 0, self.f_1], [0, self.f_1, 0]]
        Dr_d_D2[:, :, 2, 1] = Dr_d_D2[:, :, 1, 2]
        Dr_d_D2[:, :, 2, 2] = [[0, 0, 0], [0, 0, 0], [0, 0, self.f_1]]

        Dr_d_f1 = np.zeros((3,3))
        Dr_d_f1[0,0] = 1/self.gamma*(self.D_1[0,0] - self.D_2[0,0])*self.D_r[0,0]
        Dr_d_f1[0, 1] = 1 / self.gamma * ((self.D_1[0,0] - self.D_2[0,0])*self.D_r[0,1] + self.D_1[0,1]*self.D_2[0,0] - self.D_2[0,1]*self.D_1[0,0])
        Dr_d_f1[1,0] = Dr_d_f1[0,1]
        Dr_d_f1[0, 2] = 1 / self.gamma * ((self.D_1[0,0] - self.D_2[0,0])*self.D_r[0,2] + self.D_1[0,2]*self.D_2[0,0] - self.D_2[0,2]*self.D_1[0,0])
        Dr_d_f1[2, 0] = Dr_d_f1[0, 2]
        Dr_d_f1[1,1] = self.D_1[1,1] - self.D_2[1,1] + 1/self.gamma**2*(self.f_1**2*self.D_2[0,0] - self.f_2**2*self.D_1[0,0])*(self.D_1[0,1] - self.D_2[0,1])**2
        Dr_d_f1[1, 2] = self.D_1[1, 2] - self.D_2[1, 2] + 1 / self.gamma ** 2 * (
                    self.f_1 ** 2 * self.D_2[0, 0] - self.f_2**2 * self.D_1[0, 0]) * (self.D_1[0, 2] - self.D_2[0, 2])*(self.D_1[0,1] - self.D_2[0,1])
        Dr_d_f1[2,1] = Dr_d_f1[1,2]
        Dr_d_f1[2, 2] = self.D_1[2, 2] - self.D_2[2, 2] + 1 / self.gamma ** 2 * (
                    self.f_1 ** 2 * self.D_2[0, 0] - self.f_2 ** 2 * self.D_1[0, 0]) * (
                                    self.D_1[0, 2] - self.D_2[0, 2]) ** 2

        self.D_d_theta = np.matmul(np.matmul(-self.rot_mat_prime(-self.theta), self.D_bar), self.rot_mat(self.theta)) +\
                    np.matmul(np.matmul(-self.rot_mat(-self.theta), self.D_bar), self.rot_mat_prime(self.theta))

        self.Dr_d_D1 = Dr_d_D1
        self.Dr_d_D2 = Dr_d_D2
        self.D_d_Dr = D_d_Dr
        self.Dr_d_f1 = Dr_d_f1

    def update_weights(self):
        """
        Update weight for branch node. use RELu for input layer only.
        :return:
        """
        if self.input:
            self.z -= self.eta_z*np.mean(self.dC_dZj)
        self.theta -= self.eta_theta*np.mean(self.dC_dTheta)
        self.dC_dZj = []
        self.dC_dTheta = []

    def calc_delta(self, child_1):
        delta_new = np.zeros((3, 3))
        if child_1:  # if current node is child 1 of the parent node
            Dr_d_D = self.Dr_d_D1
        else:
            Dr_d_D = self.Dr_d_D2
        temp = np.zeros((3, 3))
        for j in range(3):
            for k in range(3):
                temp[j, k] = np.sum(self.delta*self.D_d_Dr[:, :, j, k])
        for j in range(3):
            for k in range(3):
                delta_new[j, k] = np.sum(temp*Dr_d_D[j, k, :, :])

        return delta_new


class Network:
    """
    Represents the network structure.
    Contains N layers
    """
    def __init__(self, N, D_1, D_2, D_correct):
        self.C = []
        self.N = N  # network depth
        self.D_1 = D_1  # phase 1 compliance (all samples)
        self.D_2 = D_2  # phase 2 compliance (all samples)
        self.D_correct = D_correct  # correct compliance after homogenization
        self.input_layer = []
        rng = np.random.default_rng()
        for j in range(2 ** (N - 1)):
            if np.mod(j, 2) == 0:
                # Phase 1 input nodes
                in_node = Branch(D_1, D_1, 0, True,0)
            else:
                # Phase 2 input nodes
                in_node = Branch(D_2, D_2, 0, True,0)
            in_node.D_bar = D_1
            self.input_layer.append(in_node)

        self.layers = []
        first_layer = []
        for j in range(int(len(self.input_layer)/2)):
            samples = rng.uniform(size=(2, 1))
            first_layer.append(Branch(self.input_layer[j*2], self.input_layer[j*2 + 1], samples[0]*np.pi - np.pi/2, True, samples[1]))
        self.layers.append(first_layer)
        for i in range(1, N-1):
            self.layers.append(self.fill_layer(i))

    def fill_layer(self, i):
        prev_layer = self.layers[i-1]
        new_layer = []
        rng = np.random.default_rng()
        for j in range(int(len(prev_layer)/2)):
            samples = rng.uniform(size=(1, 1))
            new_layer.append(Branch(prev_layer[j*2], prev_layer[j*2 + 1], samples[0]*np.pi - np.pi/2, False, 0))
        return new_layer

    def get_comp(self):
        return self.layers[-2][0].D_bar

    def forward_pass(self):
        for layer in self.layers:
            for parent in layer:
                parent.homogen()
                parent.rotate_comp()
                parent.gradients()

    def update_phases(self, D_1, D_2, DC):
        self.D_correct = DC
        for j in range(2 ** (self.N - 1)):
            if np.mod(j, 2) == 0:
                # Phase 1 input nodes
                self.input_layer[j].D_bar = D_1
            else:
                # Phase 2 input nodes
                self.input_layer[j].D_bar = D_2

    def calc_cost(self):
        D_bar = self.get_comp()
        self.del_C = (D_bar - self.D_correct)/(np.linalg.norm(self.D_correct, 'fro')**2)  # cost gradient
        self.C.append(np.linalg.norm((self.D_correct - D_bar), 'fro')**2/np.linalg.norm(D_bar)**2)

    def backwards_prop(self):
        for i, layer in enumerate(reversed(self.layers)):
            for k, node in enumerate(layer):
                if i == 0:
                    # output layer
                    delta_0 = self.del_C
                    delta_new = delta_0
                    prev_layer = layer
                else:
                    if np.mod(k, 2) == 0:
                        parent_node = prev_layer[int((i-np.mod(i, 2))/2)]
                        delta_new = parent_node.calc_delta(True)
                    else:
                        parent_node = prev_layer[int((i - np.mod(i, 2)) / 2)]
                        delta_new = parent_node.calc_delta(False)
                node.delta = delta_new
                dC_d_theta = np.sum(delta_new*node.D_d_theta)
                node.dC_dTheta.append(dC_d_theta)
            prev_layer = layer
        for j, node in enumerate(self.layers[0]):  # update the layers with activations z
            dC_dZj = 0
            for i, node_2 in enumerate(self.layers[0]):
                parent_ind = int((i - np.mod(i, 2)) / 2)
                parent_node = self.layers[1][parent_ind]
                if i == j:
                    dwn_dwn = 1
                else:
                    dwn_dwn = 0
                if parent_ind*2 + 1 == j:
                    dwn_dwnp = 1
                else:
                    dwn_dwnp = 0
                df_dw = 1/parent_node.w*(dwn_dwn - node_2.f_1*dwn_dwnp)
                alpha = np.zeros((3, 3))
                for k in range(3):
                    for m in range(3):
                        alpha[k, m] = np.sum(parent_node.delta[k, m]*node_2.D_d_Dr[:, :, k, m])
                temp_2 = np.sum(alpha*node_2.Dr_d_f1)
                dC_dZj += df_dw*temp_2
            if node.z < 0:
                dC_dZj = 0
            node.dC_dZj.append(dC_dZj)

    def learn_step(self):
        for layer in self.layers:
            for node in layer:
                node.update_weights()


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

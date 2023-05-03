import numpy as np


class Branch:
    """
    represents a branch in the deep-material network.
    The branch is the connection between two nodes
    """
    def __init__(self, child_1, child_2, theta, inp):
        self.input = inp
        self.ch_1 = child_1
        self.ch_2 = child_2
        if not inp:
            self.w = self.ch_1.w + self.ch_2.w
            self.f_1 = self.ch_1.w/self.w
        else:
            self.w = 0
            self.f_1 = self.w
            self.z = np.max(self.w)
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

        self.D_r_vec = component_vec(self.D_r)
        self.gamma = gamma

    def rot_mat(self, theta):
        R = np.zeros((3, 3))
        R[0, :] = [np.cos(theta)**2, np.sin(theta)**2, np.sqrt(2)*np.sin(theta)*np.cos(theta)]
        R[1, :] = [np.sin(theta) ** 2, np.cos(theta) ** 2, -np.sqrt(2) * np.sin(theta) * np.cos(theta)]
        R[2, :] = [-np.sqrt(2) * np.sin(theta) * np.cos(theta), np.sqrt(2) * np.sin(theta) * np.cos(theta), np.cos(theta)**2 - np.sin(theta)**2]
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



        self.Dr_d_D1 = Dr_d_D1
        self.Dr_d_D2 = Dr_d_D2
        self.D_d_Dr = D_d_Dr
        self.Dr_d_f1 = Dr_d_f1



    def update_weights(self, del_C):
        """
        Update weight for branch node. use RELu for input layer only.
        :return:
        """

    def calc_delta(self, delta_pre, child_1):
        delta_new = np.zeros((1,9))
        if child_1:  # if current node is child 1 of the parent node
            Dr_d_D = self.Dr_d_D1
        else:
            Dr_d_D = self.Dr_d_D1
        temp = np.zeros((3,3))
        for j in range(3):
            for k in range(3):
                temp[j, k] = np.sum(delta_pre*self.D_d_Dr[:,:,j,k])
        for j in range(3):
            for k in range(3):
                delta_new[j,k] = np.sum(temp*Dr_d_D[:,:,j,k])
        return delta_new


class Network:
    """
    Represents the network structure.
    Contains N layers
    """
    def __init__(self, N, D_1, D_2, D_correct):
        self.N = N  # network depth
        self.D_1 = D_1  # phase 1 compliance (all samples)
        self.D_2 = D_2  # phase 2 compliance (all samples)
        self.D_correct = D_correct  # correct compliance after homogenization
        self.input_layer = []
        rng = np.random.default_rng()
        for j in range(2 ** (N - 1)):
            samples = rng.uniform(size=(2, 1))
            if np.mod(j, 2) == 0:
                # Phase 1 input nodes
                in_node = Branch(D_1, D_1, samples[1]*np.pi/2 - np.pi/2, True)
            else:
                # ´Phase 2 input nodes
                in_node = Branch(D_2, D_2, samples[1]*np.pi/2 - np.pi/2, True)
            in_node.w = samples[0]
            self.input_layer.append(in_node)

        self.layers = []
        self.layers.append(self.input_layer)
        for i in range(1, N):
            self.layers.append(self.fill_layer(i))

    def fill_layer(self, i):
        prev_layer = self.layers[i-1]
        new_layer = []
        rng = np.random.default_rng()
        for i in range(int(len(prev_layer)/2)):
            samples = rng.uniform(size=(1, 1))
            new_layer.append(Branch(prev_layer[i*2], prev_layer[i*2 + 1], samples[0]*np.pi/2 - np.pi/2, False))
        return new_layer

    def get_comp(self):
        return self.layers[-1][0].D_bar

    def forward_pass(self):
        for i in range(self.N):
            for parent in self.layers[i]:
                parent.homogen()
                parent.rotate_comp()
                parent.gradients()


    def update_phases(self, D_1, D_2, DC):
        self.D_correct = DC
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
        D = np.zeros((1, 9))
        D_cor = np.zeros((1, 9))
        k = 0
        for i in range(3):
            for j in range(3):
                D_cor[0,k] = self.D_correct[i,j]
                D[0,k] = D_bar[i,j]
                k += 1
        self.del_C = (D_bar - self.D_correct)/(np.linalg.norm(self.D_correct, 'fro')**2)  # cost gradient

    def backwards_prop(self):
        delta_pre = np.zeros((1, 9))
        for i, layer in enumerate(reversed(self.layers)):
            for k, node in enumerate(layer):
                if i == 0:
                    # output layer
                    delta_0 = self.del_C
                    delta_pre = delta_0
                else:
                    if 
                    delta_new = node.calc_delta(delta_pre)
            print(1)


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

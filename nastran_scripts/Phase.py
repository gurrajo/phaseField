import re
import numpy as np
import sympy as sm
from sympy.solvers import solve
import subprocess
import scipy
from scipy import optimize
import time
import os


class PhaseMat:
    """
    Object representing a phase of a material
    """

    def __init__(self, D):
        E_1 = 1/D[0,0]
        E_2 = 1/D[1,1]
        nu_12 = -D[1,0]*E_1
        G_12 = 1/(2*D[2,2])

        self.E_1 = list("%.6f" % E_1)
        while len(self.E_1) > 8:
            self.E_1.pop(-1)
        self.E_1 = "".join(self.E_1)

        self.E_2 = list("%.6f" % E_2)
        while len(self.E_2) > 8:
            self.E_2.pop(-1)
        self.E_2 = "".join(self.E_2)

        self.nu_12 = list("%.6f" % nu_12)
        while len(self.nu_12) > 8:
            self.nu_12.pop(-1)
        self.nu_12 = "".join(self.nu_12)

        self.G_12 = list("%.6f" % G_12)
        while len(self.G_12) > 8:
            self.G_12.pop(-1)
        self.G_12 = "".join(self.G_12)

        self.D = D
        self.C = np.ndarray((1, 4))
        self.C[0] = [E_1, E_2, nu_12, G_12]


class Micro:
    def __init__(self, phase_1, phase_2, test_nr, grid_data, e_area):
        self.strain = np.zeros((1,3))
        self.strain[0] = [0.04, 0.04, 0.04]
        self.vol_frac = 0.5
        self.grid_data = grid_data
        self.e_area = e_area
        self.n_el = len(e_area)  # number of elements
        # 3 start files, one for each load case
        self.start_file_x = "new_orig_x_10.2"
        self.start_file_y = "new_orig_y_10.2"
        self.start_file_xy = "new_orig_xy_10.2"
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.test_nr = test_nr
        #self.change_material(self.start_file_x)
        #self.change_material(self.start_file_y)
        #self.change_material(self.start_file_xy)
        #self.run_nastran()  # generate f06 file for each load case

    def calc_stresses(self, el_nodes):
        stress_x = self.ele_stress(self.start_file_x)
        stress_y = self.ele_stress(self.start_file_y)
        stress_xy = self.ele_stress(self.start_file_xy)

        self.stress_x = self.calc_stress(stress_x)
        self.stress_y = self.calc_stress(stress_y)
        self.stress_xy = self.calc_stress(stress_xy)

        # self.stress_x = self.sort_forces(self.start_file_x)
        # self.stress_y = self.sort_forces(self.start_file_y)
        # self.stress_xy = self.sort_forces(self.start_file_xy)
        self.D = self.calc_comp_mat()
        C = np.zeros((1,4))
        C[0,0] = 1/self.D[0,0]
        C[0,1] = 1/self.D[1,1]
        C[0,2] = -C[0,0]*self.D[1,0]
        C[0,3] = 1/(2*self.D[2,2])
        self.C = C

    def change_material(self, start_file):
        # for orthotropic materials
        with open(f'nastran_input/{start_file}.bdf', 'r') as file:
            # read a list of lines into data
            data = file.readlines()
        next_line = False
        for i, line in enumerate(data):
            if next_line == "Mat_1":
                data[i] = (
                    f'MAT8           7{self.phase_1.E_1}{self.phase_1.E_2}{self.phase_1.nu_12}{self.phase_1.G_12}1.0     1.0             \n')
            elif next_line == "Mat_2":
                data[i] = (
                    f'MAT8           8{self.phase_2.E_1}{self.phase_2.E_2}{self.phase_2.nu_12}{self.phase_2.G_12}1.0     1.0             \n')
            if re.findall("HWCOLOR MAT                   7       9", line):  # Find pattern that starts with "pts_time:"
                next_line = "Mat_1"
            elif re.findall("HWCOLOR MAT                   8       8", line):
                next_line = "Mat_2"
            else:
                next_line = False
        with open(f'nastran_output/{start_file}_{self.test_nr}.bdf', mode='w') as file:
            file.writelines(data)

    def run_nastran(self):
        """
        run the generated .bdf files in nastran MSC
        Move generated .f06 file to \nastran_sol
        """
        # call os with start file x and self.test_nr
        p1 = subprocess.call(['C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.3\\bin\\nastranw.exe',
                             f'C:\\Users\\u086939\\PycharmProjects\\pythonProject\\nastran_output\\{self.start_file_x}_{self.test_nr}.bdf'])
        # call os with start file y and self.test_nr
        p2 = subprocess.call(['C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.3\\bin\\nastranw.exe',
                             f'C:\\Users\\u086939\\PycharmProjects\\pythonProject\\nastran_output\\{self.start_file_y}_{self.test_nr}.bdf'])

        # call os with start file xy and self.test_nr
        p3 = subprocess.call(['C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.3\\bin\\nastranw.exe',
                             f'C:\\Users\\u086939\\PycharmProjects\\pythonProject\\nastran_output\\{self.start_file_xy}_{self.test_nr}.bdf'])

    def move_f06_files(self):
        os.replace(f"C:\\Users\\u086939\\PycharmProjects\\pythonProject\\{self.start_file_x}_{self.test_nr}.f06", f"C:\\Users\\u086939\\PycharmProjects\\pythonProject\\nastran_sol\\{self.start_file_x}_{self.test_nr}.f06")
        os.replace(f"C:\\Users\\u086939\\PycharmProjects\\pythonProject\\{self.start_file_y}_{self.test_nr}.f06",
                   f"C:\\Users\\u086939\\PycharmProjects\\pythonProject\\nastran_sol\\{self.start_file_y}_{self.test_nr}.f06")
        os.replace(f"C:\\Users\\u086939\\PycharmProjects\\pythonProject\\{self.start_file_xy}_{self.test_nr}.f06",
                   f"C:\\Users\\u086939\\PycharmProjects\\pythonProject\\nastran_sol\\{self.start_file_xy}_{self.test_nr}.f06")

    def calc_stress(self, e_stress):
        tot_area = 255 * 255
        tot_stress = np.zeros((1,3))
        for e_s in e_stress:
            el_area = self.e_area[np.where(self.e_area[:, 0] == e_s[0]), 1]
            for i in range(3):
                tot_stress[0,i] += el_area * e_s[i + 1]
        avg_stress = tot_stress/tot_area
        return avg_stress

    def read_reac_force(self, start_file):
        with open(f'nastran_sol/{start_file}_{self.test_nr}.f06', 'r') as file:
            in_data = file.readlines()
        disp_flag = False
        out_data = []
        in_data_iter = iter(in_data)
        for line in in_data_iter:
            if line[0] == '1':
                disp_flag = False
            if re.findall(" \*\*\*", line):
                disp_flag = False
            if disp_flag:
                out_data.append(line)
            if re.findall(
                    "                               F O R C E S   O F   S I N G L E - P O I N T   C O N S T R A I N T",
                    line):
                disp_flag = True
                next(in_data_iter)
                next(in_data_iter)
        data = np.ndarray((342, 7))
        for i, s in enumerate(out_data):
            s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
            data[i] = [float(val) for val in s_out]
        return data

    def sort_forces(self, start_file):
        reac_data = self.read_reac_force(start_file)
        top_side = np.zeros((1, 2))
        bot_side = np.zeros((1, 2))
        left_side = np.zeros((1, 2))
        right_side = np.zeros((1, 2))
        for i, node in enumerate(reac_data[:, 0]):
            reac_x = reac_data[i, 1]
            reac_y = reac_data[i, 2]
            grid_value = self.grid_data[np.where(self.grid_data[:, 0] == node)]
            if grid_value[0, 1] == 0:
                if grid_value[0, 2] == 0:
                    bot_side[0, 1] += reac_y
                    left_side[0, 0] += reac_x
                    # corner point
                elif grid_value[0, 2] == 255:
                    left_side[0, 0] += reac_x
                    top_side[0, 1] += reac_y
                    # corner point
                else:
                    left_side[0, 0] += reac_x
                    left_side[0, 1] += reac_y
            elif grid_value[0, 1] == 255:
                if grid_value[0, 2] == 0:
                    right_side[0, 0] += reac_x
                    bot_side[0, 1] += reac_y
                    # corner point
                elif grid_value[0, 2] == 255:
                    right_side[0, 0] += reac_x
                    top_side[0, 1] += reac_y
                    # corner point
                else:
                    right_side[0, 0] += reac_x
                    right_side[0, 1] += reac_y
            elif grid_value[0, 2] == 255:
                top_side[0, 0] += reac_x
                top_side[0, 1] += reac_y
            elif grid_value[0, 2] == 0:
                bot_side[0, 0] += reac_x
                bot_side[0, 1] += reac_y
            else:
                print("grid_value not on boundary")
        stress_x = np.abs((left_side[0, 0] * -1 + right_side[0, 0]) / 2)/255
        stress_y = np.abs((bot_side[0, 1] * -1 + top_side[0, 1]) / 2)/255
        stress_xy = (np.abs((np.abs(bot_side[0, 0]) + np.abs(top_side[0, 0]) + np.abs(left_side[0, 1]) + np.abs(right_side[0, 1])) / 4))/255
        stress = np.zeros((1, 3))
        stress[0] = [stress_x, stress_y, stress_xy]
        return stress

    def ele_stress(self, start_file):
        with open(f'nastran_sol\\{start_file}_{self.test_nr}.f06', 'r') as file:
            in_data = file.readlines()
        disp_flag = False
        out_data = []
        in_data_iter = iter(in_data)
        for line in in_data_iter:
            if line[0] == '1':
                disp_flag = False
            if re.findall(" \*\*\*", line):
                disp_flag = False
            if disp_flag:
                out_data.append(line)
                next(in_data_iter)  # skip duplicate
            if re.findall(
                    "  ELEMENT      FIBER               STRESSES IN MATERIAL COORD SYSTEM            PRINCIPAL STRESSES",
                    line):
                disp_flag = True
                next(in_data_iter)
        data = np.ndarray((self.n_el, 4))
        for i, s in enumerate(out_data):
            s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
            nums = [float(val) for val in s_out]
            data[i] = [nums[1], nums[3], nums[4], nums[5]]
        return data

    def rotate_stress_field(self, stress_data, el_nodes):
        grid_data = self.grid_data
        new_stress_vec = np.zeros((len(el_nodes), 4))
        for i, el_stress in enumerate(stress_data):
            nodes = el_nodes[np.where(el_nodes[:, 0] == el_stress[0])]
            xy_node_1 = grid_data[np.where(grid_data[:, 0] == nodes[0, 1])]
            xy_node_2 = grid_data[np.where(grid_data[:, 0] == nodes[0, 2])]
            x_hat = np.zeros((1, 2))
            x_hat[0, 0] = xy_node_2[0, 1] - xy_node_1[0, 1]
            x_hat[0, 1] = xy_node_2[0, 2] - xy_node_1[0, 2]
            x_hat = x_hat / np.linalg.norm(x_hat)
            rot_mat = np.zeros((2, 2))
            rot_mat[0, 0] = x_hat[0, 0]
            rot_mat[0, 1] = -x_hat[0, 1]
            rot_mat[1, 1] = x_hat[0, 0]
            rot_mat[1, 0] = x_hat[0, 1]
            stress_mat = np.zeros((2, 2))
            stress_mat[0, 0] = el_stress[1]
            stress_mat[0, 1] = el_stress[3]
            stress_mat[1, 0] = el_stress[3]
            stress_mat[1, 1] = el_stress[2]
            new_stress_mat = np.matmul(np.matmul(rot_mat, stress_mat), np.transpose(rot_mat))
            new_stress_vec[i, :] = [nodes[0, 0], new_stress_mat[0, 0], new_stress_mat[1, 1], new_stress_mat[0, 1]]
        temp = np.max(new_stress_vec[:,1])
        temp2 = np.max(new_stress_vec[:, 2])
        return new_stress_vec

    def calc_comp_mat(self):
        D = sm.symarray("D", (3,3))
        eqs = []
        eqs.extend(np.matmul(D, self.stress_x[0, :]) - [self.strain[0,0], 0, 0])
        eqs.extend(np.matmul(D, self.stress_y[0, :]) - [0, self.strain[0,1], 0])
        eqs.extend(np.matmul(D, self.stress_xy[0, :]) - [0, 0, self.strain[0,2]])
        sol = solve(eqs)
        vals = [val for val in sol.values()]
        D = np.zeros((3, 3))
        D[0,0] = vals[0]
        D[0, 1] = vals[1]
        D[0, 2] = vals[2]
        D[1, 0] = vals[3]
        D[1, 1] = vals[4]
        D[1, 2] = vals[5]
        D[2, 0] = vals[6]
        D[2, 1] = vals[7]
        D[2, 2] = vals[8]

        D_2 = np.zeros((3, 3))
        D_2[0,0] = self.strain[0,0]/(self.stress_x[0,0]*(1-(self.stress_x[0,1]*self.stress_y[0,0])/(self.stress_x[0,0]*self.stress_y[0,1])))
        D_2[0,1] = -D_2[0,0]*self.stress_y[0,0]/self.stress_y[0,1]
        D_2[1,0] = D_2[0,1]
        D_2[1,1] = self.strain[0,1]/self.stress_y[0,1] - D[0,1]*self.stress_y[0,0]/self.stress_y[0,1]
        D_2[2,2] = self.strain[0,2]/self.stress_xy[0,2]
        return D

    def elast_bounds(self):
        """
        calculate lower Reuss value and upper Voigt value for the representative elasticity parameters
        """

        D_voigt = np.zeros((3, 3))
        D_voigt[0, 0] = 1 / (self.vol_frac / self.phase_1.D[0, 0] + (1 - self.vol_frac) / self.phase_2.D[0, 0])
        D_voigt[0, 1] = 1 / (self.vol_frac / self.phase_1.D[0, 1] + (1 - self.vol_frac) / self.phase_2.D[0, 1])
        D_voigt[1, 0] = D_voigt[0,1]
        D_voigt[1, 1] = 1 / (self.vol_frac / self.phase_1.D[1, 1] + (1 - self.vol_frac) / self.phase_2.D[1, 1])
        D_voigt[2, 2] = 1 / (self.vol_frac / self.phase_1.D[2, 2] + (1 - self.vol_frac) / self.phase_2.D[2, 2])

        C_voigt = np.zeros((1, 4))
        C_voigt[0, 0] = 1 / D_voigt[0, 0]
        C_voigt[0, 1] = 1 / D_voigt[1, 1]
        C_voigt[0, 3] = 1 / (2*D_voigt[2, 2])
        C_voigt[0, 2] = -C_voigt[0, 0] * D_voigt[1, 0]

        D_reuss = np.zeros((3,3))
        D_reuss[0, 0] = self.vol_frac * self.phase_1.D[0, 0] + (1 - self.vol_frac) * self.phase_2.D[0, 0]
        D_reuss[0, 1] = self.vol_frac * self.phase_1.D[0, 1] + (1 - self.vol_frac) * self.phase_2.D[0, 1]
        D_reuss[1, 0] = D_reuss[0, 1]
        D_reuss[1, 1] = self.vol_frac * self.phase_1.D[1, 1] + (1 - self.vol_frac) * self.phase_2.D[1, 1]
        D_reuss[2, 2] = self.vol_frac*self.phase_1.D[2, 2] + (1-self.vol_frac)*self.phase_2.D[2, 2]

        C_reuss = np.zeros((1, 4))
        C_reuss[0, 0] = 1 / D_reuss[0, 0]
        C_reuss[0, 1] = 1 / D_reuss[1, 1]
        C_reuss[0, 3] = 1 / (2*D_reuss[2, 2])
        C_reuss[0, 2] = -C_reuss[0, 0] * D_reuss[1, 0]

        self.D_voigt = D_voigt
        self.C_voigt = C_voigt
        self.D_reuss = D_reuss
        self.C_reuss = C_reuss

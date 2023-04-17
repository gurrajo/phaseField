import re
import numpy as np
import sympy as sm


class PhaseMat:
    """
    Object representing a phase of a material
    """
    def __init__(self, E_1, E_2, nu_12, G_12):
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

        self.C = np.ndarray((1, 4))
        self.C[0] = [E_1, E_2, nu_12, G_12]


class Micro:
    def __init__(self, phase_1, phase_2, test_nr, grid_data, e_area):
        self.vol_frac = 0.5
        self.grid_data = grid_data
        self.e_area = e_area
        self.n_el = len(e_area)  # number of elements
        # 3 start files, one for each load case
        self.start_file_x = "2_orth_stress_out"
        self.start_file_y = "2_orth_stress_out"
        self.start_file_xy = "shear_orig"
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.test_nr = test_nr
        self.change_material(self.start_file_x)
        self.change_material(self.start_file_y)
        self.change_material(self.start_file_xy)
        self.run_nastran()  # generate f06 file for each load case
        self.stress_x = self.calc_stress(self.start_file_x)
        self.stress_y = self.calc_stress(self.start_file_y)
        self.stress_xy = self.calc_stress(self.start_file_xy)
        self.C = self.calc_elast_mat()

    def change_material(self, start_file):
        # for orthotropic materials
        with open(f'nastran_input/{start_file}.bdf', 'r') as file:
            # read a list of lines into data
            data = file.readlines()
        next_line = False
        for i, line in enumerate(data):
            if next_line == "Mat_1":
                data[i] = (f'MAT8           9{self.phase_1.E_1}{self.phase_1.E_2}{self.phase_1.nu_12}{self.phase_1.G_12}1.0     1.0             \n')
            elif next_line == "Mat_2":
                data[i] = (f'MAT8          10{self.phase_2.E_1}{self.phase_2.E_2}{self.phase_2.nu_12}{self.phase_2.G_12}1.0     1.0             \n')
            if re.findall("HWCOLOR MAT                   9       4", line):  # Find pattern that starts with "pts_time:"
                next_line = "Mat_1"
            elif re.findall("HWCOLOR MAT                  10       5", line):
                next_line = "Mat_2"
            else:
                next_line = False
        with open(f'nastran_output/{start_file}_{self.test_nr}.bdf', mode='w') as file:
            file.writelines(data)

    def run_nastran(self):
        print(1)
        # call os with start file x and self.test_nr
        # call os with start file y and self.test_nr
        # call os with start file xy and self.test_nr

    def calc_stress(self, start_file):
        e_stress = self.ele_stress(start_file)
        tot_area = 255 * 255
        tot_stress = [0, 0, 0]
        for e_s in e_stress:
            el_area = self.e_area[np.where(self.e_area[:, 0] == e_s[0]), 1]
            for i in range(3):
                tot_stress[i] += el_area * e_s[i + 1]
        avg_stress = [s / tot_area for s in tot_stress]
        return avg_stress

    def ele_stress(self, start_file):
        with open(f'nastran_output/{start_file}_{self.test_nr}.f06', 'r') as file:
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
                    "  ELEMENT      FIBER               STRESSES IN ELEMENT COORD SYSTEM             PRINCIPAL STRESSES",
                    line):
                disp_flag = True
                next(in_data_iter)
        data = np.ndarray((self.n_el, 4))
        for i, s in enumerate(out_data):
            s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
            nums = [float(val) for val in s_out]
            data[i] = [nums[1], nums[3], nums[4], nums[5]]
        return data

    def calc_elast_mat(self):
        # use symbolic math
        # TODO
        self.stress_x[0] = E_1/(1-nu_12**2)
        self.stress_x[1] = nu_12*E_1/(1-nu_12**2)
        E_2 = self.stress_y[0] * 0.02
        nu_12 = self.stress_x[1] * 0.02  # fix
        G_12 = self.stress_xy[0] * 0.02
        C = [E_1, E_2, nu_12, G_12]
        return C

    def elast_bounds(self):
        """
        calculate lower Reuss value and upper Voigt value for the representative elasticity parameters
        """
        C_voigt = self.vol_frac*self.phase_1.C + (1-self.vol_frac)*self.phase_2.C
        C_reuss = np.linalg.inv(self.vol_frac/self.phase_1.C + (1-self.vol_frac)/self.phase_2.C)

        return C_voigt, C_reuss
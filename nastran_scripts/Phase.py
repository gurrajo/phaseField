import re
import numpy as np


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


class Micro:
    def __init__(self, phase_1, phase_2, test_nr, grid_data):
        self.grid_data = grid_data
        self.start_file = "2_orth_stress_out"
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.test_nr = test_nr
        self.change_material()
        #self.linear_displacement(grid_data, "x")
        #self.linear_displacement(grid_data, "y")

    def change_material(self):
        # for orthotropic materials
        with open(f'nastran_input/{self.start_file}.bdf', 'r') as file:
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
        with open(f'nastran_output/material{self.test_nr}.bdf', mode='w') as file:
            file.writelines(data)

    def linear_displacement(self, grid_data, direc):
        """
        writes a new bdf file with linearly increasing load in direc direction
        :param grid_data: node, x, and y values
        :param direc: load direction
        :return:
        """
        with open(f'nastran_output/material{self.test_nr}.bdf', 'r') as file:
            in_data = file.readlines()
        for i, line in enumerate(in_data):
            if re.findall("SPCD           ", line):
                node = int(line[16:24])
                ind = np.where(grid_data[:, 0] == node)
                x = grid_data[ind, 1]
                y = grid_data[ind, 2]
                if direc == "x":
                    direc_val = 1
                    spcd = 5.1 * x[0, 0] / 255
                elif direc == "y":
                    direc_val = 2
                    spcd = 5.1 * y[0, 0] / 255
                else:
                    print("no load direction stated")
                    return
                spcd_string = "%.5f" % spcd
                spcd_string = list(spcd_string)
                spcd_string.insert(0, " ")
                spcd_string = "".join(spcd_string)
                node_string = list(str(node))
                while len(node_string) < 8:
                    node_string.insert(0, " ")
                node_string = "".join(node_string)
                in_data[i] = f"SPCD           2{node_string}       {direc_val}{spcd_string}\n"
        with open(f'nastran_output/linear{direc}_{self.test_nr}.bdf', mode='w') as file:
            file.writelines(in_data)

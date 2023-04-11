import numpy as np
import re
import Phase


def change_material(phase_1, phase_2, test_nr):
    with open(f'nastran_input/micro_struc_hm_stress_out.bdf', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    next_line = False
    for i, line in enumerate(data):
        if i == 15438:
            print(line)
        if next_line == "Mat_1":
            data[i] = (f'MAT1           {phase_1.E}           {phase_1.nu} \n')
            print(1)
        elif next_line == "Mat_2":
            data[i] = (f'MAT1           {phase_2.E}           {phase_2.nu} \n')
            print(2)

        if re.findall("HWCOLOR MAT                   7       5", line):  # Find pattern that starts with "pts_time:"
            next_line = "Mat_1"
        elif re.findall("HWCOLOR MAT                   8       5", line):
            next_line = "Mat_2"
        else:
            next_line = False
    with open(f'nastran_scripts/nastran_input/micro_struc_hm_tension{test_nr}.bdf', mode='w') as file:
        file.writelines(data)


def read_grid_data(filename):
    """
    :param filename:
    :return: point ID, x, y
    """
    with open(f'nastran_input/{file_name}.bdf', 'r') as file:
        in_data = file.readlines()
    disp_flag = False
    out_data = np.ndarray((7247, 3))
    in_data_iter = iter(in_data)
    i = 0
    for line in in_data_iter:
        if re.findall("\$\$", line):
            disp_flag = False
        if disp_flag:

            gp = float(line[4:16])
            temp1 = line[24:24+8]
            if re.findall("\\.\d+-\d+", temp1):
                temp1 = 0
            s1 = float(temp1)

            temp2 = line[24 + 8:24 + 16]
            if re.findall("\\.\d+-\d+", temp2):
                temp2 = 0
            s2 = float(temp2)
            out_data[i, 0] = gp
            out_data[i, 1] = s1
            out_data[i, 2] = s2
            i += 1

        if re.findall("\$\$  GRID Data", line):
            disp_flag = True
            next(in_data_iter)
    return out_data


def linear_displacement(file_name, grid_data, direc):
    """

    :param file_name: name of file in nastran_input folder to edit
    :param grid_data: node, x, and y values
    :return: writes a new bdf file with linearly increasing load
    """
    with open(f'nastran_input/{file_name}.bdf', 'r') as file:
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
    with open(f'nastran_input/{file_name}_linear_y.bdf', mode='w') as file:
        file.writelines(in_data)


def change_load():
    with open(f'nastran_input/micro_struc_hm_stress_out.bdf', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    next_line = False
    for i, line in enumerate(data):
        if next_line:
            data[i] = ("+              1     1.0     0.0     0.0\n")

        if re.findall("PLOAD4         2", line):  # Find pattern
            next_line = True
        else:
            next_line = False
    with open(f'nastran_scripts/nastran_input/micro_struc_hm_tension.bdf', mode='w') as file:
        file.writelines(data)


def read_reac_force(file_name):
    with open(f'nastran_input/{file_name}.f06', 'r') as file:
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
        if re.findall("                               F O R C E S   O F   S I N G L E - P O I N T   C O N S T R A I N T", line):
            disp_flag = True
            next(in_data_iter)
            next(in_data_iter)
    data = np.ndarray((342, 7))
    for i, s in enumerate(out_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        data[i] = [float(val) for val in s_out]
    return data


def normal_force(reac_data, grid_data):
    x_side = 0
    y_side = 0
    x_tick = 0
    y_tick = 0
    for i, node in enumerate(reac_data[:, 0]):
        reac_x = reac_data[i, 1]
        reac_y = reac_data[i, 2]
        if grid_data[i, 1] == 0:
            if grid_data[i, 2] == 0:
                x_side += -1*reac_x
                y_side += -1*reac_y
                x_tick += 1
                y_tick += 1
                # corner point
            elif grid_data[i, 2] == 255:
                x_side += -1*reac_x
                y_side += reac_y
                x_tick += 1
                y_tick += 1
                # corner point
            else:
                x_side += -1*reac_x
                x_tick += 1
        elif grid_data[i, 1] == 255:
            if grid_data[i, 2] == 0:
                x_side += reac_x
                y_side += -1*reac_y
                x_tick += 1
                y_tick += 1
                # corner point
            elif grid_data[i, 2] == 255:
                x_side += reac_x
                y_side += reac_y
                x_tick += 1
                y_tick += 1
                # corner point
            else:
                y_side += -1*reac_y
                y_tick += 1
        elif grid_data[i, 2] == 255:
            y_side += reac_y
            y_tick += 1
        elif grid_data[i, 2] == 0:
            x_side += reac_x
            x_tick += 1
    return x_side, y_side, x_tick, y_tick


def read_disp(file_name):
    with open(f'nastran_scripts/nastran_input/{file_name}.f06', 'r') as file:
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
        if re.findall("                                             D I S P L A C E M E N T   V E C T O R", line):
            disp_flag = True
            next(in_data_iter)
            next(in_data_iter)
    data = np.ndarray((86, 7))
    for i, s in enumerate(out_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        data[i] = [float(val) for val in s_out]
    return data


E_1 = 100.001
nu_1 = 0.3501

E_2 = 200.001
nu_2 = 0.3901
file_name = "spcd_x_4_linear_y"
grid_data = read_grid_data(file_name)
#linear_displacement(file_name, data, "y")
reac_data = read_reac_force(file_name)
grid_bound = np.zeros((len(reac_data), 3))
i = 0
# determine grid nodes
for node in grid_data:
    if node[0] in reac_data[:, 0]:
        grid_bound[i] = node
        i += 1
(x_reac, y_reac, x_tick, y_tick) = normal_force(reac_data, grid_bound)
E_22 = [x_reac, y_reac, 0]
#output_files = os.system(f'C:\Program^ Files\MSC.Software\MSC_Nastran\\2021.3\\bin\\nastranw.exe nastran_input\\{file_name}.bdf')

#D_22 = calculate_compliance(file_name)


phase_1 = Phase.PhaseMat(E_1, nu_1)
phase_2 = Phase.PhaseMat(E_2, nu_2)
#change_material(phase_1, phase_2, 1)
#change_load()

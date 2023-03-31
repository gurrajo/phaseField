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


def grid_data(filename):
    """
    :param filename:
    :return: point ID, x, y
    """
    with open(f'nastran_scripts/nastran_input/{file_name}.bdf', 'r') as file:
        in_data = file.readlines()
    disp_flag = False
    out_data = np.ndarray((7560, 3))
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


def calculate_compliance(file_name):
    L = 255  # initial length
    p = -1/L  # pressure

    disp_data = read_disp(file_name)
    y_disp = np.mean(disp_data[0:-1, 2])  # mean might not be best
    eps = (L-y_disp)/L
    D_22 = eps/p
    return D_22

E_1 = 100.001
nu_1 = 0.3501

E_2 = 200.001
nu_2 = 0.3901
file_name = "micro_struc_hm_current_test_new_load"
data = grid_data(file_name)
#output_files = os.system(f'C:\Program^ Files\MSC.Software\MSC_Nastran\\2021.3\\bin\\nastranw.exe nastran_input\\{file_name}.bdf')

#D_22 = calculate_compliance(file_name)


phase_1 = Phase.PhaseMat(E_1, nu_1)
phase_2 = Phase.PhaseMat(E_2, nu_2)
#change_material(phase_1, phase_2, 1)
#change_load()


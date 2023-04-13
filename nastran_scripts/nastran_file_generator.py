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
    #
    with open(f'nastran_scripts/nastran_input/{file_name}.bdf', 'r') as file:
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


def linear_shear_displacement(file_name, grid_data):
    """
    creates linear pure shear strain condition
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
            spcd_x = 5.1 * y[0, 0] / 255
            spcd_y = 5.1 * x[0, 0] / 255
            spcd_x_string = "%.5f" % spcd_x
            spcd_x_string = list(spcd_x_string)
            spcd_x_string.insert(0, " ")
            spcd_x_string = "".join(spcd_x_string)
            node_string = list(str(node))
            while len(node_string) < 8:
                node_string.insert(0, " ")
            node_string = "".join(node_string)
            in_data[i] = f"SPCD           2{node_string}       1{spcd_x_string}\n"
    with open(f'nastran_input/{file_name}_linear_shear.bdf', mode='w') as file:
        file.writelines(in_data)


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
    with open(f'nastran_input/{file_name}_linear_{direc}.bdf', mode='w') as file:
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


def quad_element_nodes(file_name):
    """
    create element node matrix, each row contains element index and 4 corresponding node index
    :param file_name:
    :return:
    """
    #
    with open(f'nastran_scripts/nastran_input/{file_name}.bdf', 'r') as file:
        in_data = file.readlines()
    data_flag = False
    out_data = []
    in_data_iter = iter(in_data)
    for line in in_data_iter:
        if re.findall("\$", line):
            data_flag = False
        if data_flag:
            out_data.append(line)
        if re.findall("\$\$  CQUAD4 Elements", line):
            data_flag = True
            next(in_data_iter)
    e_node = np.ndarray((len(out_data), 5))
    for i, s in enumerate(out_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        data = [float(val) for val in s_out]
        e_node[i] = [data[1], data[3], data[4], data[5], data[6]]
    return e_node


def tri_element_nodes(file_name):
    """
    create element node matrix, each row contains element index and 3 corresponding node index
    :param file_name:
    :return:
    """
    with open(f'nastran_scripts/nastran_input/{file_name}.bdf', 'r') as file:
        in_data = file.readlines()
    data_flag = False
    out_data = []
    in_data_iter = iter(in_data)
    for line in in_data_iter:
        if re.findall("\$\$", line):
            data_flag = False
        if data_flag:
            out_data.append(line)
        if re.findall("\$\$  CTRIA3 Data", line):
            data_flag = True
            next(in_data_iter)
    e_node = np.ndarray((len(out_data), 4))
    for i, s in enumerate(out_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        data = [float(val) for val in s_out]
        e_node[i] = [data[1], data[3], data[4], data[5]]
    return e_node


def tri_element_area(grid_data, e_node):
    """
    Calcualte area of triangular elements
    :param grid_data: node x, and y data
    :param e_node: matrix with, element index, and 3 node index
    :return: e_area: element index, element area
    """
    e_area = np.ndarray((len(e_node), 2))
    for i, ele in enumerate(e_node):
        node_1 = np.where(grid_data[:, 0] == ele[1])
        node_2 = np.where(grid_data[:, 0] == ele[2])
        node_3 = np.where(grid_data[:, 0] == ele[3])
        M = np.ndarray((3,3))
        M[0] = [grid_data[node_1, 1], grid_data[node_1, 2], 1]
        M[1] = [grid_data[node_2, 1], grid_data[node_2, 2], 1]
        M[2] = [grid_data[node_3, 1], grid_data[node_3, 2], 1]
        area = 0.5*np.linalg.det(M)
        e_area[i] = [ele[0], area]
    return e_area


def quad_element_area(grid_data, e_node):
    """
    Calcualte area of triangular elements
    :param grid_data: node x, and y data
    :param e_node: matrix with, element index, and 3 node index
    :return: e_area: element index, element area
    """
    e_area = np.ndarray((len(e_node), 2))
    for i, ele in enumerate(e_node):
        node_1 = np.where(grid_data[:, 0] == ele[1])
        node_2 = np.where(grid_data[:, 0] == ele[2])
        node_3 = np.where(grid_data[:, 0] == ele[3])
        node_4 = np.where(grid_data[:, 0] == ele[4])

        M_1 = np.ndarray((2, 2))
        M_2 = np.ndarray((2, 2))
        M_3 = np.ndarray((2, 2))
        M_4 = np.ndarray((2, 2))

        M_1[0] = [grid_data[node_1, 1], grid_data[node_2, 1]]
        M_1[1] = [grid_data[node_1, 2], grid_data[node_2, 2]]

        M_2[0] = [grid_data[node_2, 1], grid_data[node_3, 1]]
        M_2[1] = [grid_data[node_2, 2], grid_data[node_3, 2]]

        M_3[0] = [grid_data[node_3, 1], grid_data[node_4, 1]]
        M_3[1] = [grid_data[node_3, 2], grid_data[node_4, 2]]

        M_4[0] = [grid_data[node_4, 1], grid_data[node_1, 1]]
        M_4[1] = [grid_data[node_4, 2], grid_data[node_1, 2]]
        area = 0.5*(np.linalg.det(M_1) + np.linalg.det(M_2) + np.linalg.det(M_3) + np.linalg.det(M_4))
        e_area[i] = [ele[0], area]
    return e_area

E_1 = 100.001
nu_1 = 0.3501

E_2 = 200.001
nu_2 = 0.3901
file_name = "micro_struc_hm_current_test_new_load"

grid_data = read_grid_data(file_name)
e_node_tri = tri_element_nodes(file_name)
e_area_tri = tri_element_area(grid_data, e_node_tri)
e_node_quad = quad_element_nodes(file_name)
e_area_quad = quad_element_area(grid_data, e_node_quad)

#linear_displacement(file_name, data, "y")
#reac_data = read_reac_force(file_name)
#grid_bound = np.zeros((len(reac_data), 3))
#i = 0
# determine grid nodes
#for node in grid_data:
#    if node[0] in reac_data[:, 0]:
#        grid_bound[i] = node
#        i += 1
#(x_reac, y_reac, x_tick, y_tick) = normal_force(reac_data, grid_bound)
#E_22 = [x_reac, y_reac, 0]
#output_files = os.system(f'C:\Program^ Files\MSC.Software\MSC_Nastran\\2021.3\\bin\\nastranw.exe nastran_input\\{file_name}.bdf')

#D_22 = calculate_compliance(file_name)

#change_material(phase_1, phase_2, 1)
#change_load()

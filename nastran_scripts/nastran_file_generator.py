import numpy as np
import re
import Phase


def change_material(phase_1, phase_2, test_nr):
    with open(f'nastran_input/micro_struc_hm_stress_out.bdf', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    next_line = False
    for i, line in enumerate(data):
        if next_line == "Mat_1":
            data[i] = (f'MAT8           9{phase_1.E_1}50.0    0.35    40.0    1.0     1.0\n')
            print(1)
        elif next_line == "Mat_2":
            data[i] = (f'MAT1           {phase_2.E}           {phase_2.nu} \n')
            print(2)

        if re.findall("\$HWCOLOR MAT                   9       4", line):  # Find pattern that starts with "pts_time:"
            next_line = "Mat_1"
        elif re.findall("\$HWCOLOR MAT                   10       5", line):
            next_line = "Mat_2"
        else:
            next_line = False
    with open(f'nastran_scripts/nastran_input/micro_struc_hm_tension{test_nr}.bdf', mode='w') as file:
        file.writelines(data)


def read_el_stress(file_name, n_el):
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
            next(in_data_iter)  # skip duplicate
        if re.findall("  ELEMENT      FIBER               STRESSES IN ELEMENT COORD SYSTEM             PRINCIPAL STRESSES", line):
            disp_flag = True
            next(in_data_iter)
    data = np.ndarray((n_el, 4))
    for i, s in enumerate(out_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        nums = [float(val) for val in s_out]
        data[i] = [nums[1], nums[3], nums[4], nums[5]]
    return data


def read_el_strain(file_name, n_el):
    with open(f'nastran_output/{file_name}.f06', 'r') as file:
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
        if re.findall("  ELEMENT      STRAIN               STRAINS IN ELEMENT COORD SYSTEM             PRINCIPAL  STRAINS", line):
            disp_flag = True
            next(in_data_iter)
    data = np.ndarray((n_el, 4))
    for i, s in enumerate(out_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        nums = [float(val) for val in s_out]
        data[i] = [nums[1], nums[3], nums[4], nums[5]]
    return data


def read_grid_data(file_name):
    """
    :param filename:
    :return: point ID, x, y
    """
    #
    with open(f'nastran_input/{file_name}.bdf', 'r') as file:
        in_data = file.readlines()
    disp_flag = False
    out_data = np.ndarray((16223, 3))
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


def linear_shear_displacement(file_name, grid_data, max_disp):
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
            direc = int(line[24:32])
            node = int(line[16:24])
            ind = np.where(grid_data[:, 0] == node)
            if direc == 1:
                val = grid_data[ind, 2]

            elif direc == 2:
                val = grid_data[ind, 1]
            else:
                print("fail")
            spcd = max_disp * val[0, 0] / 255000
            spcd_string = "%.5f" % spcd
            spcd_string = list(spcd_string)
            while len(spcd_string) > 7:
                spcd_string.pop(-1)
            spcd_string.insert(0, " ")
            spcd_string = "".join(spcd_string)
            node_string = list(str(node))
            while len(node_string) < 8:
                node_string.insert(0, " ")
            node_string = "".join(node_string)
            in_data[i] = f"SPCD           2{node_string}       {direc}{spcd_string}\n"
    with open(f'nastran_input/{file_name}_{max_disp}.bdf', mode='w') as file:
        file.writelines(in_data)


def linear_displacement(file_name, grid_data, direc, max_disp):
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
                spcd = float(max_disp * x[0, 0] / 255000)
            elif direc == "y":
                direc_val = 2
                spcd = float(max_disp * y[0, 0] / 255000)
            else:
                print("no load direction stated")
                return
            spcd_string = "%.7f" % spcd
            spcd_string = list(spcd_string)
            while len(spcd_string) > 7:
                spcd_string.pop(-1)
            spcd_string.insert(0, " ")
            spcd_string = "".join(spcd_string)
            node_string = list(str(node))
            while len(node_string) < 8:
                node_string.insert(0, " ")
            node_string = "".join(node_string)
            in_data[i] = f"SPCD           2{node_string}       {direc_val}{spcd_string}\n"
    with open(f'nastran_input/{file_name}_{max_disp}.bdf', mode='w') as file:
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


def sort_forces(reac_data, grid_data):
    top_side = np.zeros((1, 2))
    bot_side = np.zeros((1, 2))
    left_side = np.zeros((1, 2))
    right_side = np.zeros((1, 2))
    for i, node in enumerate(reac_data[:, 0]):
        reac_x = reac_data[i, 1]
        reac_y = reac_data[i, 2]
        grid_value = grid_data[np.where(grid_data[:, 0] == node)]
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
    return left_side, right_side, top_side, bot_side


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
    with open(f'nastran_input/{file_name}.bdf', 'r') as file:
        in_data = file.readlines()
    data_flag = False
    out_data = []
    in_data_iter = iter(in_data)
    for line in in_data_iter:
        if re.findall("\$\$", line):
            data_flag = False
        if data_flag:
            if re.findall("\$", line):
                continue
            out_data.append(line)
        if re.findall("\$\$  CQUAD4 Elements", line):
            data_flag = True
            next(in_data_iter)
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
    with open(f'nastran_input/{file_name}.bdf', 'r') as file:
        in_data = file.readlines()
    data_flag = False
    out_data = []
    in_data_iter = iter(in_data)
    for line in in_data_iter:
        if re.findall("\$\$", line):
            data_flag = False
        if data_flag:
            if re.findall("\$", line):
                continue
            out_data.append(line)
        if re.findall("\$\$  CTRIA3 Data", line):
            data_flag = True
            next(in_data_iter)
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


def vol_avg_stress(e_area, e_stress):
    tot_area = 255 * 255
    tot_stress = [0,0,0]
    for e_s in e_stress:
        el_area = e_area[np.where(e_area[:, 0] == e_s[0]), 1]
        for i in range(3):
            tot_stress[i] += el_area*e_s[i+1]
    avg_stress = [s/tot_area for s in tot_stress]
    return avg_stress


def vol_avg_strain(e_area, e_strain):
    tot_area = 255 * 255
    tot_strain = [0, 0, 0]
    for e_s in e_strain:
        el_area = e_area[np.where(e_area[:, 0] == e_s[0]), 1]
        for i in range(3):
            tot_strain[i] += el_area*e_s[i+1]
    avg_strain = [s/tot_area for s in tot_strain]
    return avg_strain


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


def rotate_stress_field(stress_data, grid_data, el_nodes):
    new_stress_vec = np.zeros((len(el_nodes), 4))
    for i, el_stress in enumerate(stress_data):
        nodes = el_nodes[np.where(el_nodes[:, 0] == el_stress[0])]
        temp = np.where(grid_data[:, 0] == nodes[0, 1])
        xy_node_1 = grid_data[np.where(grid_data[:, 0] == nodes[0, 1])]
        xy_node_2 = grid_data[np.where(grid_data[:, 0] == nodes[0, 2])]
        x_hat = np.zeros((1, 2))
        x_hat[0, 0] = xy_node_2[0,0] - xy_node_1[0,0]
        x_hat[0, 1] = xy_node_2[0,1] - xy_node_1[0,1]
        x_hat = x_hat/np.linalg.norm(x_hat)
        x = np.ndarray((1,2))
        x[0] = [1,0]
        rot_mat = np.zeros((2,2))
        temp = rot_mat[0,0]
        rot_mat[0,0] = np.vdot(x,x_hat)
        rot_mat[0,1] = -np.linalg.norm(np.cross(x,x_hat))
        rot_mat[1,1] = np.vdot(x,x_hat)
        rot_mat[1,0] = np.linalg.norm(np.cross(x,x_hat))
        stress_mat = np.zeros((2,2))
        stress_mat[0,0] = el_stress[1]
        stress_mat[0,1] = el_stress[3]
        stress_mat[1,0] = el_stress[3]
        stress_mat[1,1] = el_stress[2]
        new_stress_mat = np.matmul(np.matmul(rot_mat, stress_mat), np.transpose(rot_mat))
        new_stress_vec[i,:] = [nodes[0,0], new_stress_mat[0,0], new_stress_mat[1,1], new_stress_mat[0,1]]
    return new_stress_vec
# E_1 = 100.001
# nu_1 = 0.3501
#
# E_2 = 200.001
# nu_2 = 0.3901
# file_name = "micro_struc_hm_current_test_new_load"
#
# grid_data = read_grid_data(file_name)
# e_node_tri = tri_element_nodes(file_name)
# e_area_tri = tri_element_area(grid_data, e_node_tri)
# e_node_quad = quad_element_nodes(file_name)
# e_area_quad = quad_element_area(grid_data, e_node_quad)

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

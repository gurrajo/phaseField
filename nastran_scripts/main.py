import numpy as np
import time
import Phase
import nastran_file_generator as nas_gen
import phase_comp_generator as ph_gen
import re


def write_dataset(arr):
    """
    Writes a dataset file. test_nr, phase_1, Phase_2, micro
    :param arr: micro obj. array.
    """
    data = []
    for i, micro in enumerate(arr):
        data.append(f"{str(micro.test_nr)} {str(micro.phase_1)}{str(micro.phase_2)}{str(micro)}\n")
    with open(f'data/data_set_5.txt', mode='w') as file:
        file.writelines(data)

def string_func(D):
    string = [" "]
    for i in range(3):
        for j in range(3):
            string.append(str(D[i, j]))
            string.append(" ")
    return "".join(string)


def write_phases(D_1,D_2):
    data = []
    for i, D1 in enumerate(D_1):
        data.append(f"{string_func(D1)}{string_func(D_2[i])}\n")
    with open(f'data/phase_data.txt', mode='w') as file:
        file.writelines(data)

def extract_phases(phase_file):
    with open(f'data/{phase_file}.txt', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    out_data = []
    for i, line in enumerate(data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
        out_data.append([float(val) for val in s_out])
    return out_data


start_time = time.time()
orig_file_name = "morph_5_x"
# generate material space with latin hypercube
# (D_1, D_2) = ph_gen.phaseinator()
# write_phases(D_1,D_2)
phase_data = extract_phases("phase_data")
# calculate pre-proc data from bdf original file
grid_data = nas_gen.read_grid_data(orig_file_name)  # node position data
e_node_tri = nas_gen.tri_element_nodes(orig_file_name)
e_area_tri = nas_gen.tri_element_area(grid_data, e_node_tri)  # area for each triangular element
e_node_quad = nas_gen.quad_element_nodes(orig_file_name)
e_area_quad = nas_gen.quad_element_area(grid_data, e_node_quad)  # area for each quadrilateral element
e_area = np.append(e_area_quad, e_area_tri, 0)
nel = len(e_area)
e_node_2 = np.append(e_node_quad[:, 0:3], e_node_tri[:, 0:3], 0)
# from a material sample (2 phases) generate a Micro object

# ---- only required once----- #
#nas_gen.linear_displacement("morph_2_x", grid_data, "x", 10200.0)
#nas_gen.linear_displacement("morph_2_y", grid_data, "y", 10200.0)
#nas_gen.linear_shear_displacement("morph_2_xy", grid_data,  10200.0)


test_nr = 0
tests = 100
micro_arr = []
for i in range(tests):
    k = 0
    D1 = np.zeros((3,3))
    D2 = np.zeros((3,3))
    for j in range(3):
        for m in range(3):
            D1[j,m] = phase_data[i][k]
            D2[j,m] = phase_data[i][k+9]
            k += 1
    phase_1 = Phase.PhaseMat(D1)
    phase_2 = Phase.PhaseMat(D2)
    micro_arr.append(Phase.Micro(phase_1, phase_2, test_nr, grid_data, e_area))
    micro_arr[i].elast_bounds()
    test_nr += 1
    time.sleep(10)
time.sleep(25)

for i, micro in enumerate(micro_arr):
    micro.move_f06_files()

for micro in micro_arr:
    micro.calc_stresses(e_node_2)
    micro.check_elast_bounds()
    micro.check_symm()
    if micro.bound_check:
        print("Passed Voigt-Reuss check")
    else:
        print("Failed")
    if micro.sym_check:
        print("Passed symmetry check")
    else:
        print("Failed")


# # Micro object changes material data and generates new bdf files
# # Micro object runs bdf files in Nastran
# # Micro object interperates f06 file, gets (area, and element stresses)
write_dataset(micro_arr)
stop_time = time.time()
run_time = stop_time - start_time  # seconds
print(run_time)

print(1)

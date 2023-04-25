import numpy as np
import time
import Phase
import nastran_file_generator as nas_gen
import phase_comp_generator as ph_gen
import re


def write_dataset(arr):
    """
    Writes a
    :param arr:
    :return:
    """
    data = []
    for i, micro in enumerate(arr):
        data.append(f"{str(micro.test_nr)} {str(micro.phase_1)}{str(micro.phase_2)}{str(micro)}\n")
    with open(f'data_set.txt', mode='w') as file:
        file.writelines(data)


def read_dataset(file_name):
    with open(f'{file_name}.txt', 'r') as file:
        in_data = file.readlines()
    data = np.ndarray((len(in_data), 28))
    for i, s in enumerate(in_data):
        s_out = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
        nums = [float(val) for val in s_out]
        data[i] = nums
    return data


start_time = time.time()
orig_file_name = "new_orig_x"
# generate material space with latin hypercube
(D_1, D_2) = ph_gen.phaseinator()

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
# nas_gen.linear_displacement("new_orig_x", grid_data, "x", 10.2)
# nas_gen.linear_displacement("new_orig_y", grid_data, "y", 10.2)
# nas_gen.linear_shear_displacement("new_orig_xy", grid_data,  10.2)


test_nr = 0
tests = 20
micro_arr = []
for i in range(tests):
    D1 = D_1[i]
    D2 = D_2[i]
    phase_1 = Phase.PhaseMat(D1)
    phase_2 = Phase.PhaseMat(D2)
    micro_arr.append(Phase.Micro(phase_1, phase_2, test_nr, grid_data, e_area))
    micro_arr[i].elast_bounds()
    test_nr += 1
    time.sleep(7)
time.sleep(25)

for i in range(tests):
    micro_arr[i].move_f06_files()

for i in range(tests):
    micro_arr[i].calc_stresses(e_node_2)
    micro_arr[i].check_elast_bounds()
    if micro_arr[i].bound_check:
        print("Passed")
    else:
        print("Failed")


# # Micro object changes material data and generates new bdf files
# # Micro object runs bdf files in Nastran
# # Micro object interperates f06 file, gets (area, and element stresses)
write_dataset(micro_arr)
data = read_dataset("data_set")
stop_time = time.time()
run_time = stop_time - start_time  # seconds
print(run_time)

print(1)

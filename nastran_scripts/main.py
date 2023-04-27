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
    with open(f'data_set.txt', mode='w') as file:
        file.writelines(data)


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
tests = 1000
micro_arr = []
for i in range(tests):
    D1 = D_1[i]
    D2 = D_2[i]
    phase_1 = Phase.PhaseMat(D1)
    phase_2 = Phase.PhaseMat(D2)
    micro_arr.append(Phase.Micro(phase_1, phase_2, test_nr, grid_data, e_area))
    micro_arr[i].elast_bounds()
    test_nr += 1
    time.sleep(10)
time.sleep(25)

for micro in micro_arr:
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

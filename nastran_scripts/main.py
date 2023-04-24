import numpy as np
import time
import Phase
import nastran_file_generator as nas_gen
import phase_comp_generator as ph_gen

# def write_dataset(micro_arr):
#     data = str
#     for micro in micro_arr:
#         data.append(f"{micro.}\n")
#     with open(f'nastran_output/data_set.txt', mode='w') as file:
#         line = f"{micro}\n"
#         file.writelines()
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

# nas_gen.linear_displacement("new_orig_x", grid_data, "x", 10.2)
# nas_gen.linear_displacement("new_orig_y", grid_data, "y", 10.2)
# nas_gen.linear_shear_displacement("new_orig_xy", grid_data,  10.2)


test_nr = 0
micro_arr = []
for i in range(10):
    D1 = D_1[i]
    D2 = D_2[i]
    phase_1 = Phase.PhaseMat(D1)
    phase_2 = Phase.PhaseMat(D2)
    micro_arr.append(Phase.Micro(phase_1, phase_2, test_nr, grid_data, e_area))
    micro_arr[i].elast_bounds()
    test_nr += 1

# for i in range(10):
#     micro_arr[i].move_f06_files()

for i in range(10):
    micro_arr[i].calc_stresses(e_node_2)

# micro_test.elast_bounds()
# micro_test.calc_stresses(e_node_2)
# # Micro object changes material data and generates new bdf files
# # Micro object runs bdf files in Nastran
# # Micro object interperates f06 file, gets (area, and element stresses)
# print(1)
# test_file = "x_orig_eps1"
# el_strain = nas_gen.read_el_strain(test_file, nel)
# vol_avg_strain = nas_gen.vol_avg_strain(e_area, el_strain)
#
# nas_gen.linear_shear_displacement(orig_file_name, grid_data)
print(1)

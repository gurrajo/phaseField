import numpy as np
import time
import Phase
import nastran_file_generator as nas_gen
orig_file_name = "new_orig_xy_no_lin"

# generate material space with latin hypercube
#(E_1, E_2, nu_1, nu_2) = ph_gen.phaseinator()

# calculate pre-proc data from bdf original file
grid_data = nas_gen.read_grid_data(orig_file_name)  # node position data
e_node_tri = nas_gen.tri_element_nodes(orig_file_name)
e_area_tri = nas_gen.tri_element_area(grid_data, e_node_tri)  # area for each triangular element
e_node_quad = nas_gen.quad_element_nodes(orig_file_name)
e_area_quad = nas_gen.quad_element_area(grid_data, e_node_quad)  # area for each quadrilateral element
e_area = np.append(e_area_quad, e_area_tri, 0)
nel = len(e_area)
e_node_2 = np.append(e_node_quad[:, 0:3], e_node_tri[:, 0:3], 0)
nas_gen.linear_shear_displacement(orig_file_name, grid_data)
# from a material sample (2 phases) generate a Micro object
test_nr = 1
E_1 = 60
E_2 = 40
nu_2 = 0.3
G_12 = 20
phase_1 = Phase.PhaseMat(E_1, E_2, nu_2, G_12)
E_1 = 45
E_2 = 30
nu_2 = 0.2
G_12 = 15
phase_2 = Phase.PhaseMat(E_1, E_2, nu_2, G_12)
micro_test = Phase.Micro(phase_1, phase_2, test_nr, grid_data, e_area)
micro_test.calc_stresses(e_node_2)
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

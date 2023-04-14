import numpy as np

import Phase
import nastran_file_generator as nas_gen
orig_file_name = "spcd_shear_strain"

# generate material space with latin hypercube
#(E_1, E_2, nu_1, nu_2) = ph_gen.phaseinator()

# calculate pre-proc data from bdf original file
grid_data = nas_gen.read_grid_data(orig_file_name)  # node position data
e_node_tri = nas_gen.tri_element_nodes(orig_file_name)
e_area_tri = nas_gen.tri_element_area(grid_data, e_node_tri)  # area for each triangular element
e_node_quad = nas_gen.quad_element_nodes(orig_file_name)
e_area_quad = nas_gen.quad_element_area(grid_data, e_node_quad)  # area for each quadrilateral element

# generates 3 bdf files with linearized loads
# nas_gen.linear_displacement(orig_file_name, grid_data, "x")
# nas_gen.linear_displacement(orig_file_name, grid_data, "y")
nas_gen.linear_shear_displacement(orig_file_name, grid_data)

# from a material sample (2 phases) generate a Micro object
test_nr = 1
E_1 = 13.41919191919191
E_2 = 78.912312313211231231313
nu_2 = 0.36
G_12 = 23.1
phase_1 = Phase.PhaseMat(E_1, E_2, nu_2, G_12)
E_1 = 305.4
E_2 = 23.9
nu_2 = 0.87
G_12 = 211.1
phase_2 = Phase.PhaseMat(E_1, E_2, nu_2, G_12)
n_el = len(e_node_tri) + len(e_node_quad)
micro_test = Phase.Micro(phase_1, phase_2, test_nr, grid_data)
e_stress = nas_gen.read_el_stress(orig_file_name, n_el)
e_strain = nas_gen.read_el_strain(orig_file_name, n_el)
e_area = np.append(e_area_quad, e_area_tri, 0)
eps = nas_gen.vol_avg_stress(e_area, e_stress)
sig = nas_gen.vol_avg_strain(e_area, e_strain)
# Micro object changes material data and generates new bdf files
# Micro object runs bdf files in Nastran
# Micro object interperates f06 file, gets (area, and element stresses)
print(1)

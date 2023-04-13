import Phase
import phase_comp_generator as ph_gen
import nastran_file_generator as nas_gen
orig_file_name = "micro_struc_hm_current_test_new_load"

# generate material space with latin hypercube
(E_1, E_2, nu_1, nu_2) = ph_gen.phaseinator()

# calculate pre-proc data from bdf original file
grid_data = nas_gen.read_grid_data(orig_file_name)  # node position data
e_node_tri = nas_gen.tri_element_nodes(orig_file_name)
e_area_tri = nas_gen.tri_element_area(grid_data, e_node_tri)  # area for each triangular element
e_node_quad = nas_gen.quad_element_nodes(orig_file_name)
e_area_quad = nas_gen.quad_element_area(grid_data, e_node_quad)  # area for each quadrilateral element

# generates 3 bdf files with linearized loads
nas_gen.linear_displacement(orig_file_name, grid_data, "x")
nas_gen.linear_displacement(orig_file_name, grid_data, "y")
nas_gen.linear_displacement(orig_file_name, grid_data, "xy")

# from a material sample (2 phases) generate a Micro object
# Micro object changes material data and generates new bdf files
# Micro object runs bdf files in Nastran
# Micro object interperates f06 file, gets (area, and element stresses), or (boundary reaction forces)


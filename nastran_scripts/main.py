import Phase
import phase_comp_generator as ph_gen

# generate material space with latin hypercube
(E_1, E_2, nu_1, nu_2) = ph_gen.phaseinator()

# calculate pre-proc data from bdf original file

# generates 3 bdf files with linearized loads

# from a material sample (2 phases) generate a Micro object
# Micro object changes material data and generates new bdf files
# Micro object runs bdf files in Nastran
# Micro object interperates f06 file, gets (area, and element stresses), or (boundary reaction forces)


# phaseField
The phasefield repository includes 2 main parts the phase filed simulations and the computational homogenization simulations using nstran scripts.
The phasefield simulations are based on the pystencils library and the tutorials for spinodal decompostion by Martin Bauer, Markus Holzer, Frederik Hennig, https://pycodegen.pages.i10git.cs.fau.de/pystencils/notebooks/05_tutorial_phasefield_spinodal_decomposition.html

The nastran scripts folder includes the Phase file with the Micro class which representes a dual-phased micro structure.
The micro class also performs the necessary modifications to nastran files to alter loads and change material values and after simulations extracting the results.
In total 1000 micro objects are made to get a dataset for differernt homogenized parameteres from phase parameters. 
![Alt text](prog_flow_chart.PNG"Flowchart")

# phaseField
The phasefield repository incldues 2 main parts the phase filed simulations and the computational homogenization simulations using nstran scripts.
The nastran scripts folder includes the Micro class which representes a dual-phased micro structure.
The micro class also performs the necessary modifications to nastran files to alter loads and change material values and after simulations extracting the results.
In total 1000 micro objects are made to get a dataset for differernt homogenized parameteres from phase parameters. 

This folder contains simulation scripts to run our project simulations.

To run a simulation, follow these steps:
    1. Put the desired parameters in the `parameters.txt` file.
        model: model type ("XY", "ln")
        L: lattice size
        T: list of temperatures
        J: coupling constant for XY model
        epsilon: disorder strength for ln model
        gamma: parameter for ln model
        A: parameter for ln model
        rho: density
        n_therm: number of thermalization steps
        n_meas: number of measurement steps
        overrelax_interval: interval for overrelaxation updates
        meas_interval: interval for measurements
        coeur: max number of CPU cores to use
    2. Run the simulation script launch_parallel.py
    3. Results will be saved in an output file named according to the parameters used.

Source code for the simulation are in:
    - MC_XY.py
    - MC_ln.py

Both files contain numba optimized functions, a Geometry class for lattice handling, a Simulation class for running the simulations and saving results. Plotting functions are at the end of each file.

Jupyter notebooks are used for data analysis and visualization.

To-do:
    - Possibility to resume simulations from a checkpoint
    - Create a separate plotting script with parameter loading functionality
    - Try to write simulation in C++ for performance comparison
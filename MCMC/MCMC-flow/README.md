# MCMC Flow 

## Instructions:

### Set up the parameter space:  
- Edit the `init.py` file in the `src` directory with the desired statepoints.

    ```
    parameters["n_steps"] = [[1e7, 1e8]]
    parameters["kT"] = [[10, 1.5]]
    parameters["max_trans"] = [[3.0, 0.5]]
    ```
  Make sure the length of the list for above parameters are the same. The first stage of run is used for mixing the system (changing it from a hex packed system to a random one) at high kT.

 In order to keep the runs consistent, for the first stage (first element in the above lists), let's always use `n_steps=1e7`, `kT=10` and `max_trans=3`.
- Run `python init.py` to generate the signac workspace 

## Submit simulations:
- Run `python project.py submit` to run MCMC simulation.



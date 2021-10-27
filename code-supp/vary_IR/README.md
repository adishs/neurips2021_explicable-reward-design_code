# Explicable Reward Design for Reinforcement Learning Agents
## Prerequisites:
```
Python3
Matplotlib
Numpy
Itertools
cvxpy
```

## Running the code
To get results, you will need to run the following scripts:

### For the visualization of the Q-learning convergence plots for RoomsNavEnv run:
```
python run_fourroom_plot_convergence.py --figure=<figureName> --n_average=<numberOfRunsToAverage>
```
The argument ``figureName`` should be one of the following:  9.a, 9.b, 9.c, 9.d, 9.e, 9.f.
As a concrete example, one can run the following scripts to get result:
```
run_fourroom_plot_convergence.py --figure=9.a --n_average=5
```
### Results
After running the above script, new plots will be created in output/ directory.
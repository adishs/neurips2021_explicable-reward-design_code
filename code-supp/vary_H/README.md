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
The first parameter ``--figure=<figureName>`` is one of the following:  10.a, 10.b, and 10.c.

The second parameter `--n_average=<numberOfRunsToAverage>` denotes the number of the runs to be averaged.

As a concrete example, one can run the following script to get result:
```
python run_fourroom_plot_convergence.py --figure=10.a --n_average=5
```
### Results
After running the above script, new plots will be created in output/ directory.
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


### For the visualization of the Q-learning convergence plots for LineKeyNavEnv run:
```
python run_linekey_plot_convergence.py --n_average=<numberOfRuns>
```
The parameter `--n_average=<numberOfRunsToAverage>` denotes the number of the runs to be averaged.

As a concrete example, one can run the following scripts to get result:
```
python run_linekey_plot_convergence.py --n_average=5
```

### Results
After running the above script, new plots will be created in output/ directory.

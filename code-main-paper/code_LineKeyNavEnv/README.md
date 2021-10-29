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

### For the visualization of the designed reward for  LineKeyNavEnv, run:
```
python linekey_run.py --algorithm=<algorithmName> --B=<budget>
```
where the parameter `--algorithm=<algorithmName>` is one of the following: orig, pbrs, and exprd.

For the algorithm exprd, the parameter `--B=<budget>` denotes the budget, i.e., the number of states to be selected for designing the reward function. Set a small number for faster execution.

As a concrete example, one can run the following scripts to get result:
```
python linekey_run.py --algorithm=exprd --B=1
```
```
python linekey_run.py --algorithm=orig --B=1
```
```
python linekey_run.py --algorithm=pbrs --B=1
```

### For the visualization of the Q-learning convergence plots for LineKeyNavEnv run:
```
python run_linekey_plot_convergence.py --n_average=<numberOfRuns>
```
The parameter `--n_average=<numberOfRunsToAverage>` denotes the number of the runs to be averaged.

As a concrete example, one can run the following script to get result:
```
python run_linekey_plot_convergence.py --n_average=5
```

### Results
After running the above script, new plots will be created in output/ directory.

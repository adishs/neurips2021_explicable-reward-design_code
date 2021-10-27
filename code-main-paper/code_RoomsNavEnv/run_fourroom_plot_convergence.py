import sys
import os
import argparse




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for Plots : Number of average')
    parser.add_argument('--n_average', nargs='?', const=1, type=int, default=30)
    args = parser.parse_args()
    # Parameters
    n_average = args.n_average

    # #Reward design
    print(" Reward design started ...")
    os.system('python run_fourroom_reward_design.py')
    print(" Reward design ended ...")


    #Q-learning on designed rewards
    print(" Q-learning  started ...")
    os.system('python run_fourroom_q_learning.py {}'.format(n_average))
    print(" Q-learning  ended ...")

    #Plot Q-learning Data
    print(" Plotting started ...")
    os.system('python fourroom_plottingscript.py {}'.format(n_average))
    print(" Plotting end ...")

    # #remove directory where the data was stored
    os.system("rm -rf results/")







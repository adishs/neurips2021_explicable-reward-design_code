import sys
import os
import argparse



if __name__ == "__main__":

    settings = ['9.a', '9.b', '9.c', '9.d', '9.e', '9.f']

    parser = argparse.ArgumentParser(description='Arguments for Plots : '
                                                 'Figure, Number of average')

    parser.add_argument('--figure', nargs='?', const=1, type=str, default="")
    parser.add_argument('--n_average', nargs='?', const=1, type=int, default=30)
    args = parser.parse_args()
    # Parameters

    figure = args.figure
    n_average = args.n_average


    if figure not in settings:
        print("Please choose one of the settings from: ", settings)
        exit(0)

    if figure == '9.a':
        setting_to_run = 'max_a_hinge_delta_s'

    if figure == '9.b':
        setting_to_run = 'max_a_hinge_delta_s_a'

    if figure == "9.c":
        setting_to_run = 'sum_a_hinge_delta_s'

    if figure == '9.d':
        setting_to_run = 'sum_a_hinge_delta_s_a'

    if figure == '9.e':
        setting_to_run = 'sum_a_linear_delta_s_a'

    if figure == "9.f":
        setting_to_run = 'sum_a_exp_delta_s_a'



    # #Reward design
    print(" Reward design started ...")
    os.system('{0} run_{1}.py'.format('python', setting_to_run))
    print(" Reward design ended ...")


    # #Q-learning on designed rewards
    print(" Q-learning  started ...")
    input_dir_name = "designed_rewards_room_{}".format(setting_to_run)
    os.system('{0} {1} {2} {3}'.format('python',
                                                        'run_fourroom_q_learning.py',
                                                         input_dir_name, n_average))
    print(" Q-learning  ended ...")

    #Plot Q-learning Data
    print(" Plotting started ...")
    os.system('python fourroom_plottingscript.py Q_learning_{} {} {}'.format(
        input_dir_name, n_average, figure))
    print(" Plotting end ...")

    #remove directory where the data was stored
    os.system("rm -rf results/")







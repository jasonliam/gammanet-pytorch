from experiment import Experiment
import sys

import numpy as np

from file_utils import read_file_in_dir


def run_experiment(experiment_name):
    print("Running Experiment: ", experiment_name)
    exp = Experiment(experiment_name)
    exp.run()


if __name__ == "__main__":
    exp_name = 'default'

    # if len(sys.argv) > 1:
    #     exp_name = sys.argv[1]
    #
    # run_experiment(exp_name)

    data = read_file_in_dir('./', 'results-2.json')
    for e in data:
        d = data[e]
        a = ['current', 'best_loss', 'best_dice']
        dices = [d[i][1] for i in a]
        best = np.argmax(dices)
        str = " {} : {:.2f} ({:.2f}, {:.2f})".format(e, 100 * d[a[best]][1], 100 * d[a[best]][2], 100 * d[a[best]][3])
        print(str)

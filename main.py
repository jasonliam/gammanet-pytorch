from experiment import Experiment
import sys


def run_experiment(experiment_name):
    print("Running Experiment: ", experiment_name)
    exp = Experiment(experiment_name)
    exp.run()


if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    run_experiment(exp_name)

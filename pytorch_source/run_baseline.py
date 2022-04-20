import utility
import pandas
from baselines import always_on
from baselines import optimal
from baselines import naive
from baselines import svm


def simulate_perfect_knowledge(envs):
    """Runs an optimal model on a list of environments and stores the results in optimal.csv
    Arguments:
        envs (list): A list of StreetParkingEnv environments.
    """
    dics = []
    for i in range(len(envs)):
        optimal.simulate_perfect_knowledge_env(envs[i])
        dics.append(utility.get_run_stats(envs[i]))

    report = {}
    for k in dics[0].keys():
        report[k] = []
    for d in dics:
        for k in report.keys():
            report[k].append(d[k])

    df = pandas.DataFrame(report)
    df.to_csv("optimal.csv")

def simulate_always_on(end_with_battery, envs):
    """Runs an always-on model on a list of environments and stores the results in always_on.csv
    Arguments:
        end_with_battery (bool): If end_with_battery==True, the simulation of the environment stops when the battery depletes 
            otherwise it goes into recovery mode and recharge. 
        envs (list): A list of StreetParkingEnv environments.
    """
    dics = []
    for i in range(len(envs)):
        always_on.simulate_always_on_env(envs[i], end_with_battery=end_with_battery)
        dics.append(utility.get_run_stats(envs[i]))

    report = {}
    for k in dics[0].keys():
        report[k] = []
    for d in dics:
        for k in report.keys():
            report[k].append(d[k])

    df = pandas.DataFrame(report)
    df.to_csv("always_on.csv")

def simulate_naive(end_with_battery, train_env, test_envs):
    """Runs a naive model on a list of environments and stores the results in naive.csv
    Arguments:
        end_with_battery (bool): If end_with_battery==True, the simulation of the environment stops when the battery depletes 
            otherwise it goes into recovery mode and recharge. 
        train_env (StreetParkingEnv): A StreetParkingEnv environment used for training.
        test_envs (list): A list of StreetParkingEnv environments used for testing.
    """
    dics = []
    for i in range(len(test_envs)):
        lh, uh = naive.train(train_env, i)
        naive.test(lh, uh, test_envs[i], end_with_battery=end_with_battery)
        dics.append(utility.get_run_stats(test_envs[i]))

    report = {}
    for k in dics[0].keys():
        report[k] = []
    for d in dics:
        for k in report.keys():
            report[k].append(d[k])

    df = pandas.DataFrame(report)
    df.to_csv("naive.csv")


def simulate_svm(end_with_battery, train, train_env, test_envs):
    """Runs an SVM model on a list of environments and stores the results in svm.csv
    Arguments:
        end_with_battery (bool): If end_with_battery==True, the simulation of the environment stops when the battery depletes 
            otherwise it goes into recovery mode and recharge.
        train (bool): If train == True a new svm model will be trained on train_env otherwise it will use an existing model
            in ./svm_models/ directory.
        train_env (StreetParkingEnv): A StreetParkingEnv environment used for training.
        test_envs (list): A list of StreetParkingEnv environments used for testing.
    """
    dics = []
    for i in range(len(test_envs)):
        if train:
            svm.train(train_env, "./svm_models/", idx = i, second_class_weight = 10)
        svm.test(test_envs[i], "./svm_models/", env_idx = i, end_with_battery=end_with_battery, render = False , idx = 0)
        dics.append(utility.get_run_stats(test_envs[i]))

    report = {}
    for k in dics[0].keys():
        report[k] = []
    for d in dics:
        for k in report.keys():
            report[k].append(d[k])

    df = pandas.DataFrame(report)
    df.to_csv("svm.csv")

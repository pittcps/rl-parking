import pandas
import numpy as np
from sklearn import svm
import pickle


def data_prepration(original_data):
    data = original_data.to_numpy()
    x = data[:, 2:5]
    y = (data[:, -3] == 2)

    return x, y
    

def train(env, weights_saving_address, idx = 0, second_class_weight = 5):

    data = env.df_list[idx]
    train_x, train_y = data_prepration(data)


    model = svm.SVC(class_weight={1: second_class_weight})
    model = model.fit(train_x, train_y)

    filename = 'svm_model_{}.sav'.format(idx)
    pickle.dump(model, open(weights_saving_address + filename, 'wb'))

def test(env, weights_saving_address, env_idx, end_with_battery=False, render = False , idx = 0):
    data = env.df_list[idx]
    test_x, test_y = data_prepration(data)

    print("loading weights")

    filename = 'svm_model_{}.sav'.format(env_idx)
    model = pickle.load(open(weights_saving_address + filename, 'rb'))

    print("weights loaded")

    predicts = model.predict(test_x)

    print("prediction finished!")


    env.reset()
    env.current_step = 0
    env.df_index = idx

    on_actions = 0

    while True:

        if env.current_step + 1 >= len(env.df_list[env.df_index]):
            break
        if predicts[env.current_step] == 1:
            action = 1
            on_actions += 1
        else:
            action = 0

        obs, reward, done, info = env.step(action)
        if render:
            env.render()
        if done and info['is_unfinished'] == False:
            break
        if done and end_with_battery:
            break


    print("{} steps out of {}".format(env.current_step, len(env.df_list[env.df_index])))
    env.render()
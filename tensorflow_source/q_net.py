from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Flatten

def get_model(env):
    nb_actions = env.action_space.n
    observation_space = env.observation_space.shape[0]

    ## create model
    model = Sequential()
    model.add(Dense(input_shape=(1, observation_space), units=32, activation="relu"))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model

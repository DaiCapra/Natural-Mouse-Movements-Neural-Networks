from keras import Sequential
from keras.layers import Dense, Reshape

target_path_count = 100


def init_model_paths():
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=2))
    model.add(Dense(target_path_count * 2, activation='linear'))
    model.add(Reshape((target_path_count, 2)))

    # model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

    return model


def init_model_time():
    model = Sequential()
    model.add(Dense(216, activation='relu', input_dim=2))
    model.add(Dense(target_path_count, activation='linear'))
    model.add(Reshape((target_path_count, 1)))

    # model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

    return model

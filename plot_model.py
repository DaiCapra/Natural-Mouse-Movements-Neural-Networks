import matplotlib.pyplot as plt
import numpy
import data_loader as dl


# Predict
def predict(model, inputs):
    values = numpy.array(inputs)
    v = model.predict(values.reshape(1, 2))
    return v


def plot_paths(paths):
    for path in paths:
        X = []
        Y = []

        for point in path:
            X.append(point[0])
            Y.append(point[1])

        plt.plot(X, Y)
    return


def plot(model, history, train_inputs, train_paths):
    plt.figure(1)
    plt.title('user input')
    plot_paths(train_paths)

    plt.figure(2)
    size = train_inputs.__len__()
    for x in range(size):
        v = train_inputs[x]
        paths = predict(model, v)
        plot_paths(paths)

    plt.title('generated')

    plt.figure(3)
    plt.title('accuracy')
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(['acc', 'loss'], loc='upper left')
    plt.show()

    return

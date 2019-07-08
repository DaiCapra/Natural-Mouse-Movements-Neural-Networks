import data_loader as dl
import init_model
import keras
from keras.callbacks import ModelCheckpoint
import numpy
import matplotlib.pyplot as plt

from matplotlib.pyplot import bar

(train_inputs, train_paths, train_time) = dl.load_data("train/data.json")

# Model
model = init_model.init_model_time()

file_path_best = "models/time/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path_best, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit
history = model.fit(train_inputs, train_time, epochs=1000, verbose=0, validation_split=0.05,
                    callbacks=callbacks_list)

# train_inputs.__len__()
size = 1
for x in range(size):
    values = numpy.array(train_inputs[x])
    times = model.predict(values.reshape(1, 2))
    flatted = times.flatten()
    i = 0
    for value in flatted:
        plt.bar(i, value)
        i = i + 1

plt.show()

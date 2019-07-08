import onnxmltools
from keras import Sequential
from keras.models import load_model
import data_loader
import plot_model
import matplotlib.pyplot as plt


file_name_output = "model.onnx"
file_name_folder = "models/"
file_name_model = "path/weights-improvement-314-0.90.hdf5"
s = file_name_folder + file_name_model

(train_inputs, train_paths, train_time) = data_loader.load_data("train/data.json")

model: Sequential = load_model(s)

# Plot generated movements
size = 20
for x in range(size):
    v = train_inputs[x]
    line = plot_model.predict(model, v)
    plot_model.plot_paths(line)

plt.show()

# Save Model
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, file_name_folder + file_name_output)
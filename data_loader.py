import json
import numpy
import pandas

def load_data(file_path_paths):
    # Load paths
    data = load_data_from_json(file_path_paths)

    # Format paths
    paths = pandas.DataFrame(data).values

    l_path = []
    l_destination = []
    l_time = []

    for path in paths:
        t = path[0]
        x = path[1]
        y = path[2]
        last_x = x[-1]
        last_y = y[-1]
        l_destination.append([last_x, last_y])

        a = list(zip(x, y))
        l_path.append(a)
        l_time.append(list(zip(t)))

    l_path = numpy.array(l_path)
    l_destination = numpy.array(l_destination)
    l_time = numpy.array(l_time)
    return l_destination, l_path, l_time


def load_data_from_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)

    return data


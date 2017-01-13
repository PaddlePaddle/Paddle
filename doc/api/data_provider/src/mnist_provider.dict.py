from paddle.trainer.PyDataProvider2 import *


# Define a py data provider
@provider(
    input_types={'pixel': dense_vector(28 * 28),
                 'label': integer_value(10)})
def process(settings, filename):  # settings is not used currently.
    f = open(filename, 'r')  # open one of training file

    for line in f:  # read each line
        label, pixel = line.split(';')

        # get features and label
        pixels_str = pixel.split(' ')

        pixels_float = []
        for each_pixel_str in pixels_str:
            pixels_float.append(float(each_pixel_str))

        # give data to paddle.
        yield {"pixel": pixels_float, 'label': int(label)}

    f.close()  # close file

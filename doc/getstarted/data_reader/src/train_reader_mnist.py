def train_reader(file_name):
    def reader():
        # open one of training file
        with open(file_name, 'r') as f:
            # read each line
            for line in f:
                label, pixel = line.strip().split(';')
                # get features and label
                pixels_str = pixel.split(' ')
                pixels_float = []
                for each_pixel_str in pixels_str:
                    pixels_float.append(float(each_pixel_str))
                # give data to paddle.
                yield pixels_float, int(label)

    return reader

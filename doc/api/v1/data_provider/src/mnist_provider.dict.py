#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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

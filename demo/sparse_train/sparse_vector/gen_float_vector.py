# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from random import uniform,randint

# total samples to generate
samples=2000

# total DIMENSION
dimension=10000000

# float value number
float_num_min=10
float_num_max=100

# float value range
float_min=0.01
float_max=5.00

# output range
regression_min=0.01
regression_max=100.00

data = "train.txt"


'''
generate dummy data for testing sparse_float_vector input
format:
    label, [index, value], [index, value], ....
'''
with open(data, "w") as f:
    for ids in range(samples):
        num = randint(float_num_min, float_num_max)
        index = 0
        value = 0.0
        output = uniform(regression_min, regression_max)
        f.write("%f," % output)
        for i in range(num):
            index = randint(0, dimension)
            value = uniform(float_min, float_max)
            if i == num -1:
                f.write("%d %f" % (index, value))
            else:
                f.write("%d %f," % (index, value))

        f.write("\n")



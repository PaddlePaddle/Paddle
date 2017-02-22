# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from py_paddle import DataProviderConverter

__all__ = ['DataFeeder']
"""
DataFeeder converts the data returned by paddle.reader into a data structure
of Arguments which is defined in the API. The paddle.reader usually returns
a list of mini-batch data. Each item in the list is a tuple or list, which is
one sample with multiple features. DataFeeder converts this mini-batch data
into Arguments in order to feed it to C++ interface.

The example usage:

    data_types = [paddle.data_type.dense_vector(784),
                  paddle.data_type.integer_value(10)]
    feeder = DataFeeder(input_types=data_types)
    minibatch_data = [
                       ( [1.0,2.0,3.0,4.0], 5, [6,7,8] ),  # first sample
                       ( [1.0,2.0,3.0,4.0], 5, [6,7,8] )   # second sample
                     ]

    #  or 
    #  minibatch_data = [
    #                     [ [1.0,2.0,3.0,4.0], 5, [6,7,8] ],  # first sample
    #                     [ [1.0,2.0,3.0,4.0], 5, [6,7,8] ]   # second sample
    #                   ]
    arg = feeder(minibatch_data)


Args:
    input_types: A list of input data types. It's length is equal to the length
                 of data returned by paddle.reader. Each item specifies the type
                 of each feature.
    mintbatch_data: A list of mini-batch data. Each item is a list or tuple,
                    for example:
                    [ 
                      (feature_0, feature_1, feature_2, ...), # first sample
                      (feature_0, feature_1, feature_2, ...), # second sample
                      ...
                    ]

Returns:
    An Arguments object contains this mini-batch data with multiple features.
    The Arguments definition is in the API.
"""

DataFeeder = DataProviderConverter

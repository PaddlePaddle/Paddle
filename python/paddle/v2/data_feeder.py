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

from py_paddle import swig_paddle
from py_paddle import DataProviderConverter
import data_type

__all__ = ['DataFeeder']


class DataFeeder(DataProviderConverter):
    """
    DataFeeder converts the data returned by paddle.reader into a data structure
    of Arguments which is defined in the API. The paddle.reader usually returns
    a list of mini-batch data entries. Each data entry in the list is one sampe.
    Each sample is a list or a tuple with one feature or multiple features.
    DataFeeder converts this mini-batch data entries into Arguments in order
    to feed it to C++ interface.
    
    The example usage:
    
        data_types = [('image', paddle.data_type.dense_vector(784)),
                      ('label', paddle.data_type.integer_value(10))]
        reader_dict = {'image':0, 'label':1}
        feeder = DataFeeder(data_types=data_types, reader_dict=reader_dict)
        minibatch_data = [
                           ( [1.0,2.0,3.0,4.0], 5, [6,7,8] ),  # first sample
                           ( [1.0,2.0,3.0,4.0], 5, [6,7,8] )   # second sample
                         ]
        # or minibatch_data = [
        #                       [ [1.0,2.0,3.0,4.0], 5, [6,7,8] ],  # first sample
        #                       [ [1.0,2.0,3.0,4.0], 5, [6,7,8] ]   # second sample
        #                     ]
        arg = feeder(minibatch_data)
    """

    def __init__(self, data_types, reader_dict):
        """
        :param data_types: A list to specify data name and type. Each item is
                           a tuple of (data_name, data_type). For example:
                           [('image', paddle.data_type.dense_vector(784)),
                            ('label', paddle.data_type.integer_value(10))]

        :type data_types: A list of tuple
        :param reader_dict: A dictionary to specify the position of each data
                            in the input data.
        :type reader_dict: dict()
        """
        self.input_names = []
        self.input_types = []
        self.reader_dict = reader_dict
        for each in data_types:
            self.input_names.append(each[0])
            self.input_types.append(each[1])
            assert isinstance(each[1], data_type.InputType)
        DataProviderConverter.__init__(self, self.input_types)

    def convert(self, dat, argument=None):
        """
        :param dat: A list of mini-batch data. Each sample is a list or tuple
                    one feature or multiple features.
                    for example:
                    [ 
                      ([0.2, 0.2], ), # first sample
                      ([0.8, 0.3], ), # second sample
                    ]
                    or,
                    [ 
                      [[0.2, 0.2], ], # first sample
                      [[0.8, 0.3], ], # second sample
                    ]

        :type dat: List
        :param argument: An Arguments object contains this mini-batch data with
                         one or multiple features. The Arguments definition is
                         in the API.
        :type argument: swig_paddle.Arguments
        """

        if argument is None:
            argument = swig_paddle.Arguments.createArguments(0)
        assert isinstance(argument, swig_paddle.Arguments)
        argument.resize(len(self.input_types))

        scanners = [
            DataProviderConverter.create_scanner(i, each_type)
            for i, each_type in enumerate(self.input_types)
        ]

        for each_sample in dat:
            for name, scanner in zip(self.input_names, scanners):
                scanner.scan(each_sample[self.reader_dict[name]])

        for scanner in scanners:
            scanner.finish_scan(argument)

        return argument

    def __call__(self, dat, argument=None):
        return self.convert(dat, argument)

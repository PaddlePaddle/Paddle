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
import collections
import paddle.trainer.PyDataProvider2 as pydp2

__all__ = ['DataFeeder']


def default_feeding_map(data_types):
    reader_dict = dict()
    for i, tp in enumerate(data_types):
        reader_dict[tp[0]] = i
    return reader_dict


class DataFeeder(DataProviderConverter):
    """
    DataFeeder converts the data returned by paddle.reader into a data structure
    of Arguments which is defined in the API. The paddle.reader usually returns
    a list of mini-batch data entries. Each data entry in the list is one sample.
    Each sample is a list or a tuple with one feature or multiple features.
    DataFeeder converts this mini-batch data entries into Arguments in order
    to feed it to C++ interface.
    
    The simple usage shows below

    ..  code-block:: python

        feeding = ['image', 'label']
        data_types = enumerate_data_types_of_data_layers(topology)
        feeder = DataFeeder(data_types=data_types, feeding=feeding)

        minibatch_data = [([1.0, 2.0, 3.0, ...], 5)]

        arg = feeder(minibatch_data)


    If mini-batch data and data layers are not one to one mapping, we
    could pass a dictionary to feeding parameter to represent the mapping
    relationship.


    ..  code-block:: python

        data_types = [('image', paddle.data_type.dense_vector(784)),
                      ('label', paddle.data_type.integer_value(10))]
        feeding = {'image':0, 'label':1}
        feeder = DataFeeder(data_types=data_types, feeding=feeding)
        minibatch_data = [
                           ( [1.0,2.0,3.0,4.0], 5, [6,7,8] ),  # first sample
                           ( [1.0,2.0,3.0,4.0], 5, [6,7,8] )   # second sample
                         ]
        # or minibatch_data = [
        #                       [ [1.0,2.0,3.0,4.0], 5, [6,7,8] ],  # first sample
        #                       [ [1.0,2.0,3.0,4.0], 5, [6,7,8] ]   # second sample
        #                     ]
        arg = feeder.convert(minibatch_data)

    ..  note::

        This module is for internal use only. Users should use the `reader`
        interface.



    :param data_types: A list to specify data name and type. Each item is
                       a tuple of (data_name, data_type).

    :type data_types: list
    :param feeding: A dictionary or a sequence to specify the position of each
                    data in the input data.
    :type feeding: dict|collections.Sequence|None
    """

    def __init__(self, data_types, feeding=None):
        self.input_names = []
        input_types = []
        if feeding is None:
            feeding = default_feeding_map(data_types)
        elif isinstance(feeding, collections.Sequence):
            feed_list = feeding
            feeding = dict()
            for i, name in enumerate(feed_list):
                feeding[name] = i
        elif not isinstance(feeding, dict):
            raise TypeError("Feeding should be dict or sequence or None.")

        self.feeding = feeding
        for each in data_types:
            self.input_names.append(each[0])
            if not isinstance(each[1], pydp2.InputType):
                raise TypeError("second item in each data_type should be an "
                                "InputType")
            input_types.append(each[1])
        DataProviderConverter.__init__(self, input_types)

    def __len__(self):
        return len(self.input_names)

    def convert(self, dat, argument=None):
        """
        :param dat: A list of mini-batch data. Each sample is a list or tuple
                    one feature or multiple features.

        :type dat: list
        :param argument: An Arguments object contains this mini-batch data with
                         one or multiple features. The Arguments definition is
                         in the API.
        :type argument: py_paddle.swig_paddle.Arguments
        """

        def reorder_data(data):
            retv = []
            for each in data:
                reorder = []
                for name in self.input_names:
                    reorder.append(each[self.feeding[name]])
                retv.append(reorder)
            return retv

        return DataProviderConverter.convert(self, reorder_data(dat), argument)

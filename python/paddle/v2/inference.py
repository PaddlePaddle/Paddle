#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import collections
import topology
import paddle
import cPickle

__all__ = ['infer', 'Inference']


class Inference(object):
    """
    Inference combines neural network output and parameters together
    to do inference.

    ..  code-block:: python

        inferer = Inference(output_layer=prediction, parameters=parameters)
        for data_batch in batches:
            print inferer.infer(data_batch)


    :param output_layer: The neural network that should be inferenced.
    :type output_layer: paddle.v2.config_base.Layer or the sequence
                        of paddle.v2.config_base.Layer
    :param parameters: The parameters dictionary.
    :type parameters: paddle.v2.parameters.Parameters
    """

    def __init__(self, parameters, output_layer=None, fileobj=None):
        import py_paddle.swig_paddle as api

        if output_layer is not None:
            topo = topology.Topology(output_layer)
            gm = api.GradientMachine.createFromConfigProto(
                topo.proto(), api.CREATE_MODE_TESTING, [api.PARAMETER_VALUE])
            self.__data_types__ = topo.data_type()
        elif fileobj is not None:
            tmp = cPickle.load(fileobj)
            gm = api.GradientMachine.createByConfigProtoStr(
                tmp['protobin'], api.CREATE_MODE_TESTING,
                [api.PARAMETER_VALUE])
            self.__data_types__ = tmp['data_type']
        else:
            raise ValueError("Either output_layer or fileobj must be set")

        for param in gm.getParameters():
            val = param.getBuf(api.PARAMETER_VALUE)
            name = param.getName()
            assert isinstance(val, api.Vector)
            val.copyFromNumpyArray(parameters.get(name).flatten())
            # the setValueUpdated function is called in randomize, zeroMem,
            # load function in paddle/parameter/Parameter.cpp. But in the
            # inference mode, the setValueUpdated is never called, it will
            # cause the parameter will not be dispatched
            # in MultiGradientMachine for multi-GPU. So setValueUpdated is
            # called here, but it's better to call this function in one place.
            param.setValueUpdated()
        self.__gradient_machine__ = gm

    def iter_infer(self, input, feeding=None):
        from data_feeder import DataFeeder
        feeder = DataFeeder(self.__data_types__, feeding)
        batch_size = len(input)

        def __reader_impl__():
            for each_sample in input:
                yield each_sample

        reader = paddle.batch(__reader_impl__, batch_size=batch_size)

        self.__gradient_machine__.start()
        for data_batch in reader():
            yield self.__gradient_machine__.forwardTest(feeder(data_batch))
        self.__gradient_machine__.finish()

    def iter_infer_field(self, field, **kwargs):
        if not isinstance(field, list) and not isinstance(field, tuple):
            field = [field]

        for result in self.iter_infer(**kwargs):
            for each_result in result:
                item = [each_result[each_field] for each_field in field]
                yield item

    def infer(self, input, field='value', flatten_result=True, **kwargs):
        """
        Infer a data by model.
        :param input: input data batch. Should be python iterable object.
        :param field: output field.
        """
        retv = None
        kwargs['input'] = input
        for result in self.iter_infer_field(field=field, **kwargs):
            if retv is None:
                retv = [[] for i in xrange(len(result))]
            for i, item in enumerate(result):
                retv[i].append(item)

        if retv == None:
            return []

        if flatten_result:
            retv = [numpy.concatenate(out) for out in retv]

        if len(retv) == 1:
            return retv[0]
        else:
            return retv


def infer(output_layer, parameters, input, feeding=None, field='value'):
    """
    Infer a neural network by given neural network output and parameters.  The
    user should pass either a batch of input data or reader method.

    Example usage for sinlge output_layer:

    ..  code-block:: python

        result = paddle.infer(output_layer=prediction,
                              parameters=parameters,
                              input=SomeData)
        print result

    Example usage for multiple outout_layers and fields:

    ..  code-block:: python

        result = paddle.infer(output_layer=[prediction1, prediction2],
                              parameters=parameters,
                              input=SomeData,
                              field=[id, value]])
        print result

    :param output_layer: output of the neural network that would be inferred
    :type output_layer: paddle.v2.config_base.Layer or a list of
                        paddle.v2.config_base.Layer
    :param parameters: parameters of the neural network.
    :type parameters: paddle.v2.parameters.Parameters
    :param input: input data batch. Should be a python iterable object, and each
                  element is the data batch.
    :type input: collections.Iterable
    :param feeding: Reader dictionary. Default could generate from input
                        value.
    :param field: The prediction field. It should in [`value`, `id`, `prob`].
                  `value` and `prob` mean return the prediction probabilities,
                  `id` means return the prediction labels. Default is `value`.
                  Note that `prob` only used when output_layer is beam_search
                  or max_id.
    :type field: str
    :return: The prediction result. If there are multiple outout_layers and fields,
             the return order is outout_layer1.field1, outout_layer2.field1, ...,
             outout_layer1.field2, outout_layer2.field2 ...
    :rtype: numpy.ndarray
    """

    inferer = Inference(output_layer=output_layer, parameters=parameters)
    return inferer.infer(field=field, input=input, feeding=feeding)

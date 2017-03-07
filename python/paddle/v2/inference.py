import numpy
import py_paddle.swig_paddle as api
import collections
import topology
import minibatch
from data_feeder import DataFeeder

__all__ = ['infer']


class Inference(object):
    def __init__(self, output, parameters):
        topo = topology.Topology(output)
        gm = api.GradientMachine.createFromConfigProto(
            topo.proto(), api.CREATE_MODE_TESTING, [api.PARAMETER_VALUE])
        for param in gm.getParameters():
            val = param.getBuf(api.PARAMETER_VALUE)
            name = param.getName()
            assert isinstance(val, api.Vector)
            val.copyFromNumpyArray(parameters.get(name).flatten())
        self.__gradient_machine__ = gm
        self.__data_types__ = topo.data_type()

    def iter_infer(self, input=None, batch_size=None, reader=None,
                   feeding=None):

        if reader is None:
            assert input is not None and isinstance(input, collections.Iterable)
            if not isinstance(input, collections.Iterable):
                raise TypeError("When reader is None, input should be whole "
                                "inference data and should be iterable")

            if batch_size is None:
                if not hasattr(input, '__len__'):
                    raise ValueError("Should set batch size when input data "
                                     "don't contain length.")
                batch_size = len(input)

            def __reader_impl__():
                for each_sample in input:
                    yield each_sample

            reader = minibatch.batch(__reader_impl__, batch_size=batch_size)
        else:
            if input is not None:
                raise ValueError("User should set either input or reader, "
                                 "should not set them both.")

        feeder = DataFeeder(self.__data_types__, feeding)
        self.__gradient_machine__.start()
        for data_batch in reader():
            yield self.__gradient_machine__.forwardTest(feeder(data_batch))
        self.__gradient_machine__.finish()

    def iter_infer_field(self, field, **kwargs):
        for result in self.iter_infer(**kwargs):
            yield [each_result[field] for each_result in result]

    def infer(self, field='value', **kwargs):
        retv = None
        for result in self.iter_infer_field(field=field, **kwargs):
            if retv is None:
                retv = [[]] * len(result)
            for i, item in enumerate(result):
                retv[i].append(item)
        retv = [numpy.concatenate(out) for out in retv]
        if len(retv) == 1:
            return retv[0]
        else:
            return retv


def infer(output,
          parameters,
          input=None,
          batch_size=None,
          reader=None,
          feeding=None,
          field='value'):
    """
    Infer a neural network by given neural network output and parameters.  The
    user should pass either a batch of input data or reader method.

    Example usages:

    ..  code-block:: python

        result = paddle.infer(prediction, parameters, input=SomeData,
                              batch_size=32)
        print result

    :param output: output of the neural network that would be inferred
    :type output: paddle.v2.config_base.Layer
    :param parameters: parameters of the neural network.
    :type parameters: paddle.v2.parameters.Parameters
    :param input: input data batch. Should be a python iterable object, and each
                  element is the data batch.
    :type input: collections.Iterable
    :param batch_size: the batch size when perform inference. Default is the
                       length of input.
    :type batch_size: int
    :param reader: input data reader creator in batch. If this field is set, the
                   `input` and `batch_size` will be ignored.
    :type reader: callable
    :param feeding: Reader dictionary. Default could generate from input
                        value.
    :param field: The prediction field. It should in [`value`, `ids`]. `value`
                  means return the prediction probabilities, `ids` means return
                  the prediction labels. Default is `value`
    :type field: str
    :return: a numpy array
    :rtype: numpy.ndarray
    """

    inferer = Inference(output=output, parameters=parameters)
    return inferer.infer(
        field=field,
        input=input,
        batch_size=batch_size,
        reader=reader,
        feeding=feeding)

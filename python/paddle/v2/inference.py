import numpy
import py_paddle.swig_paddle as api
import collections
import paddle.v2.topology as topology
import paddle.v2.minibatch as minibatch
import paddle.v2.data_feeder as data_feeder

__all__ = ['infer']


class Inference(object):
    def __init__(self, output_layer, parameters):
        topo = topology.Topology(output_layer)
        gm = api.GradientMachine.createFromConfigProto(
            topo.proto(), api.CREATE_MODE_TESTING, [api.PARAMETER_VALUE])
        for param in gm.getParameters():
            val = param.getBuf(api.PARAMETER_VALUE)
            name = param.getName()
            assert isinstance(val, api.Vector)
            val.copyFromNumpyArray(parameters.get(name).flatten())
        self.__gradient_machine__ = gm
        self.__data_types__ = topo.data_type()

    def iter_infer(self, input, feeding=None):
        feeder = data_feeder.DataFeeder(self.__data_types__, feeding)
        batch_size = len(input)

        def __reader_impl__():
            for each_sample in input:
                yield each_sample

        reader = minibatch.batch(__reader_impl__, batch_size=batch_size)

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


def infer(output_layer, parameters, input, feeding=None, field='value'):
    """
    Infer a neural network by given neural network output and parameters.  The
    user should pass either a batch of input data or reader method.

    Example usages:

    ..  code-block:: python

        result = paddle.infer(prediction, parameters, input=SomeData,
                              batch_size=32)
        print result

    :param output_layer: output of the neural network that would be inferred
    :type output_layer: paddle.v2.config_base.Layer
    :param parameters: parameters of the neural network.
    :type parameters: paddle.v2.parameters.Parameters
    :param input: input data batch. Should be a python iterable object, and each
                  element is the data batch.
    :type input: collections.Iterable
    :param feeding: Reader dictionary. Default could generate from input
                        value.
    :param field: The prediction field. It should in [`value`, `ids`]. `value`
                  means return the prediction probabilities, `ids` means return
                  the prediction labels. Default is `value`
    :type field: str
    :return: a numpy array
    :rtype: numpy.ndarray
    """

    inferer = Inference(output_layer=output_layer, parameters=parameters)
    return inferer.infer(field=field, input=input, feeding=feeding)

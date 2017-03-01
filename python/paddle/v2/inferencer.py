import py_paddle.swig_paddle as api

import topology
from data_feeder import DataFeeder
import itertools
import numpy

__all__ = ['InferenceEngine', 'infer']


class InferenceEngine(object):
    def __init__(self, output, parameters):
        topo = topology.Topology(output)
        gm = api.GradientMachine.createFromConfigProto(
            topo.proto(), api.CREATE_MODE_TESTING, [api.PARAMETER_VALUE])
        for param in gm.getParameters():
            val = param.getBuf(api.PARAMETER_VALUE)
            name = param.getName()
            assert isinstance(val, api.Matrix)
            val.copyFromNumpyMat(parameters.get(name))
        self.__gradient_machine__ = gm
        self.__data_types__ = topo.data_type()

    def iter_infer(self, reader, reader_dict=None):
        feeder = DataFeeder(self.__data_types__, reader_dict)
        out_args = api.Arguments.createArguments(0)
        self.__gradient_machine__.start()
        for data_batch in reader():
            yield self.__gradient_machine__.forwardTest(
                feeder(data_batch), out_args, api.PASS_TEST)
        self.__gradient_machine__.finish()

    def iter_infer_field(self, field, **kwargs):
        for result in self.iter_infer(**kwargs):
            yield [each_result[field] for each_result in result]

    def infer(self, field='value', **kwargs):
        retv = []
        for result in itertools.izip(
                self.iter_infer_field(
                    field=field, **kwargs)):
            retv.append(numpy.concatenate(result))
        return retv

    def default_reader_dict(self):
        reader_dict = dict()
        for i, tp in enumerate(self.__data_types__):
            reader_dict[tp[0]] = i
        return reader_dict


def infer(output, parameters, reader, reader_dict=None, field='value'):
    inferer = InferenceEngine(output=output, parameters=parameters)
    return inferer.infer(field=field, reader=reader, reader_dict=reader_dict)

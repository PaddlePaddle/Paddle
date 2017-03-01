import py_paddle.swig_paddle as api

import topology
from data_feeder import DataFeeder
import itertools
import numpy

from . import common

__all__ = ['Inference', 'infer']


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

    def iter_infer(self, reader, reader_dict=None):
        if reader_dict is None:
            reader_dict = self.default_reader_dict()
        feeder = DataFeeder(self.__data_types__, reader_dict)
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

    def default_reader_dict(self):
        reader_dict = dict()
        for i, tp in enumerate(self.__data_types__):
            reader_dict[tp[0]] = i
        return reader_dict


def infer(output,
          parameters,
          reader,
          batch_size,
          reader_dict=None,
          field='value'):
    reader = common.make_batch(reader, batch_size)
    inferer = Inference(output=output, parameters=parameters)
    return inferer.infer(field=field, reader=reader, reader_dict=reader_dict)

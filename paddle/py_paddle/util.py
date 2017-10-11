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
"""
Some Useful method for py_paddle.
"""

import swig_paddle
import os
import paddle.trainer.PyDataProviderWrapper
import paddle.proto.ParameterConfig_pb2
import paddle.proto.ModelConfig_pb2
import paddle.proto.TrainerConfig_pb2
import weakref
import numpy
import struct
import sys
import copy


def initializePaddle(*args):
    """
    To initialize paddle process.
    :param args: Command line options, such as --use_gpu=0, etc.
    :return: Nothing.
    """
    old_argv = copy.deepcopy(sys.argv)
    old_pypath = os.getenv("PYTHONPATH")
    pypath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if old_pypath is not None:
        pypath = os.pathsep.join([pypath, old_pypath])
        os.putenv("PYTHONPATH", pypath)
    args = [""] + list(args)  # argv[0] is command name, it is not important.
    swig_paddle.__initPaddle__(args)
    sys.argv = old_argv


def __monkeypatch_init_paddle__():
    swig_paddle.__initPaddle__ = swig_paddle.initPaddle
    swig_paddle.initPaddle = initializePaddle


class __ParameterCallbackWrapper__(swig_paddle.UpdateCallback):
    """
    Wrap the python callable object to paddle.UpdateCallback.

    INTERNAL USE ONLY.
    """

    def __init__(self, callback):
        swig_paddle.UpdateCallback.__init__(self)
        self.callback = callback

    def apply(self, param):
        self.callback(param)

    @staticmethod
    def wrap(callback):
        """
        Cast the python callable object/paddle.UpdateCallback to
        swig_paddle.UpdateCallback.__disown__
        :param callback: callable or swig_paddle.UpdateCallback object.
        """
        if isinstance(callback, swig_paddle.UpdateCallback):
            return callback.__disown__()
        elif isinstance(callback, weakref.ProxyType):
            raise RuntimeError("Should not pass __disown__ object")
        else:
            return __ParameterCallbackWrapper__(callback).__disown__()


def __arguments_to_numpy__(i, arg):
    assert isinstance(arg, swig_paddle.Arguments)
    value = arg.getSlotValue(i)
    ids = arg.getSlotIds(i)
    prob = arg.getSlotIn(i)
    if value is not None:
        assert isinstance(value, swig_paddle.Matrix)
        value = value.copyToNumpyMat()
    if ids is not None:
        assert isinstance(ids, swig_paddle.IVector)
        ids = ids.copyToNumpyArray()
    if prob is not None:
        assert isinstance(prob, swig_paddle.Matrix)
        prob = prob.copyToNumpyMat()
    return {"value": value, "id": ids, "prob": prob}


def __monkeypatch_gradient_machine__():
    """
    Add some class methods to GradientMachine.
    This method should be only used internally.
    """
    swig_paddle.GradientMachine.loadFromConfigFile = \
        staticmethod(loadGradientMachine)

    def __matrix_to_numpy__(m):
        if isinstance(m, swig_paddle.Matrix):
            return m.copyToNumpyMat()
        elif isinstance(m, swig_paddle.IVector):
            return m.copyToNumpyArra()
        else:
            raise RuntimeError("Input arg should be matrix or vecotr.")

    def createFromConfigProto(protoObj,
                              createMode=swig_paddle.CREATE_MODE_NORMAL,
                              paramTypes=[
                                  swig_paddle.PARAMETER_VALUE,
                                  swig_paddle.PARAMETER_GRADIENT,
                                  swig_paddle.PARAMETER_MOMENTUM
                              ]):
        """
        Create Gradient Machine From Proto object.
        :param protoObj: Model config
        :type protoObj: proto.ModelConfig_pb2.ModelConfig
        :param createMode: Create Mode, default is normal.
        :type createMode: int
        :param paramTypes: the gradient machine parameter type.
        :type paramTypes: list of int
        :return: paddle.GradientMachine
        """
        assert isinstance(protoObj, paddle.proto.ModelConfig)
        return swig_paddle.GradientMachine.createByConfigProtoStr(
            protoObj.SerializeToString(), createMode, paramTypes)

    swig_paddle.GradientMachine.createFromConfigProto = \
        staticmethod(createFromConfigProto)

    def forwardTest(self, inArgs):
        """
        forwardTest. forward gradient machine in test mode, and return a numpy
        matrix dict.

        :param inArgs: The input arguments
        :type inArgs: paddle.Arguments
        :return: A dictionary with keys ['id', 'value'], each value is a
                 numpy.ndarray.
        """
        outArgs = swig_paddle.Arguments.createArguments(0)
        self.forward(inArgs, outArgs, swig_paddle.PASS_TEST)
        return [
            __arguments_to_numpy__(i, outArgs)
            for i in xrange(outArgs.getSlotNum())
        ]

    swig_paddle.GradientMachine.forwardTest = forwardTest

    # Monkey patching backward
    swig_paddle.GradientMachine.__backward__ = swig_paddle.GradientMachine.backward

    def backward(self, callback):
        """
        GradientMachine Backward
        :param callback: a callback which parameter is (paddle.Parameter) or
                         a paddle.UpdateCallback object.
        """
        self.__backward__(__ParameterCallbackWrapper__.wrap(callback))

    swig_paddle.GradientMachine.backward = backward

    # Monkey patching forwardBackward.
    swig_paddle.GradientMachine.__forwardBackward__ = \
        swig_paddle.GradientMachine.forwardBackward

    def forwardBackward(self,
                        inArgs,
                        outArgs,
                        passType,
                        callback=swig_paddle.UpdateCallback()):
        """
        GradientMachine forward backward.
        :param inArgs: Input Arguments for GradientMachine.
        :type inArgs: paddle.Arguments
        :param outArgs: Output Arguments for GradientMachine.
        :type outArgs: paddle.Arguments
        :param passType: gradient machine's pass type.
        :type passType: paddle.PassType
        :param callback: a callable object with arguments (paddle.Parameter) or
                         a paddle.UpdateCallback it will be called when
                         backward
        """
        self.__forwardBackward__(inArgs, outArgs, passType,
                                 __ParameterCallbackWrapper__.wrap(callback))

    swig_paddle.GradientMachine.forwardBackward = forwardBackward

    def getParameters(self):
        return (self.getParameter(i) for i in xrange(self.getParameterSize()))

    swig_paddle.GradientMachine.getParameters = getParameters

    def getNonStaticParameters(self):
        return (self.getNonStaticParameter(i)
                for i in xrange(self.getNonStaticParameterSize()))

    swig_paddle.GradientMachine.getNonStaticParameters = getNonStaticParameters

    def getLayerOutputs(self, layerNames):
        """
        getLayerOutputs. get outputs of layers and return a numpy matrix dict.
        :param layerNames: layer names.
        :type layerNames: string or list.
        """
        if isinstance(layerNames, basestring):
            layerNames = [layerNames]
        elif not isinstance(layerNames, list):
            raise RuntimeError("Input args shuld be string or a sting list.")

        output = dict()
        for name in layerNames:
            output[name] = __arguments_to_numpy__(0, self.getLayerOutput(name))
        return output

    swig_paddle.GradientMachine.getLayerOutputs = getLayerOutputs


def loadGradientMachine(config_filename, model_dir=None):
    """
    Load a gradient machine from config file name/path.
    :param config_filename: The trainer config file name/path
    :param model_dir: The model parameter directory. None if same as the
    directory of config_filename
    :return: GradientMachine with some enhance methods.
    :rtype: paddle.GradientMachine
    """
    trainer_config = swig_paddle.TrainerConfig.createFromTrainerConfigFile(
        config_filename)
    assert isinstance(trainer_config, swig_paddle.TrainerConfig)
    model_conf = trainer_config.getModelConfig()
    network = swig_paddle.GradientMachine.createByModelConfig(model_conf)
    assert isinstance(network, swig_paddle.GradientMachine)
    if model_dir is None:
        model_dir = os.path.dirname(config_filename)
    network.loadParameters(model_dir)
    return network


def loadParameterFile(fn):
    """
    Load Paddle Parameter file to numpy.ndarray
    :param fn: file name or file like object.
    :type fn: str or file like object.
    :return: numpy array
    :rtype: numpy.ndarray
    :raise: paddle.UnsupportError when parameter format is wrong.
    """
    if isinstance(fn, str):
        with open(fn, 'rb') as f:
            return loadParameterFile(f)
    elif hasattr(fn, 'read'):  # File like object
        version, = struct.unpack('i', fn.read(4))
        if version != 0:
            raise swig_paddle.UnsupportError()
        value_length, = struct.unpack("I", fn.read(4))
        if value_length != 4 and value_length != 8:
            raise swig_paddle.UnsupportError()
        dtype = 'float32' if value_length == 4 else 'float64'
        param_size, = struct.unpack("L", fn.read(8))
        value = numpy.fromfile(fn, dtype)
        if len(value) != param_size:
            raise swig_paddle.UnsupportError()
        return value
    else:
        raise swig_paddle.UnsupportError()


class DataProviderWrapperConverter(object):
    """
    A class convert DataFormat from PyDataProvider Wrapper to
    py_paddle.paddle.Arguemnts.
    """

    class DenseValueConverter(object):
        """
        Internal class
        """

        def __init__(self, header_def):
            self.__dim__ = header_def.dim
            self.buf = []

        def append(self, other):
            assert len(other) == self.__dim__
            self.buf += other

        def __call__(self, slot_idx, arg):
            mat = swig_paddle.Matrix.createDense(self.buf,
                                                 len(self.buf) / self.__dim__,
                                                 self.__dim__)
            arg.setSlotValue(slot_idx, mat)

    class IdValueConverter(object):
        """
        Internal class
        """

        def __init__(self, *args):
            self.buf = []

        def append(self, other):
            assert isinstance(other, int)
            self.buf.append(other)

        def __call__(self, slot_idx, arg):
            arg.setSlotIds(slot_idx, swig_paddle.IVector.create(self.buf))

    class SparseNonValueConverter(object):
        """
        Internal class
        """

        def __init__(self, slot_def):
            self.indices = [0]
            self.cols = []
            self.dim = slot_def.dim

        def append(self, other):
            self.indices.append(self.indices[-1] + len(other))
            self.cols += other

        def __call__(self, slot_idx, arg):
            mat = swig_paddle.Matrix.createSparse(
                len(self.indices) - 1, self.dim, len(self.cols), True)
            assert isinstance(mat, swig_paddle.Matrix)
            mat.sparseCopyFrom(self.indices, self.cols)
            self.putIntoArg(slot_idx, arg, mat)

        def putIntoArg(self, slot_idx, arg, mat):
            arg.setSlotValue(slot_idx, mat)

    class SparseValueConverter(SparseNonValueConverter):
        """
        Internal class
        """

        def __init__(self, slot_def):
            super(DataProviderWrapperConverter.SparseValueConverter,
                  self).__init__(slot_def)
            self.values = []

        def append(self, other):
            super(DataProviderWrapperConverter.SparseValueConverter,
                  self).append(map(lambda x: x[0], other))
            self.values += map(lambda x: x[1], other)

        def __call__(self, slot_idx, arg):
            mat = swig_paddle.Matrix.createSparse(
                len(self.indices) - 1, self.dim, len(self.cols), False)
            assert isinstance(mat, swig_paddle.Matrix)
            mat.sparseCopyFrom(self.indices, self.cols, self.values)
            self.putIntoArg(slot_idx, arg, mat)

    __SLOT_VALUE_CONVERTER_MAP__ = {
        paddle.trainer.PyDataProviderWrapper.DenseSlot: DenseValueConverter,
        paddle.trainer.PyDataProviderWrapper.IndexSlot: IdValueConverter,
        paddle.trainer.PyDataProviderWrapper.SparseNonValueSlot:
        SparseNonValueConverter,
        paddle.trainer.PyDataProviderWrapper.SparseValueSlot:
        SparseValueConverter
    }

    def __init__(self, use_seq, header):
        """
        Ctor
        :param use_seq: True if use sequence.
        :param header:  List of slots type,
                       trainer.PyDataProviderWrapper.SlotType
        """
        self.__use_seq__ = use_seq
        self.__header__ = header

    def convert(self, wrapper_data, argument=None):
        """
        Convert PyDataProviderWrapper format to paddle.Argument
        :param wrapper_data: PyDataProviderWrapper yield's data list.
        :param argument: The output paddle.Arguments.
                        If it is not None, it will assign data in this
                        arguments, else it will create new arguments.
        :return: arguments that contains data.
        :rtype: paddle.Arguments
        """
        if argument is None:
            argument = swig_paddle.Arguments.createArguments(0)
        assert isinstance(argument, swig_paddle.Arguments)
        argument.resize(len(self.__header__))

        values = map(
            lambda x: DataProviderWrapperConverter.__SLOT_VALUE_CONVERTER_MAP__[x.__class__](x),
            self.__header__)

        if self.__use_seq__:
            seq_dim = [[] for _ in xrange(self.__header__.__len__())]
            seq_start_pos = [[0] for _ in xrange(self.__header__.__len__())]

            for each_sample in wrapper_data:
                for slot_idx, sequence in enumerate(each_sample):
                    for raw_data in sequence:
                        values[slot_idx].append(raw_data)
                    seq_start_pos[slot_idx].append(seq_start_pos[slot_idx][-1] +
                                                   len(sequence))
                    seq_dim[slot_idx].append(len(sequence))

            for slot_idx in xrange(len(self.__header__)):
                argument.setSlotSequenceDim(
                    slot_idx, swig_paddle.IVector.create(seq_dim[slot_idx]))
                argument.setSlotSequenceStartPositions(
                    slot_idx,
                    swig_paddle.IVector.create(seq_start_pos[slot_idx]))
        else:
            for each_sample in wrapper_data:
                for raw_data, value in zip(each_sample, values):
                    value.append(raw_data)

        for i, v in enumerate(values):
            v(i, argument)

        return argument

    def __call__(self, wrapper_data, argument=None):
        """
        Invoke self.convert. See documents in self.convert.
        """
        return self.convert(wrapper_data, argument)


def __monkey_patch_protobuf_objects__():
    def ParameterConfig_toProto(self):
        """
        Convert paddle.ParameterConfig to
        proto.ParameterConfig_pb2.ParameterConfig

        :return: proto.ParameterConfig_pb2.ParameterConfig object.
        """
        param_conf = paddle.proto.ParameterConfig_pb2.ParameterConfig()
        param_conf.ParseFromString(self.toProtoString())
        return param_conf

    swig_paddle.ParameterConfig.toProto = ParameterConfig_toProto

    def OptimizationConfig_toProto(self):
        """
        Convert paddle.OptimizationConfig to
        proto.TrainerConfig_pb2.OptimizationConfig

        :return: proto.TrainerConfig_pb2.OptimizationConfig
        """
        opt_conf = proto.TrainerConfig_pb2.OptimizationConfig()
        opt_conf.ParseFromString(self.toProtoString())
        return opt_conf

    swig_paddle.OptimizationConfig.toProto = OptimizationConfig_toProto

    def OptimizationConfig_createFromProto(protoObj):
        """
        Create a new paddle.OptimizationConfig from
        proto.TrainerConfig_pb2.OptimizationConfig

        :param protoObj: proto.TrainerConfig_pb2.OptimizationConfig
        :return: paddle.OptimizationConfig
        """

        assert isinstance(protoObj, paddle.proto.OptimizationConfig)
        return swig_paddle.OptimizationConfig.createFromProtoString(
            protoObj.SerializeToString())

    swig_paddle.OptimizationConfig.createFromProto = staticmethod(
        OptimizationConfig_createFromProto)

    def TrainerConfig_createFromProto(protoObj):
        """
        Create a new paddle.TrainerConfig from
        proto.OptimizationConfig

        :param protoObj: proto.TrainerConfig
        :return: paddle.TrainerConfig
        """
        assert isinstance(protoObj, paddle.proto.TrainerConfig)
        return swig_paddle.TrainerConfig.createFromProtoString(
            protoObj.SerializeToString())

    swig_paddle.TrainerConfig.createFromProto = staticmethod(
        TrainerConfig_createFromProto)


def __monkey_patch_parameter__():
    def getBufs(self):
        """
        get all parameter vectors.
        NOTE: the return value is a generator. Maybe you need to cast to
        list or tuple or something else.

        :return: generator of all parameter vectors.
        :rtype: generator
        """
        return (self.getBuf(i) for i in xrange(swig_paddle.NUM_PARAMETER_TYPES))

    swig_paddle.Parameter.getBufs = getBufs


def __monkey_patch_trainer__():
    swig_paddle.Trainer.__create__ = staticmethod(swig_paddle.Trainer.create)

    def Trainer_create(config, model=None):
        """
        Create a trainer for model with TrainerCOnfig trainer_config
        trainer_config.model_config will be ignored when model is supplied.
        Trainer.trainOneBatch() and Trainer.forwardOneBatch() can be used only
        when trainer_config.data_config is set.

        A typical usage for Trainer is:
        .. code-block:: python
           trainer = Trainer.create(trainer_config, model)
           for p in xrange(num_passes)
               while True:
                   data = get_next_batch(batch_size)
                   if not data:
                       break
                   trainer.trainOneDataBatch(batch_size, data)
               trainer.finishTrainPass()
           trainer.finishTrain()

        The trainer will take care of logging, model saving, distributed
        training, etc.

        :param config: trainer configuration
        :type config: paddle.proto.TrainerConfig
        :param model: the model to be trained
        :type model: swig_paddle.GradientMachine
        :return: a trainer
        :rtype swig_paddle.Trainer

        """
        assert isinstance(config, paddle.proto.TrainerConfig)
        if model is not None:
            assert isinstance(model, swig_paddle.GradientMachine)
        return swig_paddle.Trainer.__create__(
            swig_paddle.TrainerConfig.createFromProto(config), model)

    swig_paddle.Trainer.create = staticmethod(Trainer_create)

    swig_paddle.Trainer.__getForwardOutput__ = \
        swig_paddle.Trainer.getForwardOutput

    def getForwardOutput(self):
        """
        Get the netword outputs from the previous trainOneBatch(),
        trainOneDataBatch(), testOneDataPatch(), or forwardOneBatch() call.

        :return: list of dictionary with keys ['id', 'value'], each value is a
                 numpy.ndarray.
        """
        outArgs = self.__getForwardOutput__()
        return [
            __arguments_to_numpy__(i, outArgs)
            for i in xrange(outArgs.getSlotNum())
        ]

    swig_paddle.Trainer.getForwardOutput = getForwardOutput


def monkeypatches():
    patches = [
        __monkeypatch_init_paddle__, __monkeypatch_gradient_machine__,
        __monkey_patch_protobuf_objects__, __monkey_patch_parameter__,
        __monkey_patch_trainer__
    ]
    for patch in patches:
        patch()

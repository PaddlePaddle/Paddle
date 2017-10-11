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
import paddle.proto.ParameterConfig_pb2
import util
import unittest
import numpy


class TestGradientMachine(unittest.TestCase):
    def test_create_gradient_machine(self):
        conf_file_path = "./testTrainConfig.py"
        trainer_config = swig_paddle.TrainerConfig.createFromTrainerConfigFile(
            conf_file_path)
        self.assertIsNotNone(trainer_config)
        opt_config = trainer_config.getOptimizationConfig()
        model_config = trainer_config.getModelConfig()
        self.assertIsNotNone(model_config)
        machine = swig_paddle.GradientMachine.createByModelConfig(
            model_config, swig_paddle.CREATE_MODE_NORMAL,
            swig_paddle.ParameterOptimizer.create(opt_config).getParameterTypes(
            ))
        self.assertIsNotNone(machine)
        ipt, _ = util.loadMNISTTrainData()
        output = swig_paddle.Arguments.createArguments(0)

        optimizers = {}

        # Initial Machine Parameter all to 0.1
        for param in machine.getParameters():
            assert isinstance(param, swig_paddle.Parameter)
            val = param.getBuf(swig_paddle.PARAMETER_VALUE)
            assert isinstance(val, swig_paddle.Vector)
            arr = numpy.full((len(val), ), 0.1, dtype="float32")
            val.copyFromNumpyArray(arr)
            self.assertTrue(param.save(param.getName()))
            param_config = param.getConfig().toProto()
            assert isinstance(param_config,
                              paddle.proto.ParameterConfig_pb2.ParameterConfig)
            opt = swig_paddle.ParameterOptimizer.create(opt_config)
            optimizers[param.getID()] = opt
            num_rows = param_config.dims[1]
            opt.init(num_rows, param.getConfig())

        for k in optimizers:
            opt = optimizers[k]
            opt.startPass()

        batch_size = ipt.getSlotValue(0).getHeight()
        for k in optimizers:
            opt = optimizers[k]
            opt.startBatch(batch_size)

        machine.forward(ipt, output, swig_paddle.PASS_TRAIN)
        self.assertEqual(1, output.getSlotNum())
        self.isCalled = False

        def backward_callback(param_):
            self.isCalled = isinstance(param_, swig_paddle.Parameter)
            assert isinstance(param_, swig_paddle.Parameter)
            vec = param_.getBuf(swig_paddle.PARAMETER_VALUE)
            assert isinstance(vec, swig_paddle.Vector)
            vec = vec.copyToNumpyArray()
            for val_ in vec:
                self.assertTrue(
                    util.doubleEqual(val_, 0.1))  # Assert All Value is 0.1

            vecs = list(param_.getBufs())
            opt_ = optimizers[param_.getID()]
            opt_.update(vecs, param_.getConfig())

        machine.backward(backward_callback)

        for k in optimizers:
            opt = optimizers[k]
            opt.finishBatch()

        for k in optimizers:
            opt = optimizers[k]
            opt.finishPass()

        self.assertTrue(self.isCalled)

        for param in machine.getParameters():
            self.assertTrue(param.load(param.getName()))

    def test_train_one_pass(self):
        conf_file_path = './testTrainConfig.py'
        trainer_config = swig_paddle.TrainerConfig.createFromTrainerConfigFile(
            conf_file_path)
        model_config = trainer_config.getModelConfig()
        machine = swig_paddle.GradientMachine.createByModelConfig(model_config)

        at_end = False

        output = swig_paddle.Arguments.createArguments(0)
        if not at_end:
            input_, at_end = util.loadMNISTTrainData(1000)
            machine.forwardBackward(input_, output, swig_paddle.PASS_TRAIN)


if __name__ == '__main__':
    swig_paddle.initPaddle('--use_gpu=0')
    unittest.main()

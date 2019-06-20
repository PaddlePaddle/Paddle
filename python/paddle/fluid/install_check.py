# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
from . import core


def process_env():
    env = os.environ
    device_list = []
    if env.get('CUDA_VISIBLE_DEVICES') is not None:
        cuda_devices = env['CUDA_VISIBLE_DEVICES']
        if cuda_devices == "" or len(cuda_devices) == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
            device_list = [0, 1]
        elif len(cuda_devices) == 1:
            device_list.append(0)
        elif len(cuda_devices) > 1:
            for i in range(len(cuda_devices.split(","))):
                device_list.append(i)
        return device_list
    else:
        if core.get_cuda_device_count() > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
            return [0, 1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            return [0]


device_list = []
if core.is_compiled_with_cuda():
    device_list = process_env()
else:
    device_list = [0, 1]  # for CPU 0,1

from .framework import Program, program_guard, unique_name
from .param_attr import ParamAttr
from .initializer import Constant
from . import layers
from . import backward
from .dygraph import Layer, nn
from . import executor
from . import optimizer
from . import core
from . import compiler
import logging
import numpy as np

__all__ = ['run_check']


class SimpleLayer(Layer):
    def __init__(self, name_scope):
        super(SimpleLayer, self).__init__(name_scope)
        self._fc1 = nn.FC(self.full_name(),
                          3,
                          param_attr=ParamAttr(initializer=Constant(value=0.1)))

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = layers.reduce_sum(x)
        return x


def run_check():
    ''' intall check to verify if install is success

    This func should not be called only if you need to verify installation
    '''
    print("Running Verify Fluid Program ... ")
    use_cuda = False if not core.is_compiled_with_cuda() else True
    place = core.CPUPlace() if not core.is_compiled_with_cuda(
    ) else core.CUDAPlace(0)
    np_inp_single = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inp = []
    for i in range(len(device_list)):
        inp.append(np_inp_single)
    np_inp_muti = np.array(inp)
    np_inp_muti = np_inp_muti.reshape(len(device_list), 2, 2)

    def test_parallerl_exe():
        train_prog = Program()
        startup_prog = Program()
        scope = core.Scope()
        if not use_cuda:
            os.environ['CPU_NUM'] = "2"
        with executor.scope_guard(scope):
            with program_guard(train_prog, startup_prog):
                with unique_name.guard():
                    places = []
                    build_strategy = compiler.BuildStrategy()
                    build_strategy.enable_inplace = True
                    build_strategy.memory_optimize = True
                    inp = layers.data(name="inp", shape=[2, 2])
                    simple_layer = SimpleLayer("simple_layer")
                    out = simple_layer(inp)
                    exe = executor.Executor(place)
                    if use_cuda:
                        for i in device_list:
                            places.append(core.CUDAPlace(i))
                    else:
                        places = [core.CPUPlace(), core.CPUPlace()]
                    loss = layers.mean(out)
                    loss.persistable = True
                    optimizer.SGD(learning_rate=0.01).minimize(loss)
                    startup_prog.random_seed = 1
                    compiled_prog = compiler.CompiledProgram(
                        train_prog).with_data_parallel(
                            build_strategy=build_strategy,
                            loss_name=loss.name,
                            places=places)
                    exe.run(startup_prog)

                    exe.run(compiled_prog,
                            feed={inp.name: np_inp_muti},
                            fetch_list=[loss.name])

    def test_simple_exe():
        train_prog = Program()
        startup_prog = Program()
        scope = core.Scope()
        if not use_cuda:
            os.environ['CPU_NUM'] = "1"
        with executor.scope_guard(scope):
            with program_guard(train_prog, startup_prog):
                with unique_name.guard():
                    inp0 = layers.data(
                        name="inp", shape=[2, 2], append_batch_size=False)
                    simple_layer0 = SimpleLayer("simple_layer")
                    out0 = simple_layer0(inp0)
                    param_grads = backward.append_backward(
                        out0, parameter_list=[simple_layer0._fc1._w.name])[0]
                    exe0 = executor.Executor(core.CPUPlace()
                                             if not core.is_compiled_with_cuda()
                                             else core.CUDAPlace(0))
                    exe0.run(startup_prog)
                    exe0.run(feed={inp0.name: np_inp_single},
                             fetch_list=[out0.name, param_grads[1].name])

    test_simple_exe()

    print("Your Paddle Fluid works well on SINGLE GPU or CPU.")
    try:
        test_parallerl_exe()
        print("Your Paddle Fluid works well on MUTIPLE GPU or CPU.")
        print(
            "Your Paddle Fluid is installed successfully! Let's start deep Learning with Paddle Fluid now"
        )
    except Exception as e:
        logging.warning(
            "Your Paddle Fluid has some problem with multiple GPU. This may be caused by:"
            "\n 1. There is only 1 GPU visible on your Device;"
            "\n 2. No.1 or No.2 GPU or both of them are occupied now"
            "\n 3. Wrong installation of NVIDIA-NCCL2, please follow instruction on https://github.com/NVIDIA/nccl-tests "
            "\n to test your NCCL, or reinstall it following https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html"
        )

        print("\n Original Error is: {}".format(e))
        print(
            "Your Paddle Fluid is installed successfully ONLY for SINGLE GPU or CPU! "
            "\n Let's start deep Learning with Paddle Fluid now")

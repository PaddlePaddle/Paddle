# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import numpy as np

import paddle

__all__ = []


def _simple_network():
    """
    Define a simple network composed by a single linear layer.
    """
    input = paddle.static.data(
        name="input", shape=[None, 2, 2], dtype="float32"
    )
    weight = paddle.create_parameter(
        shape=[2, 3],
        dtype="float32",
        attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.1)),
    )
    bias = paddle.create_parameter(shape=[3], dtype="float32")
    linear_out = paddle.nn.functional.linear(x=input, weight=weight, bias=bias)
    out = paddle.tensor.sum(linear_out)
    return input, out, weight


def _prepare_data():
    """
    Prepare feeding data for simple network. The shape is [1, 2, 2].

    """
    # Prepare the feeding data.
    np_input_single = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    return np_input_single.reshape(1, 2, 2)


def _is_cuda_available():
    """
    Check whether CUDA is available.
    """
    try:
        assert len(paddle.static.cuda_places()) > 0
        return True
    except Exception as e:
        logging.warning(
            "You are using GPU version PaddlePaddle, but there is no GPU "
            "detected on your machine. Maybe CUDA devices is not set properly."
            f"\n Original Error is {e}"
        )
        return False


def _is_xpu_available():
    """
    Check whether XPU is available.
    """
    try:
        assert len(paddle.static.xpu_places()) > 0
        return True
    except Exception as e:
        logging.warning(
            "You are using XPU version PaddlePaddle, but there is no XPU "
            "detected on your machine. Maybe XPU devices is not set properly."
            f"\n Original Error is {e}"
        )
        return False


def _run_dygraph_single(use_cuda, use_xpu, use_custom, custom_device_name):
    """
    Testing the simple network in dygraph mode using one CPU/GPU/XPU.

    Args:
        use_cuda (bool): Whether running with CUDA.
        use_xpu (bool): Whether running with XPU.
    """
    paddle.disable_static()
    if use_cuda:
        paddle.set_device('gpu')
    elif use_xpu:
        paddle.set_device('xpu')
    elif use_custom:
        paddle.set_device(custom_device_name)
    else:
        paddle.set_device('cpu')
    weight_attr = paddle.ParamAttr(
        name="weight", initializer=paddle.nn.initializer.Constant(value=0.5)
    )
    bias_attr = paddle.ParamAttr(
        name="bias", initializer=paddle.nn.initializer.Constant(value=1.0)
    )
    linear = paddle.nn.Linear(
        2, 4, weight_attr=weight_attr, bias_attr=bias_attr
    )
    input_np = _prepare_data()
    input_tensor = paddle.to_tensor(input_np)
    linear_out = linear(input_tensor)
    out = paddle.tensor.sum(linear_out)
    out.backward()
    opt = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=linear.parameters()
    )
    opt.step()


def _run_static_single(use_cuda, use_xpu, use_custom, custom_device_name):
    """
    Testing the simple network with executor running directly, using one CPU/GPU/XPU.

    Args:
        use_cuda (bool): Whether running with CUDA.
        use_xpu (bool): Whether running with XPU.
    """
    paddle.enable_static()
    with paddle.static.scope_guard(paddle.static.Scope()):
        train_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        startup_prog.random_seed = 1
        with paddle.static.program_guard(train_prog, startup_prog):
            input, out, weight = _simple_network()
            param_grads = paddle.static.append_backward(
                out, parameter_list=[weight.name]
            )[0]

        if use_cuda:
            place = paddle.CUDAPlace(0)
        elif use_xpu:
            place = paddle.XPUPlace(0)
        elif use_custom:
            place = paddle.CustomPlace(custom_device_name, 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)
        exe.run(
            train_prog,
            feed={input.name: _prepare_data()},
            fetch_list=[out.name, param_grads[1].name],
        )
    paddle.disable_static()


def train_for_run_parallel():
    """
    train script for parallel training check
    """

    # to avoid cyclic import
    class LinearNet(paddle.nn.Layer):
        """
        simple fc network for parallel training check
        """

        def __init__(self):
            super().__init__()
            self._linear1 = paddle.nn.Linear(10, 10)
            self._linear2 = paddle.nn.Linear(10, 1)

        def forward(self, x):
            """
            forward
            """
            return self._linear2(self._linear1(x))

    paddle.distributed.init_parallel_env()

    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = paddle.nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=dp_layer.parameters()
    )

    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    loss.backward()
    adam.step()
    adam.clear_grad()


def _run_parallel(device_list):
    """
    Testing the simple network in data parallel mode, using multiple CPU/GPU.

    Args:
        use_cuda (bool): Whether running with CUDA.
        use_xpu (bool): Whether running with XPU.
        device_list (int): The specified devices.
    """
    paddle.distributed.spawn(train_for_run_parallel, nprocs=len(device_list))


def run_check():
    """
    Check whether PaddlePaddle is installed correctly and running successfully
    on your system.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.utils.run_check()
            >>> # doctest: +SKIP('the output will change in different run')
            Running verify PaddlePaddle program ...
            I0818 15:35:08.335391 30540 program_interpreter.cc:173] New Executor is Running.
            I0818 15:35:08.398319 30540 interpreter_util.cc:529] Standalone Executor is Used.
            PaddlePaddle works well on 1 CPU.
            PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
    """

    print("Running verify PaddlePaddle program ... ")

    use_cuda = False
    use_xpu = False
    use_custom = False
    custom_device_name = None

    if paddle.is_compiled_with_cuda():
        use_cuda = _is_cuda_available()
    elif paddle.is_compiled_with_xpu():
        use_xpu = _is_xpu_available()
    elif len(paddle.framework.core.get_all_custom_device_type()) > 0:
        use_custom = True
        if len(paddle.framework.core.get_all_custom_device_type()) > 1:
            logging.warning(
                f"More than one kind of custom devices detected, but run check would only be executed on {paddle.framework.core.get_all_custom_device_type()[0]}."
            )

    if use_cuda:
        device_str = "GPU"
        device_list = paddle.static.cuda_places()
    elif use_xpu:
        device_str = "XPU"
        device_list = paddle.static.xpu_places()
    elif use_custom:
        device_str = paddle.framework.core.get_all_custom_device_type()[0]
        custom_device_name = device_str
        device_list = list(
            range(
                paddle.framework.core.get_custom_device_count(
                    custom_device_name
                )
            )
        )
    else:
        device_str = "CPU"
        device_list = paddle.static.cpu_places(device_count=1)
    device_count = len(device_list)

    _run_static_single(use_cuda, use_xpu, use_custom, custom_device_name)
    _run_dygraph_single(use_cuda, use_xpu, use_custom, custom_device_name)
    print(f"PaddlePaddle works well on 1 {device_str}.")

    try:
        if len(device_list) > 1:
            if use_custom:
                import os

                os.environ['PADDLE_DISTRI_BACKEND'] = "xccl"
            _run_parallel(device_list)
            print(f"PaddlePaddle works well on {device_count} {device_str}s.")
        print(
            "PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now."
        )
    except Exception as e:
        logging.warning(
            f"PaddlePaddle meets some problem with {device_count} {device_str}s. This may be caused by:"
            "\n 1. There is not enough GPUs visible on your system"
            "\n 2. Some GPUs are occupied by other process now"
            "\n 3. NVIDIA-NCCL2 is not installed correctly on your system. Please follow instruction on https://github.com/NVIDIA/nccl-tests "
            "\n to test your NCCL, or reinstall it following https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html"
        )

        logging.warning(f"\n Original Error is: {e}")
        print(
            f"PaddlePaddle is installed successfully ONLY for single {device_str}! "
            "Let's start deep learning with PaddlePaddle now."
        )
        raise e

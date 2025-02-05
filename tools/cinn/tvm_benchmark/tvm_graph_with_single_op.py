# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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


import numpy as np
import tvm
import tvm.contrib.graph_runtime as runtime
import tvm.relay.testing
from tvm import relay

# To test different ops, change this single-op network.
# See https://github.com/apache/incubator-tvm/blob/main/docs/langref/relay_op.rst to get the op list.


def get_network_conv2d():
    input_shape = [(2, 512, 7, 7), (512, 512, 3, 3)]
    output_shape = (2, 512, 7, 7)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.nn.conv2d")
    mod = relay.Function(
        [x, y],
        relay.nn.conv2d(
            x, y, kernel_size=(3, 3), padding=(1, 1), strides=(1, 1)
        ),
    )
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_conv2d_resnet1():
    input_shape = [(2, 3, 224, 224), (64, 3, 7, 7)]
    output_shape = (2, 64, 112, 112)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.nn.conv2d resnet1")
    mod = relay.Function(
        [x, y],
        relay.nn.conv2d(
            x, y, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2)
        ),
    )
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_conv2d_resnet2():
    input_shape = [(2, 64, 56, 56), (64, 64, 3, 3)]
    output_shape = (2, 64, 56, 56)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.nn.conv2d resnet2")
    mod = relay.Function(
        [x, y],
        relay.nn.conv2d(
            x, y, kernel_size=(3, 3), padding=(1, 1), strides=(1, 1)
        ),
    )
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_conv2d_resnet3():
    input_shape = [(2, 64, 56, 56), (64, 64, 1, 1)]
    output_shape = (2, 64, 56, 56)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.nn.conv2d resnet2")
    mod = relay.Function(
        [x, y],
        relay.nn.conv2d(
            x, y, kernel_size=(1, 1), padding=(0, 0), strides=(1, 1)
        ),
    )
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_conv2d_resnet4():
    input_shape = [(2, 64, 56, 56), (128, 64, 1, 1)]
    output_shape = (2, 128, 28, 28)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.nn.conv2d resnet2")
    mod = relay.Function(
        [x, y],
        relay.nn.conv2d(
            x, y, kernel_size=(1, 1), padding=(0, 0), strides=(2, 2)
        ),
    )
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_conv2d_resnet5():
    input_shape = [(2, 128, 28, 28), (256, 128, 3, 3)]
    output_shape = (2, 256, 14, 14)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.nn.conv2d resnet2")
    mod = relay.Function(
        [x, y],
        relay.nn.conv2d(
            x, y, kernel_size=(3, 3), padding=(1, 1), strides=(2, 2)
        ),
    )
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_relu():
    input_shape = [(2, 512, 112, 112)]
    output_shape = (2, 512, 112, 112)
    input_names = ["x"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    print("[Test]Begin building graph with op relay.nn.relu")
    mod = relay.Function([x], relay.nn.relu(x))
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_elementwise():
    input_shape = [(64, 64), (64, 64)]
    output_shape = (64, 64)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.multiply")
    mod = relay.Function([x, y], relay.multiply(x, y))
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_matmul():
    input_shape = [(32, 32), (32, 32)]
    output_shape = (32, 32)
    input_names = ["x", "y"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    y = relay.Var(input_names[1], tvm.relay.TensorType(input_shape[1]))
    print("[Test]Begin building graph with op relay.nn.dense (matmul)")
    mod = relay.Function([x, y], relay.nn.dense(x, y))
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_softmax():
    input_shape = [(1024, 2048)]
    output_shape = (1024, 2048)
    input_names = ["x"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    print("[Test]Begin building graph with op relay.nn.softmax")
    mod = relay.Function([x], relay.nn.softmax(x))
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_pool2d():
    input_shape = [(2, 64, 112, 112)]
    output_shape = (2, 64, 56, 56)
    input_names = ["x"]
    x = relay.Var(input_names[0], tvm.relay.TensorType(input_shape[0]))
    print("[Test]Begin building graph with op relay.nn.max_pool2d")
    mod = relay.Function(
        [x],
        relay.nn.max_pool2d(
            x, pool_size=(3, 3), strides=(2, 2), padding=(1, 1)
        ),
    )
    params = []
    return mod, params, input_shape, output_shape, input_names


def get_network_batchnorm():
    data0 = relay.var("data0", relay.TensorType((2, 512, 32, 32), "float32"))
    bn_gamma = relay.var("bn_gamma1", relay.TensorType((512,), "float32"))
    bn_beta = relay.var("bn_beta1", relay.TensorType((512,), "float32"))
    bn_mmean = relay.var("bn_mean1", relay.TensorType((512,), "float32"))
    bn_mvar = relay.var("bn_var1", relay.TensorType((512,), "float32"))
    bn = relay.nn.batch_norm(data0, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    input_shape = [(2, 512, 32, 32), (512), (512), (512), (512)]
    output_shape = (2, 512, 32, 32)
    input_names = ["data0", "bn_gamma1", "bn_beta1", "bn_mean1", "bn_var1"]
    print("[Test]Begin building graph with op relay.nn.batch_norm")
    mod = relay.Function([data0, bn_gamma, bn_beta, bn_mmean, bn_mvar], bn)
    params = []
    return mod, params, input_shape, output_shape, input_names


##################################################################
# For CUDA backends, use
# :code:`target = "cuda"`
# For X86 backends, use
# :code:`target = "llvm"`
target = "cuda"
dtype = "float32"


def tune_and_evaluate(func):
    # extract workloads from relay program
    mod, params, input_shape, out_shape, input_names = func()

    runtime_mod = relay.build_module.build(mod, target=target)
    print("-----GPU code-----")
    print(runtime_mod.get_lib().imported_modules[0].get_source())
    # load parameters
    ctx = tvm.context(str(target), 0)
    module = runtime.GraphModule(runtime_mod["default"](ctx))
    for index in range(len(input_shape)):
        data_temp = tvm.nd.array(
            (np.random.uniform(size=input_shape[index])).astype(dtype)
        )
        module.set_input(input_names[index], data_temp)
    # evaluate
    evaluator_preheat = module.module.time_evaluator(
        "run", ctx, number=10, repeat=10
    )
    evaluator = module.module.time_evaluator("run", ctx, number=100, repeat=10)

    prof_res1 = (
        np.array(evaluator_preheat().results) * 1000
    )  # convert to millisecond
    print(
        f"[PreHeat]Mean inference time (std dev): {np.mean(prof_res1):.4f} ms ({np.std(prof_res1):.4f} ms)"
    )

    prof_res2 = np.array(evaluator().results) * 1000  # convert to millisecond
    print(
        f"[Benchmark]Mean inference time (std dev): {np.mean(prof_res2):.4f} ms ({np.std(prof_res2):.4f} ms)"
    )


# tune_and_evaluate(get_network_pool2d)
# tune_and_evaluate(get_network_softmax)
# tune_and_evaluate(get_network_matmul)
# tune_and_evaluate(get_network_batchnorm)
tune_and_evaluate(get_network_relu)
# tune_and_evaluate(get_network_elementwise)
# tune_and_evaluate(get_network_conv2d_resnet1)
# tune_and_evaluate(get_network_conv2d_resnet2)
# tune_and_evaluate(get_network_conv2d_resnet3)
# tune_and_evaluate(get_network_conv2d_resnet4)
# tune_and_evaluate(get_network_conv2d_resnet5)
# tune_and_evaluate(get_network_conv2d)

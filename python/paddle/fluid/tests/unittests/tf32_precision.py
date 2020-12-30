#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import functools
import os
import paddle
import paddle.fluid.core as core
import pycuda.driver as drv
import re


# Get compute capability of the current CUDA device
def get_compute_capability(id):
    drv.init()
    gpu_device = drv.Device(id)
    compute_capability = gpu_device.compute_capability()
    return compute_capability[0] * 10 + compute_capability[1]


# Get cuda runtime version from source file
def get_cuda_runtime_version():
    cuda_path = os.popen("whereis cuda").read()
    cuda_path = cuda_path[6:-1]
    cuda_path += '/include/cuda.h'
    try:
        cuda_version = os.popen("cat " + cuda_path +
                                " | grep '\#define CUDA_VERSION\'").read()
    except ValueError:
        print("No such file or directory")
    else:
        cuda_version = cuda_version[:-1]
        cuda_ver = re.split(' ', cuda_version)
        return int(cuda_ver[-1])


# Check if tf32 is supported on current hardware
# returns true if cuda >= 11 && arch >= ampere
def tf32_is_not_fp32():
    if not core.is_compiled_with_cuda():
        return False
    if core.get_cuda_device_count() == 0:
        return False
    if get_compute_capability(0) < 80:
        return False
    if get_cuda_runtime_version() < 11000:
        return False
    return True


# test tf32 off-mode
@contextlib.contextmanager
def tf32_off():
    old_allow_tf32_matmul = core.get_cublas_switch()
    try:
        core.set_cublas_switch(False)
        yield
    finally:
        core.set_cublas_switch(old_allow_tf32_matmul)


# test tf32 on-mode
@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    old_allow_tf32_matmul = core.get_cublas_switch()
    old_precision = self.precision
    try:
        core.set_cublas_switch(True)
        self.precision = tf32_precision
        yield
    finally:
        core.set_cublas_switch(old_allow_tf32_matmul)
        self.precision = old_precision


#Specify the precision
#example:
#@tf32_on_and_off(0.005)
#def test_matmul(self):
#    data1 = paddle.to_tensor(a)
#    data2 = paddle.to_tensor(b)
#    c = paddle.matmul(data1, data2)
#    self.assertTrue(c, expected, self.precision)
#
#When its tf32 on-mode, assertTrue will compare with reduced precison
def tf32_on_and_off(tf32_precision=1e-5):
    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(self):
            if tf32_is_not_fp32():
                with_tf32_disabled(self, lambda: func(self))
                with_tf32_enabled(self, lambda: func(self))
            else:
                func(self)

        return wrapped

    return wrapper

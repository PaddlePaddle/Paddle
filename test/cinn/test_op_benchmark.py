#!/usr/bin/env python3

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

import sys
import unittest

import numpy as np
from cinn.common import DefaultHostTarget, DefaultNVGPUTarget, Float
from cinn.frontend import Program, Variable

import paddle
from paddle import static

assert len(sys.argv) == 2
enable_gpu = sys.argv.pop()


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        paddle.enable_static()

        a = static.data(name='A', shape=[1, 128, 28, 28], dtype='float32')
        e = paddle.nn.initializer.NumpyArrayInitializer(
            np.array(result[1]).reshape((256, 128, 1, 1)).astype("float32")
        )
        res = static.nn.conv2d(
            input=a,
            num_filters=256,
            filter_size=1,
            stride=2,
            padding=0,
            dilation=1,
            param_attr=e,
        )

        exe = static.Executor(paddle.CPUPlace())
        exe.run(static.default_startup_program())

        x = np.array(result[0]).reshape((1, 128, 28, 28)).astype("float32")
        output = exe.run(feed={"A": x}, fetch_list=[res])
        output = np.array(output).reshape(-1)
        print("result in conv2d paddle_verify: \n")
        for i in range(0, output.shape[0]):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                print(
                    "Error! ",
                    i,
                    "-th data has diff with target data:\n",
                    output[i],
                    " vs: ",
                    result[len(result) - 1][i],
                    ". Diff is: ",
                    output[i] - result[len(result) - 1][i],
                )
        np.testing.assert_allclose(result[len(result) - 1], output, atol=1e-4)

    def atest_conv2d_cinn(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([1, 128, 28, 28])
        b = Variable("E").set_type(Float(32)).set_shape([256, 128, 1, 1])
        c = prog.conv2d(
            a, b, {"stride": [2, 2], "dilation": [1, 1], "padding": [0, 0]}
        )
        tensor_data = [
            np.random.random([1, 128, 28, 28]).astype("float32"),
            np.random.random([256, 128, 1, 1]).astype("float32"),
        ]
        result = prog.test_benchmark(
            self.target,
            [a, b],
            tensor_data,
            c,
            20000,
            "TESTING [conv2d] time cost with shape [1, 128, 28, 28]...",
        )
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)

    def atest_conv2d_cinn_code(self):
        prog = Program()
        a = Variable("X").set_type(Float(32)).set_shape([1, 128, 28, 28])
        b = Variable("Y").set_type(Float(32)).set_shape([256, 128, 1, 1])
        c = prog.conv2d(
            a, b, {"stride": [2, 2], "dilation": [1, 1], "padding": [0, 0]}
        )
        tensor_data = [
            np.random.random([1, 128, 28, 28]).astype("float32"),
            np.random.random([256, 128, 1, 1]).astype("float32"),
        ]
        result = prog.test_benchmark_with_code(
            self.target,
            [a, b],
            tensor_data,
            c,
            20000,
            "TESTING [conv2d of tvm schedule] time cost with shape [1, 128, 28, 28]...",
            """
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void fn_conv2d_0_kernel(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ COD)
{
  __shared__ float _input_pad_0_read_cache [ 224 ];
  float _COD_write_cache [ 2 ];
  __shared__ float _Y_read_cache [ 256 ];
  float* COD_write_cache = _COD_write_cache;
  float* COD_write_cache__reduce_init = _COD_write_cache;
  float* Y_read_cache = _Y_read_cache;
  float* input_pad_0_read_cache = _input_pad_0_read_cache;
  if ((blockIdx.z < 8)) {
    if ((blockIdx.y < 14)) {
      if ((threadIdx.z < 16)) {
        if ((threadIdx.x < 14)) {
        {
          for (int32_t rc_outer = 0; rc_outer < 2; rc_outer += 1) {
            COD_write_cache__reduce_init[rc_outer] = 0;
          };
          for (int32_t rc_outer = 0; rc_outer < 16; rc_outer += 1) {
            {
              __syncthreads();
              if ((threadIdx.z < 8)) {
                input_pad_0_read_cache[((2 * threadIdx.x) + (28 * threadIdx.z))] = X[((56 * blockIdx.y) + ((6272 * rc_outer) + ((2 * threadIdx.x) + (784 * threadIdx.z))))];
              };
            };
            for (int32_t rc_inner = 0; rc_inner < 2; rc_inner += 1) {
              if ((threadIdx.x < 8)) {
                Y_read_cache[((threadIdx.x / 2) + ((8 * (threadIdx.x % 2)) + ((4 * rc_inner) + (16 * threadIdx.z))))] = Y[((threadIdx.x / 2) + ((128 * (threadIdx.x % 2)) + ((4096 * blockIdx.z) + ((4 * rc_inner) + ((8 * rc_outer) + (256 * threadIdx.z))))))];
              };
            };
            __syncthreads();
            for (int32_t rc_inner = 0; rc_inner < 8; rc_inner += 1) {
              for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
                COD_write_cache[j_inner] = (COD_write_cache[j_inner] + (input_pad_0_read_cache[((28 * rc_inner) + (2 * threadIdx.x))] * Y_read_cache[((8 * j_inner) + ((16 * threadIdx.z) + rc_inner))]));
              };
            };
          };
          for (int32_t rc_outer = 0; rc_outer < 2; rc_outer += 1) {
            COD[((14 * blockIdx.y) + ((6272 * blockIdx.z) + ((196 * rc_outer) + ((392 * threadIdx.z) + threadIdx.x))))] = COD_write_cache[rc_outer];
          };
        }
        };
      };
    };
  };
}

}
            """,
        )
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)

    def atest_conv2d_tvm_code(self):
        prog = Program()
        a = (
            Variable("placeholder")
            .set_type(Float(32))
            .set_shape([1, 128, 28, 28])
        )
        b = (
            Variable("placeholder1")
            .set_type(Float(32))
            .set_shape([256, 128, 1, 1])
        )
        c = prog.conv2d(
            a, b, {"stride": [2, 2], "dilation": [1, 1], "padding": [0, 0]}
        )
        tensor_data = [
            np.random.random([1, 128, 28, 28]).astype("float32"),
            np.random.random([256, 128, 1, 1]).astype("float32"),
        ]
        result = prog.test_benchmark_with_code(
            self.target,
            [a, b],
            tensor_data,
            c,
            20000,
            "TESTING [conv2d of tvm schedule] time cost with shape [1, 128, 28, 28]...",
            """
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__ void fn_conv2d_0_kernel(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ Conv2d_nchw_out) {
  float compute_local[2];
  __shared__ float pad_temp_shared[216];
  __shared__ float placeholder_shared[256];
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    compute_local[(ff_c_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) < 216) {
      pad_temp_shared[(((((int)threadIdx.z) * 14) + ((int)threadIdx.x)))] = placeholder[(((((rc_outer * 6272) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) / 27) * 784)) + (((int)blockIdx.y) * 56)) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) % 27)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 3)) < 32) {
        if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 256) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 16) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 3) * 128)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int ff_c = 0; ff_c < 2; ++ff_c) {
        compute_local[(ff_c)] = (compute_local[(ff_c)] + (pad_temp_shared[(((rc_inner * 27) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner))]));
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
    Conv2d_nchw_out[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (ff_inner_inner_inner * 196)) + (((int)blockIdx.y) * 14)) + ((int)threadIdx.x)))] = compute_local[(ff_inner_inner_inner)];
  }
}

}
            """,
        )
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)

    def atest_softmax(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([1024, 2048])
        c = prog.softmax(a, {})
        tensor_data = [np.random.random([1024, 2048]).astype("float32")]
        result = prog.test_benchmark(
            self.target,
            [a],
            tensor_data,
            c,
            200,
            "TESTING [softmax] time cost with shape [1024,2048]...",
        )

    def atest_matmul(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([512, 512])
        b = Variable("B").set_type(Float(32)).set_shape([512, 512])
        c = prog.mul(a, b, 1, 1)
        tensor_data = [
            np.random.random([512, 512]).astype("float32"),
            np.random.random([512, 512]).astype("float32"),
        ]
        result = prog.test_benchmark(
            self.target,
            [a, b],
            tensor_data,
            c,
            200,
            "TESTING [matmul] time cost with shape [512,512]...",
        )

    def atest_matmul2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([128, 512])
        b = Variable("B").set_type(Float(32)).set_shape([256, 512])
        c = Variable("C").set_type(Float(32)).set_shape([128, 256])
        d = prog.mul(a, b, 1, 1)
        e = prog.add(d, c)
        tensor_data = [
            np.random.random([128, 512]).astype("float32"),
            np.random.random([256, 512]).astype("float32"),
            np.random.random([128, 256]).astype("float32"),
        ]
        result = prog.test_benchmark(
            self.target,
            [a, b, c],
            tensor_data,
            e,
            200,
            "TESTING [mul and add] time cost with shape [128,512]*[256,512]...",
        )

    def atest_matmul3(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([512, 512])
        b = Variable("B").set_type(Float(32)).set_shape([512, 512])
        c = Variable("C").set_type(Float(32)).set_shape([512, 512])
        d = prog.mul(a, b, 1, 1)
        # e = prog.add(d, c)
        tensor_data = [
            np.random.random([512, 512]).astype("float32"),
            np.random.random([512, 512]).astype("float32"),
        ]
        result = prog.test_benchmark_with_code(
            self.target,
            [a, b],
            tensor_data,
            d,
            200,
            "TESTING [matmul] time cost with shape [512,512]...",
            '''
            extern "C" {
#include "cinn_cuda_runtime_source.cuh"
#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif

 __global__
 void fn_mul_0_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ Mul_output)
 {
   const float* A_reshape = A;
   const float* B_reshape = B;
   float* Mul_output__reduce_init = Mul_output;
   if ((blockIdx.x < 512)) {
   {
     if ((threadIdx.x < 256)) {
     {
       for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
         Mul_output__reduce_init[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))] = 0;
       };
     }
     };
   }
   };
   if ((blockIdx.x < 512)) {
   {
     if ((threadIdx.x < 256)) {
     {
       for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
        for (int32_t axis_k = 0; axis_k < 512; axis_k += 1) {
          Mul_output[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))] = (Mul_output[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))] + (A_reshape[((512 * blockIdx.x) + axis_k)] * B_reshape[((512 * axis_k) + ((2 * threadIdx.x) + j_inner))])) + Mul_output[((512 * blockIdx.x) + ((2 * threadIdx.x) + j_inner))];
         };
       };
     }
     };
  }
  };
 }
 }''',
        )

    def atest_pool2d(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 64, 112, 112])
        c = prog.pool2d(
            a,
            {
                "kernel_size": (3, 3),
                "stride_size": (2, 2),
                "padding_size": (1, 1, 1, 1),
                "pool_type": "max",
            },
        )
        tensor_data = [np.random.random([2, 64, 112, 112]).astype("float32")]
        result = prog.test_benchmark(
            self.target,
            [a],
            tensor_data,
            c,
            2000,
            "TESTING [pool2d] time cost with shape [2, 64, 112, 112]...",
        )

    def atest_elementwise1(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([64, 64])
        b = Variable("B").set_type(Float(32)).set_shape([64, 64])
        c = prog.add(a, b)
        tensor_data = [
            np.random.random([64, 64]).astype("float32"),
            np.random.random([64, 64]).astype("float32"),
        ]
        result = prog.test_benchmark(
            self.target,
            [a, b],
            tensor_data,
            c,
            200,
            "TESTING [elementwise_add] time cost with shape [64, 64]...",
        )
        result = result.numpy(self.target).reshape(-1)
        np.testing.assert_allclose(
            (tensor_data[0] + tensor_data[1]).reshape(-1), result, atol=1e-4
        )

    def atest_elementwise2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 112, 112])
        b = Variable("B").set_type(Float(32)).set_shape([2, 512, 112, 112])
        c = prog.add(a, b)
        tensor_data = [
            np.random.random([2, 512, 112, 112]).astype("float32"),
            np.random.random([2, 512, 112, 112]).astype("float32"),
        ]
        result = prog.test_benchmark(
            self.target,
            [a, b],
            tensor_data,
            c,
            200,
            "TESTING [elementwise_add] time cost with shape [2, 512, 112, 112]...",
        )
        result = result.numpy(self.target).reshape(-1)
        np.testing.assert_allclose(
            (tensor_data[0] + tensor_data[1]).reshape(-1), result, atol=1e-4
        )

    def atest_elementwise3(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([4, 1024])
        b = Variable("B").set_type(Float(32)).set_shape([4, 1024])
        c = prog.add(a, b)
        tensor_data = [
            np.random.random([4, 1024]).astype("float32"),
            np.random.random([4, 1024]).astype("float32"),
        ]
        result = prog.test_benchmark_with_code(
            self.target,
            [a, b],
            tensor_data,
            c,
            200,
            "TESTING [elementwise_add] time cost with input code...",
            '''extern "C" {

__global__
void fn_elementwise_add_0_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ EleAdd_Out_0)
{

      EleAdd_Out_0[1024 * blockIdx.x + threadIdx.x] = (A[1024 * blockIdx.x + threadIdx.x] + B[1024 * blockIdx.x + threadIdx.x]);
}

}''',
        )

    def atest_batchnorm(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 32, 32])
        b = Variable("B").set_type(Float(32)).set_shape([512])
        c = Variable("C").set_type(Float(32)).set_shape([512])
        d = Variable("D").set_type(Float(32)).set_shape([512])
        e = Variable("E").set_type(Float(32)).set_shape([512])
        f = prog.batchnorm(a, b, c, d, e, {})
        tensor_data = [
            np.random.random([2, 512, 32, 32]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32"),
        ]
        result = prog.test_benchmark(
            self.target,
            [a, b, c, d, e],
            tensor_data,
            f,
            1000,
            "TESTING [batchnorm] time cost with shape [2, 512, 32, 32]...",
        )

    def atest_batchnorm2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 64, 8, 8])
        b = Variable("B").set_type(Float(32)).set_shape([64])
        c = Variable("C").set_type(Float(32)).set_shape([64])
        d = Variable("D").set_type(Float(32)).set_shape([64])
        e = Variable("E").set_type(Float(32)).set_shape([64])
        f = prog.batchnorm(a, b, c, d, e, {})
        tensor_data = [
            np.random.random([2, 64, 8, 8]).astype("float32"),
            np.random.random([64]).astype("float32"),
            np.random.random([64]).astype("float32"),
            np.random.random([64]).astype("float32"),
            np.random.random([64]).astype("float32"),
        ]
        result = prog.test_benchmark(
            self.target,
            [a, b, c, d, e],
            tensor_data,
            f,
            200,
            "TESTING [batchnorm] time cost with shape [2, 64, 8, 8]...",
        )

    def atest_relu3(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 112, 112])
        c = prog.relu(a)
        tensor_data = [np.random.random([2, 512, 112, 112]).astype("float32")]
        result = prog.test_benchmark(
            self.target,
            [a],
            tensor_data,
            c,
            200,
            "TESTING [relu] time cost with shape [2,512,112,112]...",
        )

    def atest_relu(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([64, 64])
        c = prog.sigmoid(a)
        tensor_data = [np.random.random([64, 64]).astype("float32")]
        result = prog.test_benchmark(
            self.target,
            [a],
            tensor_data,
            c,
            200,
            "TESTING [sigmoid] time cost with shape [64,64]...",
        )

    def atest_relu2(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 112, 112])
        c = prog.sigmoid(a)
        tensor_data = [np.random.random([2, 512, 112, 112]).astype("float32")]
        result = prog.test_benchmark(
            self.target,
            [a],
            tensor_data,
            c,
            200,
            "TESTING [sigmoid] time cost with shape [2,512,112,112]...",
        )


if __name__ == "__main__":
    unittest.main()

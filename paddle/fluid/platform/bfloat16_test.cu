/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/bfloat16.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>

#ifdef PADDLE_CUDA_BF16
using bf16 = __nv_bfloat16;

#define ARITHMETIC_KERNEL(op_type, sign)                                 \
  __global__ void op_type(const bf16 *in1, const bf16 *in2, bf16 *out) { \
    out[0] = in1[0] sign in2[0];                                         \
  }

#define COMPOUND_KERNEL(op_type, sign) \
  __global__ void op_type(bf16 *in1, const bf16 *in2) { in1[0] sign in2[0]; }

#define COMPARISON_KERNEL(op_type, sign)                                 \
  __global__ void op_type(const bf16 *in1, const bf16 *in2, bool *out) { \
    out[0] = in1[0] sign in2[0];                                         \
  }

#define ARITHMETIC_KERNEL_LAUNCH(op_type)                     \
  void Test##op_type(float v_in1, float v_in2, float v_out) { \
    LOG(INFO) << "Test " << #op_type << " on GPU!";           \
    bf16 *in1, *in2, *out;                                    \
    bf16 *d_in1, *d_in2, *d_out;                              \
    int size = sizeof(bf16);                                  \
    cudaMalloc(reinterpret_cast<void **>(&d_in1), size);      \
    cudaMalloc(reinterpret_cast<void **>(&d_in2), size);      \
    cudaMalloc(reinterpret_cast<void **>(&d_out), size);      \
    in1 = reinterpret_cast<bf16 *>(malloc(size));             \
    in2 = reinterpret_cast<bf16 *>(malloc(size));             \
    out = reinterpret_cast<bf16 *>(malloc(size));             \
    in1[0] = bf16(bfloat16(v_in1));                           \
    in2[0] = bf16(bfloat16(v_in2));                           \
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);     \
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);     \
    op_type<<<1, 1>>>(d_in1, d_in2, d_out);                   \
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);     \
    EXPECT_EQ(static_cast<float>(bfloat16(out[0])), v_out);   \
    free(in1);                                                \
    free(in2);                                                \
    free(out);                                                \
    cudaFree(d_in1);                                          \
    cudaFree(d_in2);                                          \
    cudaFree(d_out);                                          \
  }

#define COMPOUND_KERNEL_LAUNCH(op_type)                       \
  void Test##op_type(float v_in1, float v_in2, float v_out) { \
    LOG(INFO) << "Test " << #op_type << " on GPU!";           \
    bf16 *in1, *in2;                                          \
    bf16 *d_in1, *d_in2;                                      \
    int size = sizeof(bf16);                                  \
    cudaMalloc(reinterpret_cast<void **>(&d_in1), size);      \
    cudaMalloc(reinterpret_cast<void **>(&d_in2), size);      \
    in1 = reinterpret_cast<bf16 *>(malloc(size));             \
    in2 = reinterpret_cast<bf16 *>(malloc(size));             \
    in1[0] = bf16(bfloat16(v_in1));                           \
    in2[0] = bf16(bfloat16(v_in2));                           \
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);     \
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);     \
    op_type<<<1, 1>>>(d_in1, d_in2);                          \
    cudaMemcpy(in1, d_in1, size, cudaMemcpyDeviceToHost);     \
    EXPECT_EQ(static_cast<float>(bfloat16(in1[0])), v_out);   \
    free(in1);                                                \
    free(in2);                                                \
    cudaFree(d_in1);                                          \
    cudaFree(d_in2);                                          \
  }

#define COMPARISON_KERNEL_LAUNCH(op_type)                    \
  void Test##op_type(float v_in1, float v_in2, bool v_out) { \
    LOG(INFO) << "Test " << #op_type << " on GPU!";          \
    bf16 *in1, *in2;                                         \
    bf16 *d_in1, *d_in2;                                     \
    bool *out, *d_out;                                       \
    int size = sizeof(bf16);                                 \
    cudaMalloc(reinterpret_cast<void **>(&d_in1), size);     \
    cudaMalloc(reinterpret_cast<void **>(&d_in2), size);     \
    cudaMalloc(reinterpret_cast<void **>(&d_out), 1);        \
    in1 = reinterpret_cast<bf16 *>(malloc(size));            \
    in2 = reinterpret_cast<bf16 *>(malloc(size));            \
    out = reinterpret_cast<bool *>(malloc(1));               \
    in1[0] = bf16(bfloat16(v_in1));                          \
    in2[0] = bf16(bfloat16(v_in2));                          \
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);    \
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);    \
    op_type<<<1, 1>>>(d_in1, d_in2, d_out);                  \
    cudaMemcpy(out, d_out, 1, cudaMemcpyDeviceToHost);       \
    EXPECT_EQ(out[0], v_out);                                \
    free(in1);                                               \
    free(in2);                                               \
    free(out);                                               \
    cudaFree(d_in1);                                         \
    cudaFree(d_in2);                                         \
    cudaFree(d_out);                                         \
  }

namespace paddle {
namespace platform {

// Arithmetic operations testing
ARITHMETIC_KERNEL(Add, +)
ARITHMETIC_KERNEL(Sub, -)
ARITHMETIC_KERNEL(Mul, *)
ARITHMETIC_KERNEL(Div, /)

ARITHMETIC_KERNEL_LAUNCH(Add)
ARITHMETIC_KERNEL_LAUNCH(Sub)
ARITHMETIC_KERNEL_LAUNCH(Mul)
ARITHMETIC_KERNEL_LAUNCH(Div)

__global__ void Neg(bf16 *in) { in[0] = -in[0]; }

void TestNeg(float v_in, float v_out) {
  LOG(INFO) << "Test Neg on GPU!";
  bf16 *in, *d_in;
  int size = sizeof(bf16);
  cudaMalloc(reinterpret_cast<void **>(&d_in), size);
  in = reinterpret_cast<bf16 *>(malloc(size));
  in[0] = bf16(bfloat16(v_in));
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  Neg<<<1, 1>>>(d_in);
  cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
  EXPECT_EQ(static_cast<float>(bfloat16(in[0])), v_out);
  free(in);
  cudaFree(d_in);
}

TEST(bfloat16, arithmetic_on_gpu) {
  TestAdd(1, 2, 3);
  TestSub(2, 1, 1);
  TestMul(2, 3, 6);
  TestDiv(6, 2, 3);
  TestNeg(1, -1);
}

// Compound operations testing
COMPOUND_KERNEL(AddAssign, +=)
COMPOUND_KERNEL(SubAssign, -=)
COMPOUND_KERNEL(MulAssign, *=)
COMPOUND_KERNEL(DivAssign, /=)

COMPOUND_KERNEL_LAUNCH(AddAssign)
COMPOUND_KERNEL_LAUNCH(SubAssign)
COMPOUND_KERNEL_LAUNCH(MulAssign)
COMPOUND_KERNEL_LAUNCH(DivAssign)

TEST(float16, compound_on_gpu) {
  TestAddAssign(1, 2, 3);
  TestSubAssign(2, 1, 1);
  TestMulAssign(2, 3, 6);
  TestDivAssign(6, 2, 3);
}

// Comparison operations testing
COMPARISON_KERNEL(Equal, ==)
COMPARISON_KERNEL(NotEqual, !=)
COMPARISON_KERNEL(Less, <)
COMPARISON_KERNEL(LessEqual, <=)
COMPARISON_KERNEL(Greater, >)
COMPARISON_KERNEL(GreaterEqual, >=)

COMPARISON_KERNEL_LAUNCH(Equal)
COMPARISON_KERNEL_LAUNCH(NotEqual)
COMPARISON_KERNEL_LAUNCH(Less)
COMPARISON_KERNEL_LAUNCH(LessEqual)
COMPARISON_KERNEL_LAUNCH(Greater)
COMPARISON_KERNEL_LAUNCH(GreaterEqual)

TEST(float16, comparision_on_gpu) {
  TestEqual(1, 1, true);
  TestEqual(1, 2, false);
  TestNotEqual(2, 3, true);
  TestNotEqual(2, 2, false);
  TestLess(3, 4, true);
  TestLess(3, 3, false);
  TestLessEqual(3, 3, true);
  TestLessEqual(3, 2, false);
  TestGreater(4, 3, true);
  TestGreater(4, 4, false);
  TestGreaterEqual(4, 4, true);
  TestGreaterEqual(4, 5, false);
}

}  // namespace platform
}  // namespace paddle
#endif  // PADDLE_CUDA_BF16

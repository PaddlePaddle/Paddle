/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/math/float16.h"

#include <gtest/gtest.h>

#include "paddle/utils/Logging.h"

#define ARITHMETIC_KERNEL(op_type, sign)                                 \
  __global__ void op_type(const half* in1, const half* in2, half* out) { \
    out[0] = in1[0] sign in2[0];                                         \
  }

#define COMPOUND_KERNEL(op_type, sign) \
  __global__ void op_type(half* in1, const half* in2) { in1[0] sign in2[0]; }

#define COMPARISON_KERNEL(op_type, sign)                                 \
  __global__ void op_type(const half* in1, const half* in2, bool* out) { \
    out[0] = in1[0] sign in2[0];                                         \
  }

#define ARITHMETIC_KERNEL_LAUNCH(op_type)                     \
  void Test##op_type(float v_in1, float v_in2, float v_out) { \
    LOG(INFO) << "Test " << #op_type << " on GPU!";           \
    half *in1, *in2, *out;                                    \
    half *d_in1, *d_in2, *d_out;                              \
    int size = sizeof(half);                                  \
    cudaMalloc((void**)&d_in1, size);                         \
    cudaMalloc((void**)&d_in2, size);                         \
    cudaMalloc((void**)&d_out, size);                         \
    in1 = (half*)malloc(size);                                \
    in2 = (half*)malloc(size);                                \
    out = (half*)malloc(size);                                \
    in1[0] = half(float16(v_in1));                            \
    in2[0] = half(float16(v_in2));                            \
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);     \
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);     \
    op_type<<<1, 1>>>(d_in1, d_in2, d_out);                   \
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);     \
    EXPECT_EQ(float(float16(out[0])), v_out);                 \
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
    half *in1, *in2;                                          \
    half *d_in1, *d_in2;                                      \
    int size = sizeof(half);                                  \
    cudaMalloc((void**)&d_in1, size);                         \
    cudaMalloc((void**)&d_in2, size);                         \
    in1 = (half*)malloc(size);                                \
    in2 = (half*)malloc(size);                                \
    in1[0] = half(float16(v_in1));                            \
    in2[0] = half(float16(v_in2));                            \
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);     \
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);     \
    op_type<<<1, 1>>>(d_in1, d_in2);                          \
    cudaMemcpy(in1, d_in1, size, cudaMemcpyDeviceToHost);     \
    EXPECT_EQ(float(float16(in1[0])), v_out);                 \
    free(in1);                                                \
    free(in2);                                                \
    cudaFree(d_in1);                                          \
    cudaFree(d_in2);                                          \
  }

#define COMPARISON_KERNEL_LAUNCH(op_type)                    \
  void Test##op_type(float v_in1, float v_in2, bool v_out) { \
    LOG(INFO) << "Test " << #op_type << " on GPU!";          \
    half *in1, *in2;                                         \
    half *d_in1, *d_in2;                                     \
    bool *out, *d_out;                                       \
    int size = sizeof(half);                                 \
    cudaMalloc((void**)&d_in1, size);                        \
    cudaMalloc((void**)&d_in2, size);                        \
    cudaMalloc((void**)&d_out, 1);                           \
    in1 = (half*)malloc(size);                               \
    in2 = (half*)malloc(size);                               \
    out = (bool*)malloc(1);                                  \
    in1[0] = half(float16(v_in1));                           \
    in2[0] = half(float16(v_in2));                           \
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

#ifdef PADDLE_CUDA_FP16
namespace paddle {

#if CUDA_VERSION < 9000
ARITHMETIC_KERNEL(Add, +)
ARITHMETIC_KERNEL(Sub, -)
ARITHMETIC_KERNEL(Mul, *)
ARITHMETIC_KERNEL(Div, /)

ARITHMETIC_KERNEL_LAUNCH(Add)
ARITHMETIC_KERNEL_LAUNCH(Sub)
ARITHMETIC_KERNEL_LAUNCH(Mul)
ARITHMETIC_KERNEL_LAUNCH(Div)

// Negative sign kernel
__global__ void Neg(half* in) { in[0] = -in[0]; }

void TestNeg(float v_in, float v_out) {
  LOG(INFO) << "Test Neg on GPU!";
  half *in, *d_in;
  int size = sizeof(half);
  cudaMalloc((void**)&d_in, size);
  in = (half*)malloc(size);
  in[0] = half(float16(v_in));
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  Neg<<<1, 1>>>(d_in);
  cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
  EXPECT_EQ(float(float16(in[0])), v_out);
  free(in);
  cudaFree(d_in);
}

COMPOUND_KERNEL(AddAssign, +=)
COMPOUND_KERNEL(SubAssign, -=)
COMPOUND_KERNEL(MulAssign, *=)
COMPOUND_KERNEL(DivAssign, /=)

COMPOUND_KERNEL_LAUNCH(AddAssign)
COMPOUND_KERNEL_LAUNCH(SubAssign)
COMPOUND_KERNEL_LAUNCH(MulAssign)
COMPOUND_KERNEL_LAUNCH(DivAssign)

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

TEST(float16, arithmetic_on_gpu) {
  TestAdd(1, 2, 3);
  TestSub(2, 1, 1);
  TestMul(2, 3, 6);
  TestDiv(6, 2, 3);
  TestNeg(1, -1);
}

TEST(float16, compound_on_gpu) {
  TestAddAssign(1, 2, 3);
  TestSubAssign(2, 1, 1);
  TestMulAssign(2, 3, 6);
  TestDivAssign(6, 2, 3);
}

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
#endif  // CUDA_VERSION

TEST(float16, conversion_on_gpu) {
  // Explicit conversion to and from cuda half
  EXPECT_EQ(float16(half(float16(1.0f))).x, 0x3c00);
  EXPECT_EQ(float16(half(float16(0.5f))).x, 0x3800);
  EXPECT_EQ(float16(half(float16(0.33333f))).x, 0x3555);
  EXPECT_EQ(float16(half(float16(0.0f))).x, 0x0000);
  EXPECT_EQ(float16(half(float16(-0.0f))).x, 0x8000);
  EXPECT_EQ(float16(half(float16(65504.0f))).x, 0x7bff);
  EXPECT_EQ(float16(half(float16(65536.0f))).x, 0x7c00);

  // Assignment operator
  float16 v_assign;
  v_assign = half(float16(1.0f));
  EXPECT_EQ(v_assign.x, 0x3c00);
}

}  // namespace paddle
#endif  // PADDLE_CUDA_FP16

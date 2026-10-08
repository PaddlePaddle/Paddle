//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstring>
#include <limits>

#include "gtest/gtest.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace tests {

void fill_fp16_data(phi::dtype::float16* in_ptr,
                    size_t size,
                    const std::vector<float>& data) {
  PADDLE_ENFORCE_EQ(
      size,
      data.size(),
      common::errors::InvalidArgument(
          "The size of argument data should"
          " be equal to the argument size. Expected %d, but received %d.",
          size,
          data.size()));
  for (size_t i = 0; i < data.size(); ++i) {
    in_ptr[i] = phi::dtype::float16(data[i]);
  }
}

template <typename T>
inline phi::funcs::BlasT<phi::GPUContext, T> GetBlas(
    const phi::GPUContext& context) {
  return phi::funcs::GetBlas<phi::GPUContext, T>(context);
}

TEST(math_function, notrans_mul_trans_fp32) {
  phi::DenseTensor input1;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor out_gpu;
  phi::DenseTensor out;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input1, gpu_place, true, &input2_gpu);

  out_gpu.mutable_data<float>({2, 2}, gpu_place);
  GetBlas<float>(*context).MatMul(
      input1_gpu, false, input2_gpu, true, 1, &out_gpu, 0);

  phi::Copy(*context, out_gpu, cpu_place, true, &out);

  float* out_ptr = out.data<float>();
  context->Wait();
  EXPECT_EQ(out_ptr[0], 5);
  EXPECT_EQ(out_ptr[1], 14);
  EXPECT_EQ(out_ptr[2], 14);
  EXPECT_EQ(out_ptr[3], 50);
}

TEST(math_function, notrans_mul_trans_fp16) {
  phi::DenseTensor input1;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor out_gpu;
  phi::DenseTensor out;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context->GetComputeCapability() < 53) {
    return;
  }

  phi::dtype::float16* input1_ptr =
      input1.mutable_data<phi::dtype::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input1, gpu_place, true, &input2_gpu);

  out_gpu.mutable_data<phi::dtype::float16>({2, 2}, gpu_place);

  GetBlas<phi::dtype::float16>(*context).MatMul(input1_gpu,
                                                false,
                                                input2_gpu,
                                                true,
                                                phi::dtype::float16(1),
                                                &out_gpu,
                                                phi::dtype::float16(0));

  phi::Copy(*context, out_gpu, cpu_place, true, &out);

  phi::dtype::float16* out_ptr = out.data<phi::dtype::float16>();
  context->Wait();
  EXPECT_EQ(static_cast<float>(out_ptr[0]), 5);
  EXPECT_EQ(static_cast<float>(out_ptr[1]), 14);
  EXPECT_EQ(static_cast<float>(out_ptr[2]), 14);
  EXPECT_EQ(static_cast<float>(out_ptr[3]), 50);
}

TEST(math_function, trans_mul_notrans_fp32) {
  phi::DenseTensor input1;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor out_gpu;
  phi::DenseTensor out;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input1, gpu_place, true, &input2_gpu);

  out_gpu.mutable_data<float>({3, 3}, gpu_place);

  GetBlas<float>(*context).MatMul(
      input1_gpu, true, input2_gpu, false, 1, &out_gpu, 0);

  phi::Copy(*context, out_gpu, cpu_place, true, &out);

  float* out_ptr = out.data<float>();
  context->Wait();
  EXPECT_EQ(out_ptr[0], 9);
  EXPECT_EQ(out_ptr[1], 12);
  EXPECT_EQ(out_ptr[2], 15);
  EXPECT_EQ(out_ptr[3], 12);
  EXPECT_EQ(out_ptr[4], 17);
  EXPECT_EQ(out_ptr[5], 22);
  EXPECT_EQ(out_ptr[6], 15);
  EXPECT_EQ(out_ptr[7], 22);
  EXPECT_EQ(out_ptr[8], 29);
}

TEST(math_function, trans_mul_notrans_fp16) {
  phi::DenseTensor input1;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor out_gpu;
  phi::DenseTensor out;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context->GetComputeCapability() < 53) {
    return;
  }

  phi::dtype::float16* input1_ptr =
      input1.mutable_data<phi::dtype::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input1, gpu_place, true, &input2_gpu);

  out_gpu.mutable_data<phi::dtype::float16>({3, 3}, gpu_place);

  GetBlas<phi::dtype::float16>(*context).MatMul(input1_gpu,
                                                true,
                                                input2_gpu,
                                                false,
                                                phi::dtype::float16(1),
                                                &out_gpu,
                                                phi::dtype::float16(0));

  phi::Copy(*context, out_gpu, cpu_place, true, &out);

  phi::dtype::float16* out_ptr = out.data<phi::dtype::float16>();
  context->Wait();
  EXPECT_EQ(static_cast<float>(out_ptr[0]), 9);
  EXPECT_EQ(static_cast<float>(out_ptr[1]), 12);
  EXPECT_EQ(static_cast<float>(out_ptr[2]), 15);
  EXPECT_EQ(static_cast<float>(out_ptr[3]), 12);
  EXPECT_EQ(static_cast<float>(out_ptr[4]), 17);
  EXPECT_EQ(static_cast<float>(out_ptr[5]), 22);
  EXPECT_EQ(static_cast<float>(out_ptr[6]), 15);
  EXPECT_EQ(static_cast<float>(out_ptr[7]), 22);
  EXPECT_EQ(static_cast<float>(out_ptr[8]), 29);
}

TEST(math_function, gemm_notrans_cublas_fp32) {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor input3;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor input3_gpu;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  int m = 2;
  int n = 3;
  int k = 3;
  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr1[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1, 6 * sizeof(float));
  float* input2_ptr = input2.mutable_data<float>({3, 4}, cpu_place);
  float arr2[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  memcpy(input2_ptr, arr2, 12 * sizeof(float));
  float* input3_ptr = input3.mutable_data<float>({2, 4}, cpu_place);
  float arr3[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3, 8 * sizeof(float));

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input2, gpu_place, true, &input2_gpu);
  phi::Copy(*context, input3, gpu_place, true, &input3_gpu);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(gpu_place);

  GetBlas<float>(*context).GEMM(
      false, false, m, n, k, 1, a, 3, b + 1, 4, 1, c + 1, 4);

  phi::Copy(*context, input3_gpu, cpu_place, true, &input3);

  // numpy code:
  // a = np.arange(6).reshape(2, 3)
  // b = np.arange(12).reshape(3, 4)[:, 1:]
  // c = np.arange(8).reshape(2, 4)[:, 1:]
  // out = np.arange(8).reshape(2, 4)
  // out[:, 1:] = np.dot(a, b) + c
  context->Wait();
  EXPECT_EQ(input3_ptr[0], 0);
  EXPECT_EQ(input3_ptr[1], 24);
  EXPECT_EQ(input3_ptr[2], 28);
  EXPECT_EQ(input3_ptr[3], 32);
  EXPECT_EQ(input3_ptr[4], 4);
  EXPECT_EQ(input3_ptr[5], 73);
  EXPECT_EQ(input3_ptr[6], 86);
  EXPECT_EQ(input3_ptr[7], 99);
}

TEST(math_function, gemm_notrans_cublas_fp16) {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor input3;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor input3_gpu;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context->GetComputeCapability() < 53) {
    return;
  }

  int m = 2;
  int n = 3;
  int k = 3;
  phi::dtype::float16* input1_ptr =
      input1.mutable_data<phi::dtype::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});
  phi::dtype::float16* input2_ptr =
      input2.mutable_data<phi::dtype::float16>({3, 4}, cpu_place);
  fill_fp16_data(
      input2_ptr, input2.numel(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  phi::dtype::float16* input3_ptr =
      input3.mutable_data<phi::dtype::float16>({2, 4}, cpu_place);
  fill_fp16_data(input3_ptr, input3.numel(), {0, 1, 2, 3, 4, 5, 6, 7});

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input2, gpu_place, true, &input2_gpu);
  phi::Copy(*context, input3, gpu_place, true, &input3_gpu);
  phi::dtype::float16* a = input1_gpu.data<phi::dtype::float16>();
  phi::dtype::float16* b = input2_gpu.data<phi::dtype::float16>();
  phi::dtype::float16* c =
      input3_gpu.mutable_data<phi::dtype::float16>(gpu_place);

  GetBlas<phi::dtype::float16>(*context).GEMM(
      false,
      false,
      m,
      n,
      k,
      static_cast<phi::dtype::float16>(1),
      a,
      3,
      b + 1,
      4,
      static_cast<phi::dtype::float16>(1),
      c + 1,
      4);

  phi::Copy(*context, input3_gpu, cpu_place, true, &input3);

  // numpy code:
  // a = np.arange(6).reshape(2, 3)
  // b = np.arange(12).reshape(3, 4)[:, 1:]
  // c = np.arange(8).reshape(2, 4)[:, 1:]
  // out = np.arange(8).reshape(2, 4)
  // out[:, 1:] = np.dot(a, b) + c
  context->Wait();
  EXPECT_EQ(static_cast<float>(input3_ptr[0]), 0);
  EXPECT_EQ(static_cast<float>(input3_ptr[1]), 24);
  EXPECT_EQ(static_cast<float>(input3_ptr[2]), 28);
  EXPECT_EQ(static_cast<float>(input3_ptr[3]), 32);
  EXPECT_EQ(static_cast<float>(input3_ptr[4]), 4);
  EXPECT_EQ(static_cast<float>(input3_ptr[5]), 73);
  EXPECT_EQ(static_cast<float>(input3_ptr[6]), 86);
  EXPECT_EQ(static_cast<float>(input3_ptr[7]), 99);
}

TEST(math_function, gemm_trans_cublas_fp32) {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor input3;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor input3_gpu;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  int m = 2;
  int n = 3;
  int k = 3;
  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr1[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1, 6 * sizeof(float));
  float* input2_ptr = input2.mutable_data<float>({4, 3}, cpu_place);
  float arr2[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  memcpy(input2_ptr, arr2, 12 * sizeof(float));
  float* input3_ptr = input3.mutable_data<float>({2, 4}, cpu_place);
  float arr3[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3, 8 * sizeof(float));

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input2, gpu_place, true, &input2_gpu);
  phi::Copy(*context, input3, gpu_place, true, &input3_gpu);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(gpu_place);

  GetBlas<float>(*context).GEMM(
      false, true, m, n, k, 1, a, 3, b + 3, 3, 1, c + 1, 4);

  phi::Copy(*context, input3_gpu, cpu_place, true, &input3);

  context->Wait();
  EXPECT_EQ(input3_ptr[0], 0);
  EXPECT_EQ(input3_ptr[1], 24);
  EXPECT_EQ(input3_ptr[2], 28);
  EXPECT_EQ(input3_ptr[3], 32);
  EXPECT_EQ(input3_ptr[4], 4);
  EXPECT_EQ(input3_ptr[5], 73);
  EXPECT_EQ(input3_ptr[6], 86);
  EXPECT_EQ(input3_ptr[7], 99);
}

TEST(math_function, gemm_trans_cublas_fp16) {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor input3;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor input3_gpu;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context->GetComputeCapability() < 53) {
    return;
  }

  int m = 2;
  int n = 3;
  int k = 3;
  phi::dtype::float16* input1_ptr =
      input1.mutable_data<phi::dtype::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});
  phi::dtype::float16* input2_ptr =
      input2.mutable_data<phi::dtype::float16>({4, 3}, cpu_place);
  fill_fp16_data(
      input2_ptr, input2.numel(), {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11});
  phi::dtype::float16* input3_ptr =
      input3.mutable_data<phi::dtype::float16>({2, 4}, cpu_place);
  fill_fp16_data(input3_ptr, input3.numel(), {0, 1, 2, 3, 4, 5, 6, 7});

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input2, gpu_place, true, &input2_gpu);
  phi::Copy(*context, input3, gpu_place, true, &input3_gpu);
  phi::dtype::float16* a = input1_gpu.data<phi::dtype::float16>();
  phi::dtype::float16* b = input2_gpu.data<phi::dtype::float16>();
  phi::dtype::float16* c =
      input3_gpu.mutable_data<phi::dtype::float16>(gpu_place);

  GetBlas<phi::dtype::float16>(*context).GEMM(
      false,
      true,
      m,
      n,
      k,
      static_cast<phi::dtype::float16>(1),
      a,
      3,
      b + 3,
      3,
      static_cast<phi::dtype::float16>(1),
      c + 1,
      4);

  phi::Copy(*context, input3_gpu, cpu_place, true, &input3);

  context->Wait();
  EXPECT_EQ(static_cast<float>(input3_ptr[0]), 0);
  EXPECT_EQ(static_cast<float>(input3_ptr[1]), 24);
  EXPECT_EQ(static_cast<float>(input3_ptr[2]), 28);
  EXPECT_EQ(static_cast<float>(input3_ptr[3]), 32);
  EXPECT_EQ(static_cast<float>(input3_ptr[4]), 4);
  EXPECT_EQ(static_cast<float>(input3_ptr[5]), 73);
  EXPECT_EQ(static_cast<float>(input3_ptr[6]), 86);
  EXPECT_EQ(static_cast<float>(input3_ptr[7]), 99);
}

template <typename T>
void Gemm64BitLeadingDimensionTest(int compute_capability) {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor output;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor output_gpu;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  if (context->GetComputeCapability() < compute_capability) {
    return;
  }

  T* input1_ptr = input1.mutable_data<T>({1, 1}, cpu_place);
  T* input2_ptr = input2.mutable_data<T>({1, 1}, cpu_place);
  T* output_ptr = output.mutable_data<T>({1, 1}, cpu_place);
  input1_ptr[0] = static_cast<T>(2.0f);
  input2_ptr[0] = static_cast<T>(3.0f);
  output_ptr[0] = static_cast<T>(0.0f);

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input2, gpu_place, true, &input2_gpu);
  phi::Copy(*context, output, gpu_place, true, &output_gpu);

  const int64_t large_ld =
      static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  bool gemm_succeeded = true;
  try {
    GetBlas<T>(*context).GEMM(false,
                              false,
                              1,
                              1,
                              1,
                              static_cast<T>(1.0f),
                              input1_gpu.data<T>(),
                              1,
                              input2_gpu.data<T>(),
                              1,
                              static_cast<T>(0.0f),
                              output_gpu.data<T>(),
                              large_ld);
  } catch (const common::enforce::EnforceNotMet& error) {
    // Some CUDA versions reject oversized leading dimensions in cuBLAS even
    // through the 64-bit entry point. The dispatch must still reach cuBLAS.
    gemm_succeeded = false;
    EXPECT_EQ(error.code(), common::ErrorCode::EXTERNAL);
  }

  if (gemm_succeeded) {
    phi::Copy(*context, output_gpu, cpu_place, true, &output);
    context->Wait();
    EXPECT_FLOAT_EQ(static_cast<float>(output_ptr[0]), 6.0f);
  }
}

#if CUDA_VERSION >= 12030 && defined(__linux__)
TEST(math_function, gemm_64bit_leading_dimension_dispatch_fp16) {
  Gemm64BitLeadingDimensionTest<phi::dtype::float16>(53);
}

TEST(math_function, gemm_64bit_leading_dimension_dispatch_bf16) {
  Gemm64BitLeadingDimensionTest<phi::dtype::bfloat16>(80);
}

template <typename T>
void ComplexGemm64ApiTest() {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor output;
  phi::DenseTensor input1_gpu;
  phi::DenseTensor input2_gpu;
  phi::DenseTensor output_gpu;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  auto* input1_ptr = input1.mutable_data<T>({1, 1}, cpu_place);
  auto* input2_ptr = input2.mutable_data<T>({1, 1}, cpu_place);
  auto* output_ptr = output.mutable_data<T>({1, 1}, cpu_place);
  input1_ptr[0] = T(2.0, 0.0);
  input2_ptr[0] = T(3.0, 0.0);
  output_ptr[0] = T(0.0, 0.0);

  phi::Copy(*context, input1, gpu_place, true, &input1_gpu);
  phi::Copy(*context, input2, gpu_place, true, &input2_gpu);
  phi::Copy(*context, output, gpu_place, true, &output_gpu);

  const T alpha(1.0, 0.0);
  const T beta(0.0, 0.0);
  context->CublasCall([&](cublasHandle_t handle) {
    phi::funcs::CUBlas<T>::GEMM_64(handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   1,
                                   1,
                                   1,
                                   &alpha,
                                   input1_gpu.data<T>(),
                                   1,
                                   input2_gpu.data<T>(),
                                   1,
                                   &beta,
                                   output_gpu.data<T>(),
                                   1);
  });

  phi::Copy(*context, output_gpu, cpu_place, true, &output);
  context->Wait();
  EXPECT_DOUBLE_EQ(static_cast<double>(output_ptr[0].real), 6.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(output_ptr[0].imag), 0.0);
}

TEST(math_function, gemm_64_complex64_typed_api) {
  ComplexGemm64ApiTest<phi::complex64>();
}

TEST(math_function, gemm_64_complex128_typed_api) {
  ComplexGemm64ApiTest<phi::complex128>();
}
#endif

template <typename T>
void GemvTest(int64_t m, int64_t n, bool trans) {
  phi::DenseTensor mat_a;
  phi::DenseTensor vec_b;
  phi::DenseTensor vec_c;

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  T* data_a = mat_a.mutable_data<T>({m, n}, cpu_place);
  T* data_b = vec_b.mutable_data<T>({trans ? m : n}, cpu_place);
  T* data_c = vec_c.mutable_data<T>({trans ? n : m}, cpu_place);

  phi::DenseTensor g_mat_a;
  phi::DenseTensor g_vec_b;
  phi::DenseTensor g_vec_c;
  T* g_data_a = g_mat_a.mutable_data<T>(mat_a.dims(), gpu_place);
  T* g_data_b = g_vec_b.mutable_data<T>(vec_b.dims(), gpu_place);
  T* g_data_c = g_vec_c.mutable_data<T>(vec_c.dims(), gpu_place);

  for (int64_t i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = static_cast<T>(i);
  }
  for (int64_t i = 0; i < vec_b.numel(); ++i) {
    data_b[i] = static_cast<T>(i);
  }

  phi::Copy(*context, mat_a, gpu_place, true, &g_mat_a);
  phi::Copy(*context, vec_b, gpu_place, true, &g_vec_b);

  GetBlas<T>(*context).GEMV(trans, m, n, 1., g_data_a, g_data_b, 0., g_data_c);

  phi::Copy(*context, g_vec_c, cpu_place, true, &vec_c);

  if (!trans) {
    for (int64_t i = 0; i < m; ++i) {
      T sum = 0.0;
      for (int64_t j = 0; j < n; ++j) {
        sum += data_a[i * n + j] * data_b[j];
      }
      ASSERT_FLOAT_EQ(data_c[i], sum);
    }
  } else {
    for (int64_t i = 0; i < n; ++i) {
      T sum = 0.0;
      for (int64_t j = 0; j < m; ++j) {
        sum += data_a[j * n + i] * data_b[j];
      }
      ASSERT_FLOAT_EQ(data_c[i], sum);
    }
  }
}

TEST(math_function, gemv) {
  GemvTest<float>(3, 13, false);
  GemvTest<double>(3, 13, false);
  GemvTest<float>(3, 13, true);
  GemvTest<double>(3, 13, true);
}

// `static_cast<T>(int)` and `ASSERT_FLOAT_EQ` do not work for phi::complex64 /
// phi::complex128, so value construction and comparison go through these
// helpers. The operands are small integers, which are exactly representable in
// every dtype used here, so the reference is accumulated in double and the
// comparison stays exact.
template <typename T>
inline T BatchedGemmValue(int value) {
  return static_cast<T>(value);
}

template <>
inline phi::complex64 BatchedGemmValue<phi::complex64>(int value) {
  return phi::complex64(static_cast<float>(value), 0.0f);
}

template <>
inline phi::complex128 BatchedGemmValue<phi::complex128>(int value) {
  return phi::complex128(static_cast<double>(value), 0.0);
}

// phi::float16 / phi::bfloat16 convert to float by reinterpreting their 16-bit
// payload as CUDA's `half` / `__nv_bfloat16` (paddle/phi/common/float16.h and
// bfloat16.h). That type pun is undefined behaviour, and nvcc's host pass does
// fold the bf16 load to a stale value at -O2/-O3, so the payload is decoded
// here instead of through the conversion operator.
inline double BatchedGemmRealPart(const phi::float16& value) {
  __half raw;
  std::memcpy(&raw, &value.x, sizeof(raw));
  return static_cast<double>(__half2float(raw));
}

inline double BatchedGemmRealPart(const phi::bfloat16& value) {
  const uint32_t bits = static_cast<uint32_t>(value.x) << 16;
  float raw = 0.0f;
  std::memcpy(&raw, &bits, sizeof(raw));
  return static_cast<double>(raw);
}

template <typename T>
inline double BatchedGemmRealPart(const T& value) {
  return static_cast<double>(value);
}

inline double BatchedGemmRealPart(const phi::complex64& value) {
  return static_cast<double>(value.real);
}

inline double BatchedGemmRealPart(const phi::complex128& value) {
  return static_cast<double>(value.real);
}

template <typename T>
inline double BatchedGemmImagPart(const T&) {
  return 0.0;
}

inline double BatchedGemmImagPart(const phi::complex64& value) {
  return static_cast<double>(value.imag);
}

inline double BatchedGemmImagPart(const phi::complex128& value) {
  return static_cast<double>(value.imag);
}

// Covers the strided batched path, in particular N == 1, where the row-major
// operands are handed to cuBLAS without the usual A/B swap.
//
// `T` is the operand type and `U` the accumulator/output type. They are equal
// for the ordinary batched GEMM (bmm, baddbmm, matmul and their gradients) and
// differ for the fp16/bf16 operands with a float32 output that
// bmm_kernel_impl.h uses when bmm's out_dtype is float32: that call goes
// through a plain `funcs::Blas<Context>` with 1.0f / 0.0f, so it resolves to a
// different BatchedGEMM overload -- with its own copy of the N == 1
// rearrangement -- than the `BlasT<T>` form. Both are exercised here.
template <typename T, typename U = T>
void BatchedGemmStridedTest(int64_t batch,
                            int64_t m,
                            int64_t n,
                            int64_t k,
                            bool trans_a,
                            bool trans_b) {
  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* context = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  phi::DenseTensor mat_a, mat_b, mat_c;
  T* data_a = mat_a.mutable_data<T>({batch, m, k}, cpu_place);
  T* data_b = mat_b.mutable_data<T>({batch, k, n}, cpu_place);
  U* data_c = mat_c.mutable_data<U>({batch, m, n}, cpu_place);
  for (int64_t i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = BatchedGemmValue<T>(static_cast<int>((i % 7) - 3));
  }
  for (int64_t i = 0; i < mat_b.numel(); ++i) {
    data_b[i] = BatchedGemmValue<T>(static_cast<int>((i % 5) - 2));
  }

  phi::DenseTensor g_a, g_b, g_c;
  T* g_data_a = g_a.mutable_data<T>(mat_a.dims(), gpu_place);
  T* g_data_b = g_b.mutable_data<T>(mat_b.dims(), gpu_place);
  U* g_data_c = g_c.mutable_data<U>(mat_c.dims(), gpu_place);
  phi::Copy(*context, mat_a, gpu_place, true, &g_a);
  phi::Copy(*context, mat_b, gpu_place, true, &g_b);

  // A plain `funcs::Blas` rather than `GetBlas<U>`: `BlasT` binds the first
  // template argument to U, which would make the `float alpha ... float *C`
  // overloads unreachable (blas.h BlasT::BatchedGEMM). For U == T both forms
  // resolve to the same overload.
  phi::funcs::Blas<phi::GPUContext> blas(*context);
  // `trans_a`/`trans_b` reinterpret the same buffers as [k, m] / [n, k], so the
  // element count is unchanged and only the indexing below differs.
  blas.BatchedGEMM(trans_a ? CblasTrans : CblasNoTrans,
                   trans_b ? CblasTrans : CblasNoTrans,
                   m,
                   n,
                   k,
                   BatchedGemmValue<U>(1),
                   g_data_a,
                   g_data_b,
                   BatchedGemmValue<U>(0),
                   g_data_c,
                   batch,
                   m * k,
                   k * n);
  phi::Copy(*context, g_c, cpu_place, true, &mat_c);

  for (int64_t b = 0; b < batch; ++b) {
    const T* a = data_a + b * m * k;
    const T* bb = data_b + b * k * n;
    const U* c = data_c + b * m * n;
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        // The sum is accumulated in double rather than in T so that the
        // reference does not go through the arithmetic operators of the 16-bit
        // dtypes either.
        double sum = 0.0;
        for (int64_t p = 0; p < k; ++p) {
          sum += BatchedGemmRealPart(trans_a ? a[p * m + i] : a[i * k + p]) *
                 BatchedGemmRealPart(trans_b ? bb[j * k + p] : bb[p * n + j]);
        }
        ASSERT_DOUBLE_EQ(BatchedGemmRealPart(c[i * n + j]), sum)
            << "batch " << b << " (" << i << ", " << j << ") real part";
        // The operands are real, so the imaginary part is exactly zero for
        // every element.
        ASSERT_DOUBLE_EQ(BatchedGemmImagPart(c[i * n + j]), 0.0)
            << "batch " << b << " (" << i << ", " << j << ") imag part";
      }
    }
  }
}

TEST(math_function, batched_gemm_strided_column_output) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  const int compute_capability =
      reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()))
          ->GetComputeCapability();
  for (bool trans_a : {false, true}) {
    for (bool trans_b : {false, true}) {
      // N == 1: the torch-aligned un-swapped layout.
      BatchedGemmStridedTest<float>(3, 5, 1, 4, trans_a, trans_b);
      BatchedGemmStridedTest<double>(3, 5, 1, 4, trans_a, trans_b);
      BatchedGemmStridedTest<float>(2, 1, 1, 6, trans_a, trans_b);
      // fp16/complex cublas gemm requires GPU compute capability >= 53.
      if (compute_capability >= 53) {
        BatchedGemmStridedTest<phi::float16>(3, 5, 1, 4, trans_a, trans_b);
        BatchedGemmStridedTest<phi::complex64>(3, 5, 1, 4, trans_a, trans_b);
        BatchedGemmStridedTest<phi::complex128>(3, 5, 1, 4, trans_a, trans_b);
        // fp16 operands with a float32 accumulator and output: the
        // out_dtype=float32 path of bmm/baddbmm, which reaches the `float alpha
        // ... float *C` overloads. Those overloads only exist in the CUDA
        // backend (bmm_kernel_impl.h guards the call site the same way), so the
        // HIP build has no definition to link against.
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
        BatchedGemmStridedTest<phi::float16, float>(
            3, 5, 1, 4, trans_a, trans_b);
#endif
      }
      // The bf16 strided path is gated on cc >= 80 by the kernel itself.
      if (compute_capability >= 80) {
        BatchedGemmStridedTest<phi::bfloat16>(3, 5, 1, 4, trans_a, trans_b);
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
        BatchedGemmStridedTest<phi::bfloat16, float>(
            3, 5, 1, 4, trans_a, trans_b);
#endif
      }
      // N > 1: the swapped layout, kept as a regression guard.
      BatchedGemmStridedTest<float>(3, 5, 2, 4, trans_a, trans_b);
      BatchedGemmStridedTest<double>(3, 5, 2, 4, trans_a, trans_b);
      // fp16/complex cublas gemm requires GPU compute capability >= 53.
      if (compute_capability >= 53) {
        BatchedGemmStridedTest<phi::float16>(3, 5, 2, 4, trans_a, trans_b);
        BatchedGemmStridedTest<phi::complex64>(3, 5, 2, 4, trans_a, trans_b);
        BatchedGemmStridedTest<phi::complex128>(3, 5, 2, 4, trans_a, trans_b);
        // The `float alpha ... float *C` overloads are CUDA-only (see above).
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
        BatchedGemmStridedTest<phi::float16, float>(
            3, 5, 2, 4, trans_a, trans_b);
#endif
      }
      if (compute_capability >= 80) {
        BatchedGemmStridedTest<phi::bfloat16>(3, 5, 2, 4, trans_a, trans_b);
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
        BatchedGemmStridedTest<phi::bfloat16, float>(
            3, 5, 2, 4, trans_a, trans_b);
#endif
      }
    }
  }
}

}  // namespace tests
}  // namespace phi

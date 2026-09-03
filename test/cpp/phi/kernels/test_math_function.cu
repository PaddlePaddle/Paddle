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
#include <limits>

#include "gtest/gtest.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/cuda_stream.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
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

// ---------------------------------------------------------------------------
// Helpers for non-default-stream cuBLAS tests
// ---------------------------------------------------------------------------
namespace {

// Build a fully-initialized GPUContext bound to an externally-created stream.
// The caller owns the returned context and must call ctx->Wait() before
// reading results.
std::unique_ptr<phi::GPUContext> MakeCtxOnStream(const phi::GPUPlace& place,
                                                 gpuStream_t raw_stream) {
  // init=true: let the context fully initialize on its own default stream
  // (PartialInitWithoutAllocator + allocators + PartialInitWithAllocator).
  // Then switch to the target stream via SetCUDAStream, which also re-binds
  // the allocator internally.
  auto ctx = std::make_unique<phi::GPUContext>(place, /*init=*/true);
  ctx->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                        .GetAllocator(place, ctx->stream())
                        .get());
  ctx->SetHostAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(phi::CPUPlace())
                            .get());
  ctx->SetZeroAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetZeroAllocator(place)
                            .get());
  ctx->SetHostZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(phi::CPUPlace())
          .get());
  ctx->SetPinnedAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::GPUPinnedPlace())
          .get());
  ctx->PartialInitWithAllocator();
  // Switch to the desired stream. SetCUDAStream re-binds the allocator to
  // raw_stream and deletes the context's internally-created stream
  // (clear=true). The CUDAStream wrapper does NOT own raw_stream
  // (owned_=false), so ~CUDAStream() will not call cudaStreamDestroy on it.
  ctx->SetCUDAStream(new phi::CUDAStream(place, raw_stream), /*clear=*/true);
  return ctx;
}

// Compute C = A * B^T on CPU for a small 2x3 matrix and return the 4 entries
// of the 2x2 result as a vector, using float arithmetic.
// Input: row-major float arrays of shape [2,3].
std::vector<float> CpuMatMulNotransTransResult(const float* a, const float* b) {
  // C[i][j] = sum_k A[i][k] * B[j][k]
  std::vector<float> c(4, 0.0f);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 3; ++k) c[i * 2 + j] += a[i * 3 + k] * b[j * 3 + k];
  return c;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Test 1: CublasCall on a non-default stream produces correct GEMM results.
//
// Scenario: switch a freshly constructed GPUContext to an auxiliary stream
// using SetCUDAStream, then run MatMul through GetBlas (which internally calls
// CublasCall / TensorCoreCublasCallIfAvailable).  The cuBLAS handle must be
// re-bound to the auxiliary stream by SetBlasStream; without this fix the
// handle would remain on the original default stream and the operation would
// execute on the wrong stream, producing either incorrect results (data race)
// or zeros.
// ---------------------------------------------------------------------------
TEST(math_function, cublas_non_default_stream_correct_result) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  return;
#endif

  phi::GPUPlace gpu_place(0);
  phi::CPUPlace cpu_place;

  // Reference result computed on the default-stream context from the pool.
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* default_ctx =
      reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  float arr[6] = {0, 1, 2, 3, 4, 5};
  phi::DenseTensor input_cpu;
  float* input_ptr = input_cpu.mutable_data<float>({2, 3}, cpu_place);
  memcpy(input_ptr, arr, 6 * sizeof(float));

  phi::DenseTensor input_gpu_a, input_gpu_b, out_ref_gpu, out_ref_cpu;
  phi::Copy(
      *default_ctx, input_cpu, gpu_place, /*blocking=*/true, &input_gpu_a);
  phi::Copy(
      *default_ctx, input_cpu, gpu_place, /*blocking=*/true, &input_gpu_b);
  out_ref_gpu.mutable_data<float>({2, 2}, gpu_place);
  GetBlas<float>(*default_ctx)
      .MatMul(input_gpu_a, false, input_gpu_b, true, 1.0f, &out_ref_gpu, 0.0f);
  phi::Copy(
      *default_ctx, out_ref_gpu, cpu_place, /*blocking=*/true, &out_ref_cpu);
  default_ctx->Wait();

  const float* ref = out_ref_cpu.data<float>();
  // Verify reference against CPU calculation.
  auto cpu_ref = CpuMatMulNotransTransResult(arr, arr);
  ASSERT_FLOAT_EQ(ref[0], cpu_ref[0]);
  ASSERT_FLOAT_EQ(ref[1], cpu_ref[1]);
  ASSERT_FLOAT_EQ(ref[2], cpu_ref[2]);
  ASSERT_FLOAT_EQ(ref[3], cpu_ref[3]);

  // Now run the same GEMM on an auxiliary stream.
#ifdef PADDLE_WITH_HIP
  hipStream_t aux_raw;
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&aux_raw));
#else
  cudaStream_t aux_raw;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&aux_raw));
#endif

  auto aux_ctx = MakeCtxOnStream(gpu_place, aux_raw);

  phi::DenseTensor input_aux_a, input_aux_b, out_aux_gpu, out_aux_cpu;
  phi::Copy(*aux_ctx, input_cpu, gpu_place, /*blocking=*/true, &input_aux_a);
  phi::Copy(*aux_ctx, input_cpu, gpu_place, /*blocking=*/true, &input_aux_b);
  out_aux_gpu.mutable_data<float>({2, 2}, gpu_place);
  GetBlas<float>(*aux_ctx).MatMul(
      input_aux_a, false, input_aux_b, true, 1.0f, &out_aux_gpu, 0.0f);
  phi::Copy(*aux_ctx, out_aux_gpu, cpu_place, /*blocking=*/true, &out_aux_cpu);
  aux_ctx->Wait();

  const float* aux = out_aux_cpu.data<float>();
  EXPECT_FLOAT_EQ(aux[0], ref[0]);
  EXPECT_FLOAT_EQ(aux[1], ref[1]);
  EXPECT_FLOAT_EQ(aux[2], ref[2]);
  EXPECT_FLOAT_EQ(aux[3], ref[3]);

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(aux_raw));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(aux_raw));
#endif
}

// ---------------------------------------------------------------------------
// Test 2: Multiple stream switches — the stream cache invalidates correctly.
//
// Switch the same GPUContext between two auxiliary streams multiple times and
// verify the GEMM result is always correct.  This exercises the
// blas_handle_stream_ cache path: stream changes must trigger a new
// cublasSetStream_v2 call, while repeated calls on the same stream must use
// the cached value (no spurious rebinds).
// ---------------------------------------------------------------------------
TEST(math_function, cublas_stream_switch_cache_correctness) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  return;
#endif

  phi::GPUPlace gpu_place(0);
  phi::CPUPlace cpu_place;

  float arr[6] = {0, 1, 2, 3, 4, 5};
  phi::DenseTensor input_cpu;
  float* input_ptr = input_cpu.mutable_data<float>({2, 3}, cpu_place);
  memcpy(input_ptr, arr, 6 * sizeof(float));

  auto cpu_ref = CpuMatMulNotransTransResult(arr, arr);

#ifdef PADDLE_WITH_HIP
  hipStream_t stream_a, stream_b;
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream_a));
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream_b));
#else
  cudaStream_t stream_a, stream_b;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream_a));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream_b));
#endif

  // Create a single GPUContext and repeatedly switch it between stream_a and
  // stream_b.  This exercises the blas_handle_stream_ cache: each switch must
  // trigger SetBlasStream (cache miss), while consecutive calls on the same
  // stream must use the cached binding (no spurious rebind).  Creating a fresh
  // context per iteration would reset blas_handle_stream_ to nullptr every
  // time and never exercise the "same handle, different stream" cache path.
  auto ctx = MakeCtxOnStream(gpu_place, stream_a);

  for (int iter = 0; iter < 4; ++iter) {
    gpuStream_t cur_raw = (iter % 2 == 0) ? stream_a : stream_b;
    ctx->SetCUDAStream(new phi::CUDAStream(gpu_place, cur_raw), /*clear=*/true);

    phi::DenseTensor in_a, in_b, out_gpu, out_cpu;
    phi::Copy(*ctx, input_cpu, gpu_place, /*blocking=*/true, &in_a);
    phi::Copy(*ctx, input_cpu, gpu_place, /*blocking=*/true, &in_b);
    out_gpu.mutable_data<float>({2, 2}, gpu_place);
    GetBlas<float>(*ctx).MatMul(in_a, false, in_b, true, 1.0f, &out_gpu, 0.0f);
    phi::Copy(*ctx, out_gpu, cpu_place, /*blocking=*/true, &out_cpu);
    ctx->Wait();

    const float* result = out_cpu.data<float>();
    EXPECT_FLOAT_EQ(result[0], cpu_ref[0]) << "iter=" << iter;
    EXPECT_FLOAT_EQ(result[1], cpu_ref[1]) << "iter=" << iter;
    EXPECT_FLOAT_EQ(result[2], cpu_ref[2]) << "iter=" << iter;
    EXPECT_FLOAT_EQ(result[3], cpu_ref[3]) << "iter=" << iter;
  }

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream_a));
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream_b));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream_a));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream_b));
#endif
}

// ---------------------------------------------------------------------------
// Test 3: Two independent streams running GEMM concurrently do not corrupt
//         each other's results.
//
// Launch GEMM on stream_a and stream_b simultaneously (GPU-side concurrency),
// then synchronize both and verify each produces the correct answer.  This
// confirms that distinct GPUContext instances maintain independent cuBLAS
// handles with independent stream bindings.
// ---------------------------------------------------------------------------
TEST(math_function, cublas_concurrent_streams_independent) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  return;
#endif

  phi::GPUPlace gpu_place(0);
  phi::CPUPlace cpu_place;

  // Matrix A: {0..5}, Matrix B: {5..0} — different inputs so result differs.
  float arr_a[6] = {0, 1, 2, 3, 4, 5};
  float arr_b[6] = {5, 4, 3, 2, 1, 0};

  phi::DenseTensor cpu_a, cpu_b;
  memcpy(
      cpu_a.mutable_data<float>({2, 3}, cpu_place), arr_a, 6 * sizeof(float));
  memcpy(
      cpu_b.mutable_data<float>({2, 3}, cpu_place), arr_b, 6 * sizeof(float));

  auto ref_aa = CpuMatMulNotransTransResult(arr_a, arr_a);  // A * A^T
  auto ref_bb = CpuMatMulNotransTransResult(arr_b, arr_b);  // B * B^T

#ifdef PADDLE_WITH_HIP
  hipStream_t raw_sa, raw_sb;
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&raw_sa));
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&raw_sb));
#else
  cudaStream_t raw_sa, raw_sb;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&raw_sa));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&raw_sb));
#endif

  auto ctx_a = MakeCtxOnStream(gpu_place, raw_sa);
  auto ctx_b = MakeCtxOnStream(gpu_place, raw_sb);

  // Upload inputs.
  phi::DenseTensor ga_in_a, ga_in_b, gb_in_a, gb_in_b;
  phi::Copy(*ctx_a, cpu_a, gpu_place, true, &ga_in_a);
  phi::Copy(*ctx_a, cpu_a, gpu_place, true, &ga_in_b);
  phi::Copy(*ctx_b, cpu_b, gpu_place, true, &gb_in_a);
  phi::Copy(*ctx_b, cpu_b, gpu_place, true, &gb_in_b);

  // Launch both GEMMs without synchronizing between them.
  phi::DenseTensor out_a_gpu, out_b_gpu;
  out_a_gpu.mutable_data<float>({2, 2}, gpu_place);
  out_b_gpu.mutable_data<float>({2, 2}, gpu_place);
  GetBlas<float>(*ctx_a).MatMul(
      ga_in_a, false, ga_in_b, true, 1.0f, &out_a_gpu, 0.0f);
  GetBlas<float>(*ctx_b).MatMul(
      gb_in_a, false, gb_in_b, true, 1.0f, &out_b_gpu, 0.0f);

  // Copy results back and synchronize.
  phi::DenseTensor out_a_cpu, out_b_cpu;
  phi::Copy(*ctx_a, out_a_gpu, cpu_place, true, &out_a_cpu);
  phi::Copy(*ctx_b, out_b_gpu, cpu_place, true, &out_b_cpu);
  ctx_a->Wait();
  ctx_b->Wait();

  const float* ra = out_a_cpu.data<float>();
  const float* rb = out_b_cpu.data<float>();

  EXPECT_FLOAT_EQ(ra[0], ref_aa[0]);
  EXPECT_FLOAT_EQ(ra[1], ref_aa[1]);
  EXPECT_FLOAT_EQ(ra[2], ref_aa[2]);
  EXPECT_FLOAT_EQ(ra[3], ref_aa[3]);

  EXPECT_FLOAT_EQ(rb[0], ref_bb[0]);
  EXPECT_FLOAT_EQ(rb[1], ref_bb[1]);
  EXPECT_FLOAT_EQ(rb[2], ref_bb[2]);
  EXPECT_FLOAT_EQ(rb[3], ref_bb[3]);

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(raw_sa));
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(raw_sb));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(raw_sa));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(raw_sb));
#endif
}

}  // namespace tests
}  // namespace phi

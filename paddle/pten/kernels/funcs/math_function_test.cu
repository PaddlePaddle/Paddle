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
#include "gtest/gtest.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/kernels/funcs/math_function.h"

void fill_fp16_data(paddle::platform::float16* in_ptr,
                    size_t size,
                    const std::vector<float>& data) {
  PADDLE_ENFORCE_EQ(
      size,
      data.size(),
      paddle::platform::errors::InvalidArgument(
          "The size of argument data should"
          " be equal to the argument size. Expected %d, but received %d.",
          size,
          data.size()));
  for (size_t i = 0; i < data.size(); ++i) {
    in_ptr[i] = paddle::platform::float16(data[i]);
  }
}

template <typename T>
inline paddle::operators::math::BlasT<paddle::platform::CUDADeviceContext, T>
GetBlas(const paddle::platform::CUDADeviceContext& context) {
  return paddle::operators::math::GetBlas<paddle::platform::CUDADeviceContext,
                                          T>(context);
}

TEST(math_function, notrans_mul_trans_fp32) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor out_gpu;
  paddle::framework::Tensor out;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input1, gpu_place, &input2_gpu);

  out_gpu.mutable_data<float>({2, 2}, gpu_place);
  GetBlas<float>(context).MatMul(
      input1_gpu, false, input2_gpu, true, 1, &out_gpu, 0);

  paddle::framework::TensorCopySync(out_gpu, cpu_place, &out);

  float* out_ptr = out.data<float>();
  context.Wait();
  EXPECT_EQ(out_ptr[0], 5);
  EXPECT_EQ(out_ptr[1], 14);
  EXPECT_EQ(out_ptr[2], 14);
  EXPECT_EQ(out_ptr[3], 50);
}

TEST(math_function, notrans_mul_trans_fp16) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor out_gpu;
  paddle::framework::Tensor out;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  paddle::platform::float16* input1_ptr =
      input1.mutable_data<paddle::platform::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input1, gpu_place, &input2_gpu);

  out_gpu.mutable_data<paddle::platform::float16>({2, 2}, gpu_place);

  GetBlas<paddle::platform::float16>(context).MatMul(
      input1_gpu,
      false,
      input2_gpu,
      true,
      paddle::platform::float16(1),
      &out_gpu,
      paddle::platform::float16(0));

  paddle::framework::TensorCopySync(out_gpu, cpu_place, &out);

  paddle::platform::float16* out_ptr = out.data<paddle::platform::float16>();
  context.Wait();
  EXPECT_EQ(static_cast<float>(out_ptr[0]), 5);
  EXPECT_EQ(static_cast<float>(out_ptr[1]), 14);
  EXPECT_EQ(static_cast<float>(out_ptr[2]), 14);
  EXPECT_EQ(static_cast<float>(out_ptr[3]), 50);
}

TEST(math_function, trans_mul_notrans_fp32) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor out_gpu;
  paddle::framework::Tensor out;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input1, gpu_place, &input2_gpu);

  out_gpu.mutable_data<float>({3, 3}, gpu_place);

  GetBlas<float>(context).MatMul(
      input1_gpu, true, input2_gpu, false, 1, &out_gpu, 0);

  paddle::framework::TensorCopySync(out_gpu, cpu_place, &out);

  float* out_ptr = out.data<float>();
  context.Wait();
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
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor out_gpu;
  paddle::framework::Tensor out;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  paddle::platform::float16* input1_ptr =
      input1.mutable_data<paddle::platform::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input1, gpu_place, &input2_gpu);

  out_gpu.mutable_data<paddle::platform::float16>({3, 3}, gpu_place);

  GetBlas<paddle::platform::float16>(context).MatMul(
      input1_gpu,
      true,
      input2_gpu,
      false,
      paddle::platform::float16(1),
      &out_gpu,
      paddle::platform::float16(0));

  paddle::framework::TensorCopySync(out_gpu, cpu_place, &out);

  paddle::platform::float16* out_ptr = out.data<paddle::platform::float16>();
  context.Wait();
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
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor input3_gpu;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

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

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input2, gpu_place, &input2_gpu);
  paddle::framework::TensorCopySync(input3, gpu_place, &input3_gpu);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(gpu_place);

  GetBlas<float>(context).GEMM(
      false, false, m, n, k, 1, a, 3, b + 1, 4, 1, c + 1, 4);

  paddle::framework::TensorCopySync(input3_gpu, cpu_place, &input3);

  // numpy code:
  // a = np.arange(6).reshape(2, 3)
  // b = np.arange(12).reshape(3, 4)[:, 1:]
  // c = np.arange(8).reshape(2, 4)[:, 1:]
  // out = np.arange(8).reshape(2, 4)
  // out[:, 1:] = np.dot(a, b) + c
  context.Wait();
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
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor input3_gpu;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  int m = 2;
  int n = 3;
  int k = 3;
  paddle::platform::float16* input1_ptr =
      input1.mutable_data<paddle::platform::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});
  paddle::platform::float16* input2_ptr =
      input2.mutable_data<paddle::platform::float16>({3, 4}, cpu_place);
  fill_fp16_data(
      input2_ptr, input2.numel(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  paddle::platform::float16* input3_ptr =
      input3.mutable_data<paddle::platform::float16>({2, 4}, cpu_place);
  fill_fp16_data(input3_ptr, input3.numel(), {0, 1, 2, 3, 4, 5, 6, 7});

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input2, gpu_place, &input2_gpu);
  paddle::framework::TensorCopySync(input3, gpu_place, &input3_gpu);
  paddle::platform::float16* a = input1_gpu.data<paddle::platform::float16>();
  paddle::platform::float16* b = input2_gpu.data<paddle::platform::float16>();
  paddle::platform::float16* c =
      input3_gpu.mutable_data<paddle::platform::float16>(gpu_place);

  GetBlas<paddle::platform::float16>(context).GEMM(
      false,
      false,
      m,
      n,
      k,
      static_cast<paddle::platform::float16>(1),
      a,
      3,
      b + 1,
      4,
      static_cast<paddle::platform::float16>(1),
      c + 1,
      4);

  paddle::framework::TensorCopySync(input3_gpu, cpu_place, &input3);

  // numpy code:
  // a = np.arange(6).reshape(2, 3)
  // b = np.arange(12).reshape(3, 4)[:, 1:]
  // c = np.arange(8).reshape(2, 4)[:, 1:]
  // out = np.arange(8).reshape(2, 4)
  // out[:, 1:] = np.dot(a, b) + c
  context.Wait();
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
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor input3_gpu;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

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

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input2, gpu_place, &input2_gpu);
  paddle::framework::TensorCopySync(input3, gpu_place, &input3_gpu);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(gpu_place);

  GetBlas<float>(context).GEMM(
      false, true, m, n, k, 1, a, 3, b + 3, 3, 1, c + 1, 4);

  paddle::framework::TensorCopySync(input3_gpu, cpu_place, &input3);

  context.Wait();
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
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor input3_gpu;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  int m = 2;
  int n = 3;
  int k = 3;
  paddle::platform::float16* input1_ptr =
      input1.mutable_data<paddle::platform::float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});
  paddle::platform::float16* input2_ptr =
      input2.mutable_data<paddle::platform::float16>({4, 3}, cpu_place);
  fill_fp16_data(
      input2_ptr, input2.numel(), {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11});
  paddle::platform::float16* input3_ptr =
      input3.mutable_data<paddle::platform::float16>({2, 4}, cpu_place);
  fill_fp16_data(input3_ptr, input3.numel(), {0, 1, 2, 3, 4, 5, 6, 7});

  paddle::framework::TensorCopySync(input1, gpu_place, &input1_gpu);
  paddle::framework::TensorCopySync(input2, gpu_place, &input2_gpu);
  paddle::framework::TensorCopySync(input3, gpu_place, &input3_gpu);
  paddle::platform::float16* a = input1_gpu.data<paddle::platform::float16>();
  paddle::platform::float16* b = input2_gpu.data<paddle::platform::float16>();
  paddle::platform::float16* c =
      input3_gpu.mutable_data<paddle::platform::float16>(gpu_place);

  GetBlas<paddle::platform::float16>(context).GEMM(
      false,
      true,
      m,
      n,
      k,
      static_cast<paddle::platform::float16>(1),
      a,
      3,
      b + 3,
      3,
      static_cast<paddle::platform::float16>(1),
      c + 1,
      4);

  paddle::framework::TensorCopySync(input3_gpu, cpu_place, &input3);

  context.Wait();
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
void GemvTest(int m, int n, bool trans) {
  paddle::framework::Tensor mat_a;
  paddle::framework::Tensor vec_b;
  paddle::framework::Tensor vec_c;

  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CUDADeviceContext context(gpu_place);
  context.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu_place, context.stream())
                           .get());
  context.PartialInitWithAllocator();

  T* data_a = mat_a.mutable_data<T>({m, n}, cpu_place);
  T* data_b = vec_b.mutable_data<T>({trans ? m : n}, cpu_place);
  T* data_c = vec_c.mutable_data<T>({trans ? n : m}, cpu_place);

  paddle::framework::Tensor g_mat_a;
  paddle::framework::Tensor g_vec_b;
  paddle::framework::Tensor g_vec_c;
  T* g_data_a = g_mat_a.mutable_data<T>(mat_a.dims(), gpu_place);
  T* g_data_b = g_vec_b.mutable_data<T>(vec_b.dims(), gpu_place);
  T* g_data_c = g_vec_c.mutable_data<T>(vec_c.dims(), gpu_place);

  for (int i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = static_cast<T>(i);
  }
  for (int i = 0; i < vec_b.numel(); ++i) {
    data_b[i] = static_cast<T>(i);
  }

  paddle::framework::TensorCopySync(mat_a, gpu_place, &g_mat_a);
  paddle::framework::TensorCopySync(vec_b, gpu_place, &g_vec_b);

  GetBlas<T>(context).GEMV(trans,
                           static_cast<int>(m),
                           static_cast<int>(n),
                           1.,
                           g_data_a,
                           g_data_b,
                           0.,
                           g_data_c);

  paddle::framework::TensorCopySync(g_vec_c, cpu_place, &vec_c);

  if (!trans) {
    for (int i = 0; i < m; ++i) {
      T sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += data_a[i * n + j] * data_b[j];
      }
      ASSERT_FLOAT_EQ(data_c[i], sum);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      T sum = 0.0;
      for (int j = 0; j < m; ++j) {
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

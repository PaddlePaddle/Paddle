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
#include "paddle/fluid/operators/math/math_function.h"

void fill_fp16_data(paddle::platform::float16* in_ptr, size_t size,
                    const std::vector<float>& data) {
  PADDLE_ENFORCE_EQ(size, data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    in_ptr[i] = paddle::platform::float16(data[i]);
  }
}

TEST(math_function, notrans_mul_trans_fp32) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor out_gpu;
  Tensor out;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input1, gpu_place, context, &input2_gpu);

  out_gpu.mutable_data<float>({2, 2}, gpu_place);

  paddle::operators::math::matmul<CUDADeviceContext, float>(
      context, input1_gpu, false, input2_gpu, true, 1, &out_gpu, 0);

  TensorCopy(out_gpu, cpu_place, context, &out);

  float* out_ptr = out.data<float>();
  context.Wait();
  EXPECT_EQ(out_ptr[0], 5);
  EXPECT_EQ(out_ptr[1], 14);
  EXPECT_EQ(out_ptr[2], 14);
  EXPECT_EQ(out_ptr[3], 50);
}

TEST(math_function, notrans_mul_trans_fp16) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor out_gpu;
  Tensor out;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  float16* input1_ptr = input1.mutable_data<float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input1, gpu_place, context, &input2_gpu);

  out_gpu.mutable_data<float16>({2, 2}, gpu_place);

  paddle::operators::math::matmul<CUDADeviceContext, float16>(
      context, input1_gpu, false, input2_gpu, true, float16(1), &out_gpu,
      float16(0));

  TensorCopy(out_gpu, cpu_place, context, &out);

  float16* out_ptr = out.data<float16>();
  context.Wait();
  EXPECT_EQ(static_cast<float>(out_ptr[0]), 5);
  EXPECT_EQ(static_cast<float>(out_ptr[1]), 14);
  EXPECT_EQ(static_cast<float>(out_ptr[2]), 14);
  EXPECT_EQ(static_cast<float>(out_ptr[3]), 50);
}

TEST(math_function, trans_mul_notrans_fp32) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor out_gpu;
  Tensor out;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

  float* input1_ptr = input1.mutable_data<float>({2, 3}, cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input1, gpu_place, context, &input2_gpu);

  out_gpu.mutable_data<float>({3, 3}, gpu_place);

  paddle::operators::math::matmul<paddle::platform::CUDADeviceContext, float>(
      context, input1_gpu, true, input2_gpu, false, 1, &out_gpu, 0);

  TensorCopy(out_gpu, cpu_place, context, &out);

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
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor out_gpu;
  Tensor out;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  float16* input1_ptr = input1.mutable_data<float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input1, gpu_place, context, &input2_gpu);

  out_gpu.mutable_data<float16>({3, 3}, gpu_place);

  paddle::operators::math::matmul<paddle::platform::CUDADeviceContext, float16>(
      context, input1_gpu, true, input2_gpu, false, float16(1), &out_gpu,
      float16(0));

  TensorCopy(out_gpu, cpu_place, context, &out);

  float16* out_ptr = out.data<float16>();
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
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input2;
  Tensor input3;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor input3_gpu;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

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

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input2, gpu_place, context, &input2_gpu);
  TensorCopy(input3, gpu_place, context, &input3_gpu);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(gpu_place);

  paddle::operators::math::gemm<paddle::platform::CUDADeviceContext, float>(
      context, false, false, m, n, k, 1, a, 3, b + 1, 4, 1, c + 1, 4);

  TensorCopy(input3_gpu, cpu_place, context, &input3);

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
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input2;
  Tensor input3;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor input3_gpu;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  int m = 2;
  int n = 3;
  int k = 3;
  float16* input1_ptr = input1.mutable_data<float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});
  float16* input2_ptr = input2.mutable_data<float16>({3, 4}, cpu_place);
  fill_fp16_data(input2_ptr, input2.numel(),
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  float16* input3_ptr = input3.mutable_data<float16>({2, 4}, cpu_place);
  fill_fp16_data(input3_ptr, input3.numel(), {0, 1, 2, 3, 4, 5, 6, 7});

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input2, gpu_place, context, &input2_gpu);
  TensorCopy(input3, gpu_place, context, &input3_gpu);
  float16* a = input1_gpu.data<float16>();
  float16* b = input2_gpu.data<float16>();
  float16* c = input3_gpu.mutable_data<float16>(gpu_place);

  paddle::operators::math::gemm<paddle::platform::CUDADeviceContext, float16>(
      context, false, false, m, n, k, float16(1), a, 3, b + 1, 4, float16(1),
      c + 1, 4);

  TensorCopy(input3_gpu, cpu_place, context, &input3);

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
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input2;
  Tensor input3;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor input3_gpu;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

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

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input2, gpu_place, context, &input2_gpu);
  TensorCopy(input3, gpu_place, context, &input3_gpu);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(gpu_place);

  paddle::operators::math::gemm<paddle::platform::CUDADeviceContext, float>(
      context, false, true, m, n, k, 1, a, 3, b + 3, 3, 1, c + 1, 4);

  TensorCopy(input3_gpu, cpu_place, context, &input3);

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
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor input1;
  Tensor input2;
  Tensor input3;
  Tensor input1_gpu;
  Tensor input2_gpu;
  Tensor input3_gpu;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

  // fp16 GEMM in cublas requires GPU compute capability >= 53
  if (context.GetComputeCapability() < 53) {
    return;
  }

  int m = 2;
  int n = 3;
  int k = 3;
  float16* input1_ptr = input1.mutable_data<float16>({2, 3}, cpu_place);
  fill_fp16_data(input1_ptr, input1.numel(), {0, 1, 2, 3, 4, 5});
  float16* input2_ptr = input2.mutable_data<float16>({4, 3}, cpu_place);
  fill_fp16_data(input2_ptr, input2.numel(),
                 {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11});
  float16* input3_ptr = input3.mutable_data<float16>({2, 4}, cpu_place);
  fill_fp16_data(input3_ptr, input3.numel(), {0, 1, 2, 3, 4, 5, 6, 7});

  TensorCopy(input1, gpu_place, context, &input1_gpu);
  TensorCopy(input2, gpu_place, context, &input2_gpu);
  TensorCopy(input3, gpu_place, context, &input3_gpu);
  float16* a = input1_gpu.data<float16>();
  float16* b = input2_gpu.data<float16>();
  float16* c = input3_gpu.mutable_data<float16>(gpu_place);

  paddle::operators::math::gemm<paddle::platform::CUDADeviceContext, float16>(
      context, false, true, m, n, k, float16(1), a, 3, b + 3, 3, float16(1),
      c + 1, 4);

  TensorCopy(input3_gpu, cpu_place, context, &input3);

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
  using namespace paddle::framework;
  using namespace paddle::platform;

  Tensor mat_a;
  Tensor vec_b;
  Tensor vec_c;

  CPUPlace cpu_place;
  CUDAPlace gpu_place(0);
  CUDADeviceContext context(gpu_place);

  T* data_a = mat_a.mutable_data<T>({m, n}, cpu_place);
  T* data_b = vec_b.mutable_data<T>({trans ? m : n}, cpu_place);
  T* data_c = vec_c.mutable_data<T>({trans ? n : m}, cpu_place);

  Tensor g_mat_a;
  Tensor g_vec_b;
  Tensor g_vec_c;
  T* g_data_a = g_mat_a.mutable_data<T>(mat_a.dims(), gpu_place);
  T* g_data_b = g_vec_b.mutable_data<T>(vec_b.dims(), gpu_place);
  T* g_data_c = g_vec_c.mutable_data<T>(vec_c.dims(), gpu_place);

  for (int i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = static_cast<T>(i);
  }
  for (int i = 0; i < vec_b.numel(); ++i) {
    data_b[i] = static_cast<T>(i);
  }

  TensorCopy(mat_a, gpu_place, context, &g_mat_a);
  TensorCopy(vec_b, gpu_place, context, &g_vec_b);

  paddle::operators::math::gemv<CUDADeviceContext, T>(
      context, trans, static_cast<int>(m), static_cast<int>(n), 1., g_data_a,
      g_data_b, 0., g_data_c);

  TensorCopy(g_vec_c, cpu_place, context, &vec_c);

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

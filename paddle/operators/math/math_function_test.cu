#include "gtest/gtest.h"
#include "paddle/operators/math/math_function.h"

TEST(math_function, notrans_mul_trans) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor out_gpu;
  paddle::framework::Tensor out;

  auto* cpu_place = new paddle::platform::CPUPlace();
  float* input1_ptr = input1.mutable_data<float>({2, 3}, *cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  auto* gpu_place = new paddle::platform::GPUPlace(0);
  paddle::platform::CUDADeviceContext context(*gpu_place);

  input1_gpu.CopyFrom(input1, *gpu_place, context);
  input2_gpu.CopyFrom(input1, *gpu_place, context);

  out_gpu.mutable_data<float>({2, 2}, *gpu_place);

  paddle::operators::math::matmul<paddle::platform::GPUPlace, float>(
      context, input1_gpu, false, input2_gpu, true, 1, &out_gpu, 0);

  out.CopyFrom(out_gpu, *cpu_place, context);

  float* out_ptr = out.data<float>();
  context.Wait();
  EXPECT_EQ(out_ptr[0], 5);
  EXPECT_EQ(out_ptr[1], 14);
  EXPECT_EQ(out_ptr[2], 14);
  EXPECT_EQ(out_ptr[3], 50);
  delete gpu_place;
}

TEST(math_function, trans_mul_notrans) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor out_gpu;
  paddle::framework::Tensor out;

  auto* cpu_place = new paddle::platform::CPUPlace();
  float* input1_ptr = input1.mutable_data<float>({2, 3}, *cpu_place);
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr, 6 * sizeof(float));

  auto* gpu_place = new paddle::platform::GPUPlace(0);
  paddle::platform::CUDADeviceContext context(*gpu_place);

  input1_gpu.CopyFrom(input1, *gpu_place, context);
  input2_gpu.CopyFrom(input1, *gpu_place, context);

  out_gpu.mutable_data<float>({3, 3}, *gpu_place);

  paddle::operators::math::matmul<paddle::platform::GPUPlace, float>(
      context, input1_gpu, true, input2_gpu, false, 1, &out_gpu, 0);

  out.CopyFrom(out_gpu, *cpu_place, context);

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
  delete gpu_place;
}

TEST(math_function, gemm_notrans_cublas) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor input3_gpu;

  int m = 2;
  int n = 3;
  int k = 3;
  auto* cpu_place = new paddle::platform::CPUPlace();
  float* input1_ptr = input1.mutable_data<float>({2, 3}, *cpu_place);
  float arr1[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1, 6 * sizeof(float));
  float* input2_ptr = input2.mutable_data<float>({3, 4}, *cpu_place);
  float arr2[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  memcpy(input2_ptr, arr2, 12 * sizeof(float));
  float* input3_ptr = input3.mutable_data<float>({2, 4}, *cpu_place);
  float arr3[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3, 8 * sizeof(float));

  auto* gpu_place = new paddle::platform::GPUPlace(0);
  paddle::platform::CUDADeviceContext context(*gpu_place);

  input1_gpu.CopyFrom(input1, *gpu_place, context);
  input2_gpu.CopyFrom(input2, *gpu_place, context);
  input3_gpu.CopyFrom(input3, *gpu_place, context);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(*gpu_place);

  paddle::operators::math::gemm<paddle::platform::GPUPlace, float>(
      context, false, false, m, n, k, 1, a, 3, b + 1, 4, 1, c + 1, 4);

  input3.CopyFrom(input3_gpu, *cpu_place, context);

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
  delete gpu_place;
}

TEST(math_function, gemm_trans_cublas) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor input3_gpu;

  int m = 2;
  int n = 3;
  int k = 3;
  auto* cpu_place = new paddle::platform::CPUPlace();
  float* input1_ptr = input1.mutable_data<float>({2, 3}, *cpu_place);
  float arr1[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1, 6 * sizeof(float));
  float* input2_ptr = input2.mutable_data<float>({4, 3}, *cpu_place);
  float arr2[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  memcpy(input2_ptr, arr2, 12 * sizeof(float));
  float* input3_ptr = input3.mutable_data<float>({2, 4}, *cpu_place);
  float arr3[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3, 8 * sizeof(float));

  auto* gpu_place = new paddle::platform::GPUPlace(0);
  paddle::platform::CUDADeviceContext context(*gpu_place);

  input1_gpu.CopyFrom(input1, *gpu_place, context);
  input2_gpu.CopyFrom(input2, *gpu_place, context);
  input3_gpu.CopyFrom(input3, *gpu_place, context);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(*gpu_place);

  paddle::operators::math::gemm<paddle::platform::GPUPlace, float>(
      context, false, true, m, n, k, 1, a, 3, b + 3, 3, 1, c + 1, 4);

  input3.CopyFrom(input3_gpu, *cpu_place, context);
  context.Wait();

  EXPECT_EQ(input3_ptr[0], 0);
  EXPECT_EQ(input3_ptr[1], 24);
  EXPECT_EQ(input3_ptr[2], 28);
  EXPECT_EQ(input3_ptr[3], 32);
  EXPECT_EQ(input3_ptr[4], 4);
  EXPECT_EQ(input3_ptr[5], 73);
  EXPECT_EQ(input3_ptr[6], 86);
  EXPECT_EQ(input3_ptr[7], 99);
  delete gpu_place;
}

template <typename T>
void GemvTest(int m, int n, bool trans) {
  paddle::framework::Tensor mat_a;
  paddle::framework::Tensor vec_b;
  paddle::framework::Tensor vec_c;
  auto* cpu_place = new paddle::platform::CPUPlace();

  T* data_a = mat_a.mutable_data<T>({m, n}, *cpu_place);
  T* data_b = vec_b.mutable_data<T>({trans ? m : n}, *cpu_place);
  T* data_c = vec_c.mutable_data<T>({trans ? n : m}, *cpu_place);

  auto* gpu_place = new paddle::platform::GPUPlace(0);
  paddle::framework::Tensor g_mat_a;
  paddle::framework::Tensor g_vec_b;
  paddle::framework::Tensor g_vec_c;
  T* g_data_a = g_mat_a.mutable_data<T>(mat_a.dims(), *gpu_place);
  T* g_data_b = g_vec_b.mutable_data<T>(vec_b.dims(), *gpu_place);
  T* g_data_c = g_vec_c.mutable_data<T>(vec_c.dims(), *gpu_place);

  for (int i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = static_cast<T>(i);
  }
  for (int i = 0; i < vec_b.numel(); ++i) {
    data_b[i] = static_cast<T>(i);
  }

  paddle::platform::CUDADeviceContext context(*gpu_place);
  g_mat_a.CopyFrom(mat_a, *gpu_place, context);
  g_vec_b.CopyFrom(vec_b, *gpu_place, context);

  paddle::operators::math::gemv<paddle::platform::GPUPlace, T>(
      context, trans, static_cast<int>(m), static_cast<int>(n), 1., g_data_a,
      g_data_b, 0., g_data_c);

  vec_c.CopyFrom(g_vec_c, paddle::platform::CPUPlace(), context);

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

#include "paddle/operators/math/math_function.h"
#include "gtest/gtest.h"

#ifndef PADDLE_ONLY_CPU
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

  input1_gpu.CopyFrom<float>(input1, *gpu_place);
  input2_gpu.CopyFrom<float>(input1, *gpu_place);

  out_gpu.mutable_data<float>({2, 2}, *gpu_place);

  paddle::operators::math::matmul<paddle::platform::GPUPlace, float>(
      context, input1_gpu, false, input2_gpu, true, 1, &out_gpu, 0);

  out.CopyFrom<float>(out_gpu, *cpu_place);

  float* out_ptr = out.data<float>();
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

  input1_gpu.CopyFrom<float>(input1, *gpu_place);
  input2_gpu.CopyFrom<float>(input1, *gpu_place);

  out_gpu.mutable_data<float>({3, 3}, *gpu_place);

  paddle::operators::math::matmul<paddle::platform::GPUPlace, float>(
      context, input1_gpu, true, input2_gpu, false, 1, &out_gpu, 0);

  out.CopyFrom<float>(out_gpu, *cpu_place);

  float* out_ptr = out.data<float>();
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
#endif

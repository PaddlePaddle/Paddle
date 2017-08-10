#include "paddle/operators/math/math_function.h"
#include "gtest/gtest.h"

#ifndef PADDLE_ONLY_CPU
TEST(math_function, GPU) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input1_gpu;
  paddle::framework::Tensor input2_gpu;
  paddle::framework::Tensor out_gpu;
  paddle::framework::Tensor out;

  auto* cpu_place = new paddle::platform::CPUPlace();
  float* input1_ptr = input1.mutable_data<float>({2, 2}, *cpu_place);
  float arr[4] = {0, 1, 2, 3};
  memcpy(input1_ptr, arr, 4 * sizeof(int));

  auto* gpu_place = new paddle::platform::GPUPlace(0);
  paddle::platform::DeviceContext* context =
      new paddle::platform::CUDADeviceContext(*gpu_place);

  input1_gpu.CopyFrom<float>(input1, *gpu_place);
  input2_gpu.CopyFrom<float>(input1, *gpu_place);
  out_gpu.CopyFrom<float>(input1, *gpu_place);

  paddle::operators::math::matmul<paddle::platform::GPUPlace, float>(
      input1_gpu, false, input2_gpu, false, 1, &out_gpu, 0, context);

  out.CopyFrom<float>(out_gpu, *cpu_place);

  float* out_ptr = out.data<float>();
  EXPECT_EQ(out_ptr[0], 2);
  EXPECT_EQ(out_ptr[1], 3);
  EXPECT_EQ(out_ptr[2], 6);
  EXPECT_EQ(out_ptr[3], 11);
}
#endif

#include "paddle/operators/math/math_function.h"
#include "gtest/gtest.h"

TEST(math_function, gemm_notrans_cblas) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;

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

  paddle::platform::CPUDeviceContext context(*cpu_place);
  paddle::operators::math::gemm<paddle::platform::CPUPlace, float>(
      context, false, false, m, n, k, 1, input1_ptr, 3, input2_ptr + 1, 4, 1,
      input3_ptr + 1, 4);

  EXPECT_EQ(input3_ptr[0], 0);
  EXPECT_EQ(input3_ptr[1], 24);
  EXPECT_EQ(input3_ptr[2], 28);
  EXPECT_EQ(input3_ptr[3], 32);
  EXPECT_EQ(input3_ptr[4], 4);
  EXPECT_EQ(input3_ptr[5], 73);
  EXPECT_EQ(input3_ptr[6], 86);
  EXPECT_EQ(input3_ptr[7], 99);
}

TEST(math_function, gemm_trans_clbas) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;

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

  paddle::platform::CPUDeviceContext context(*cpu_place);
  paddle::operators::math::gemm<paddle::platform::CPUPlace, float>(
      context, false, true, m, n, k, 1, input1_ptr, 3, input2_ptr + 3, 3, 1,
      input3_ptr + 1, 4);

  EXPECT_EQ(input3_ptr[0], 0);
  EXPECT_EQ(input3_ptr[1], 24);
  EXPECT_EQ(input3_ptr[2], 28);
  EXPECT_EQ(input3_ptr[3], 32);
  EXPECT_EQ(input3_ptr[4], 4);
  EXPECT_EQ(input3_ptr[5], 73);
  EXPECT_EQ(input3_ptr[6], 86);
  EXPECT_EQ(input3_ptr[7], 99);
}

TEST(math_function, zero) {
  paddle::framework::Tensor tensor;
  auto* cpu_place = new paddle::platform::CPUPlace();
  float* t = tensor.mutable_data<float>({2, 2}, *cpu_place);
  paddle::platform::CPUDeviceContext context(*cpu_place);
  paddle::operators::math::SetConstant<paddle::platform::CPUPlace, float>
      functor;
  functor(context, &tensor, 0);
  EXPECT_EQ(t[0], 0);
  EXPECT_EQ(t[1], 0);
  EXPECT_EQ(t[2], 0);
  EXPECT_EQ(t[3], 0);

  functor(context, &tensor, 1);

  EXPECT_EQ(t[0], 1);
  EXPECT_EQ(t[1], 1);
  EXPECT_EQ(t[2], 1);
  EXPECT_EQ(t[3], 1);
}

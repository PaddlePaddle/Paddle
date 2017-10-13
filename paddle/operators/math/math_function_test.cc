#include "paddle/operators/math/math_function.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#ifdef PADDLE_WITH_CUDA
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

  input1_gpu.CopyFrom<float>(input1, *gpu_place, context);
  input2_gpu.CopyFrom<float>(input1, *gpu_place, context);

  out_gpu.mutable_data<float>({2, 2}, *gpu_place);

  paddle::operators::math::matmul<paddle::platform::GPUPlace, float>(
      context, input1_gpu, false, input2_gpu, true, 1, &out_gpu, 0);

  out.CopyFrom<float>(out_gpu, *cpu_place, context);

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

  input1_gpu.CopyFrom<float>(input1, *gpu_place, context);
  input2_gpu.CopyFrom<float>(input1, *gpu_place, context);

  out_gpu.mutable_data<float>({3, 3}, *gpu_place);

  paddle::operators::math::matmul<paddle::platform::GPUPlace, float>(
      context, input1_gpu, true, input2_gpu, false, 1, &out_gpu, 0);

  out.CopyFrom<float>(out_gpu, *cpu_place, context);

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

  input1_gpu.CopyFrom<float>(input1, *gpu_place, context);
  input2_gpu.CopyFrom<float>(input2, *gpu_place, context);
  input3_gpu.CopyFrom<float>(input3, *gpu_place, context);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(*gpu_place);

  paddle::operators::math::gemm<paddle::platform::GPUPlace, float>(
      context, false, false, m, n, k, 1, a, 3, b + 1, 4, 1, c + 1, 4);

  input3.CopyFrom<float>(input3_gpu, *cpu_place, context);

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

  input1_gpu.CopyFrom<float>(input1, *gpu_place, context);
  input2_gpu.CopyFrom<float>(input2, *gpu_place, context);
  input3_gpu.CopyFrom<float>(input3, *gpu_place, context);
  float* a = input1_gpu.data<float>();
  float* b = input2_gpu.data<float>();
  float* c = input3_gpu.mutable_data<float>(*gpu_place);

  paddle::operators::math::gemm<paddle::platform::GPUPlace, float>(
      context, false, true, m, n, k, 1, a, 3, b + 3, 3, 1, c + 1, 4);

  input3.CopyFrom<float>(input3_gpu, *cpu_place, context);
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
#endif

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

TEST(math_function, selected_rows_add) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  using namespace paddle::operators::math;

  CPUPlace cpu_place;
  CPUDeviceContext ctx(cpu_place);
  SetConstant<CPUPlace, float> functor;
  int64_t height = 10;
  int64_t row_numel = 10;

  std::vector<int64_t> rows1{0, 4, 7};
  std::unique_ptr<SelectedRows> selected_rows1{new SelectedRows(rows1, height)};
  auto* in1_value = selected_rows1->mutable_value();
  in1_value->mutable_data<float>(
      make_ddim({static_cast<int64_t>(rows1.size()), row_numel}), cpu_place);
  functor(ctx, in1_value, 1.0);

  std::vector<int64_t> rows2{0, 5, 7, 9};
  std::unique_ptr<SelectedRows> selected_rows2{new SelectedRows(rows2, height)};
  auto* in2_value = selected_rows2->mutable_value();
  in2_value->mutable_data<float>(
      make_ddim({static_cast<int64_t>(rows2.size()), row_numel}), cpu_place);
  functor(ctx, in2_value, 2.0);

  std::unique_ptr<SelectedRows> output{new SelectedRows()};
  auto* out_value = output->mutable_value();

  // simplely concat two SelectedRows
  out_value->mutable_data<float>(make_ddim({7, 10}), cpu_place);

  SelectedRowsAdd<CPUPlace, float> add_functor;
  add_functor(ctx, *selected_rows1, *selected_rows2, output.get());

  auto out_height = output->height();
  EXPECT_EQ(out_height, height);

  auto& out_rows = output->rows();

  // input1 rows
  EXPECT_EQ(out_rows[0], 0);
  EXPECT_EQ(out_rows[1], 4);
  EXPECT_EQ(out_rows[2], 7);
  // input2 rows
  EXPECT_EQ(out_rows[3], 0);
  EXPECT_EQ(out_rows[4], 5);
  EXPECT_EQ(out_rows[5], 7);
  EXPECT_EQ(out_rows[6], 9);

  auto* out_data = output->value().data<float>();
  // input1 value
  EXPECT_EQ(out_data[0 * row_numel + 0], 1.0);
  EXPECT_EQ(out_data[0 * row_numel + 8], 1.0);
  EXPECT_EQ(out_data[1 * row_numel + 1], 1.0);
  EXPECT_EQ(out_data[2 * row_numel + 6], 1.0);
  // input2 value
  EXPECT_EQ(out_data[3 * row_numel + 3], 2.0);
  EXPECT_EQ(out_data[3 * row_numel + 8], 2.0);
  EXPECT_EQ(out_data[4 * row_numel + 4], 2.0);
  EXPECT_EQ(out_data[5 * row_numel + 7], 2.0);
  EXPECT_EQ(out_data[6 * row_numel + 9], 2.0);

  std::unique_ptr<Tensor> tensor1{new Tensor()};
  tensor1->mutable_data<float>(make_ddim({height, row_numel}), cpu_place);
  SetConstant<CPUPlace, float> constant_functor;
  constant_functor(ctx, tensor1.get(), 3.0);

  std::unique_ptr<Tensor> tensor2{new Tensor()};
  tensor2->mutable_data<float>(make_ddim({height, row_numel}), cpu_place);

  SelectedRowsAddTensor<CPUPlace, float> add_tensor_functor;
  add_tensor_functor(ctx, *output, *tensor1, tensor2.get());

  auto* tensor2_data = tensor2->data<float>();
  // row0: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor2_data[0 * row_numel + 0], 6.0);
  // row1: 3.0
  EXPECT_EQ(tensor2_data[1 * row_numel + 1], 3.0);
  // row4 : 1.0 + 3.0
  EXPECT_EQ(tensor2_data[4 * row_numel + 6], 4.0);
  // row5: 2.0 + 3.0
  EXPECT_EQ(tensor2_data[5 * row_numel + 7], 5.0);
  // row6: 3.0
  EXPECT_EQ(tensor2_data[6 * row_numel + 1], 3.0);
  // row7: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor2_data[7 * row_numel + 3], 6.0);
  // row9: 2.0 + 3.0
  EXPECT_EQ(tensor2_data[9 * row_numel + 6], 5.0);
}

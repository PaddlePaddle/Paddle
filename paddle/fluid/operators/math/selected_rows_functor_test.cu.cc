/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

#include "gtest/gtest.h"
#include "paddle/phi/kernels/funcs/math_function.h"

TEST(selected_rows_functor, gpu_add) {
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CPUPlace cpu_place;
  phi::GPUContext& ctx = *reinterpret_cast<phi::GPUContext*>(
      paddle::platform::DeviceContextPool::Instance().Get(gpu_place));
  phi::funcs::SetConstant<phi::GPUContext, float> functor;
  int64_t height = 10;
  int64_t row_numel = 10;

  std::vector<int64_t> rows1{0, 4, 7};
  std::unique_ptr<phi::SelectedRows> selected_rows1{
      new phi::SelectedRows(rows1, height)};
  auto* in1_value = selected_rows1->mutable_value();
  in1_value->mutable_data<float>(
      phi::make_ddim({static_cast<int64_t>(rows1.size()), row_numel}),
      gpu_place);
  functor(ctx, in1_value, 1.0);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_EQ(hipDeviceSynchronize(),
                    0,
                    paddle::platform::errors::PreconditionNotMet(
                        "The all synchronization on the cuda is error!"));
#else
  PADDLE_ENFORCE_EQ(cudaDeviceSynchronize(),
                    0,
                    paddle::platform::errors::PreconditionNotMet(
                        "The all synchronization on the cuda is error!"));
#endif

  std::vector<int64_t> rows2{0, 5, 7, 9};
  std::unique_ptr<phi::SelectedRows> selected_rows2{
      new phi::SelectedRows(rows2, height)};
  auto* in2_value = selected_rows2->mutable_value();
  in2_value->mutable_data<float>(
      phi::make_ddim({static_cast<int64_t>(rows2.size()), row_numel}),
      gpu_place);
  functor(ctx, in2_value, 2.0);

  std::unique_ptr<phi::SelectedRows> output{new phi::SelectedRows()};
  auto* out_value = output->mutable_value();

  // simply concat two SelectedRows
  out_value->mutable_data<float>(phi::make_ddim({7, 10}), gpu_place);

  phi::funcs::SelectedRowsAdd<phi::GPUContext, float> add_functor;
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

  phi::DenseTensor out_cpu;
  paddle::framework::TensorCopy(*out_value, cpu_place, ctx, &out_cpu);
  ctx.Wait();

  auto* out_cpu_data = out_cpu.data<float>();
  // input1 value
  EXPECT_EQ(out_cpu_data[0 * row_numel + 0], 1.0);
  EXPECT_EQ(out_cpu_data[0 * row_numel + 8], 1.0);
  EXPECT_EQ(out_cpu_data[1 * row_numel + 1], 1.0);
  EXPECT_EQ(out_cpu_data[2 * row_numel + 6], 1.0);
  // input2 value
  EXPECT_EQ(out_cpu_data[3 * row_numel + 3], 2.0);
  EXPECT_EQ(out_cpu_data[3 * row_numel + 8], 2.0);
  EXPECT_EQ(out_cpu_data[4 * row_numel + 4], 2.0);
  EXPECT_EQ(out_cpu_data[5 * row_numel + 7], 2.0);
  EXPECT_EQ(out_cpu_data[6 * row_numel + 9], 2.0);

  std::unique_ptr<phi::DenseTensor> tensor1{new phi::DenseTensor()};
  tensor1->mutable_data<float>(phi::make_ddim({height, row_numel}), gpu_place);
  functor(ctx, tensor1.get(), 3.0);

  std::unique_ptr<phi::DenseTensor> tensor2{new phi::DenseTensor()};
  tensor2->mutable_data<float>(phi::make_ddim({height, row_numel}), gpu_place);

  phi::funcs::SelectedRowsAddTensor<phi::GPUContext, float> add_tensor_functor;
  add_tensor_functor(ctx, *output, *tensor1, tensor2.get());

  phi::DenseTensor tensor2_cpu;
  paddle::framework::TensorCopy(*tensor2, cpu_place, ctx, &tensor2_cpu);
  ctx.Wait();

  auto* tensor2_cpu_data = tensor2_cpu.data<float>();
  // row0: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor2_cpu_data[0 * row_numel + 0], 6.0);
  // row1: 3.0
  EXPECT_EQ(tensor2_cpu_data[1 * row_numel + 1], 3.0);
  // row4 : 1.0 + 3.0
  EXPECT_EQ(tensor2_cpu_data[4 * row_numel + 6], 4.0);
  // row5: 2.0 + 3.0
  EXPECT_EQ(tensor2_cpu_data[5 * row_numel + 7], 5.0);
  // row6: 3.0
  EXPECT_EQ(tensor2_cpu_data[6 * row_numel + 1], 3.0);
  // row7: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor2_cpu_data[7 * row_numel + 3], 6.0);
  // row9: 2.0 + 3.0
  EXPECT_EQ(tensor2_cpu_data[9 * row_numel + 6], 5.0);
}

TEST(selected_rows_functor, gpu_add_to) {
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CPUPlace cpu_place;
  phi::GPUContext& ctx = *reinterpret_cast<phi::GPUContext*>(
      paddle::platform::DeviceContextPool::Instance().Get(gpu_place));
  phi::funcs::SetConstant<phi::GPUContext, float> functor;
  int64_t height = 10;
  int64_t row_numel = 10;

  std::vector<int64_t> rows1{0, 4, 7};
  std::unique_ptr<phi::SelectedRows> selected_rows1{
      new phi::SelectedRows(rows1, height)};
  auto* in1_value = selected_rows1->mutable_value();
  in1_value->mutable_data<float>(
      phi::make_ddim({static_cast<int64_t>(rows1.size()), row_numel}),
      gpu_place);
  functor(ctx, in1_value, 1.0);

  std::vector<int64_t> rows2{0, 5, 7, 9};
  std::unique_ptr<phi::SelectedRows> selected_rows2{
      new phi::SelectedRows(rows2, height)};
  auto* in2_value = selected_rows2->mutable_value();
  in2_value->mutable_data<float>(
      phi::make_ddim({static_cast<int64_t>(rows2.size()), row_numel}),
      gpu_place);
  functor(ctx, in2_value, 2.0);

  std::unique_ptr<phi::SelectedRows> output{new phi::SelectedRows()};
  output->set_height(height);
  auto* out_value = output->mutable_value();

  // simply concat two SelectedRows
  out_value->mutable_data<float>(phi::make_ddim({7, 10}), gpu_place);

  phi::funcs::SelectedRowsAddTo<phi::GPUContext, float> add_to_functor;
  add_to_functor(ctx, *selected_rows1, 0, output.get());
  add_to_functor(ctx, *selected_rows2, in1_value->numel(), output.get());

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

  phi::DenseTensor out_cpu;
  paddle::framework::TensorCopy(*out_value, cpu_place, ctx, &out_cpu);
  ctx.Wait();

  auto* out_cpu_data = out_cpu.data<float>();
  // input1 value
  EXPECT_EQ(out_cpu_data[0 * row_numel + 0], 1.0);
  EXPECT_EQ(out_cpu_data[0 * row_numel + 8], 1.0);
  EXPECT_EQ(out_cpu_data[1 * row_numel + 1], 1.0);
  EXPECT_EQ(out_cpu_data[2 * row_numel + 6], 1.0);
  // input2 value
  EXPECT_EQ(out_cpu_data[3 * row_numel + 3], 2.0);
  EXPECT_EQ(out_cpu_data[3 * row_numel + 8], 2.0);
  EXPECT_EQ(out_cpu_data[4 * row_numel + 4], 2.0);
  EXPECT_EQ(out_cpu_data[5 * row_numel + 7], 2.0);
  EXPECT_EQ(out_cpu_data[6 * row_numel + 9], 2.0);

  std::unique_ptr<phi::DenseTensor> tensor1{new phi::DenseTensor()};
  tensor1->mutable_data<float>(phi::make_ddim({height, row_numel}), gpu_place);
  functor(ctx, tensor1.get(), 3.0);

  phi::funcs::SelectedRowsAddToTensor<phi::GPUContext, float>
      add_to_tensor_functor;
  add_to_tensor_functor(ctx, *output, tensor1.get());

  phi::DenseTensor tensor1_cpu;
  paddle::framework::TensorCopy(*tensor1, cpu_place, ctx, &tensor1_cpu);
  ctx.Wait();

  auto* tensor1_cpu_data = tensor1_cpu.data<float>();
  // row0: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor1_cpu_data[0 * row_numel + 0], 6.0);
  // row1: 3.0
  EXPECT_EQ(tensor1_cpu_data[1 * row_numel + 1], 3.0);
  // row4 : 1.0 + 3.0
  EXPECT_EQ(tensor1_cpu_data[4 * row_numel + 6], 4.0);
  // row5: 2.0 + 3.0
  EXPECT_EQ(tensor1_cpu_data[5 * row_numel + 7], 5.0);
  // row6: 3.0
  EXPECT_EQ(tensor1_cpu_data[6 * row_numel + 1], 3.0);
  // row7: 1.0 + 2.0 + 3.0
  EXPECT_EQ(tensor1_cpu_data[7 * row_numel + 3], 6.0);
  // row9: 2.0 + 3.0
  EXPECT_EQ(tensor1_cpu_data[9 * row_numel + 6], 5.0);
}

TEST(selected_rows_functor, gpu_merge_add) {
  paddle::platform::CUDAPlace gpu_place(0);
  paddle::platform::CPUPlace cpu_place;
  phi::GPUContext& ctx = *reinterpret_cast<phi::GPUContext*>(
      paddle::platform::DeviceContextPool::Instance().Get(gpu_place));
  phi::funcs::SetConstant<phi::GPUContext, float> set_const;

  int64_t height = 10;
  int64_t row_numel = 8;

  std::vector<int64_t> rows1{5, 2, 5, 3, 5};
  std::unique_ptr<phi::SelectedRows> selected_rows1{
      new phi::SelectedRows(rows1, height)};
  auto* in1_value = selected_rows1->mutable_value();
  in1_value->mutable_data<float>(
      phi::make_ddim({static_cast<int64_t>(rows1.size()), row_numel}),
      gpu_place);
  set_const(ctx, in1_value, 1.0);

  std::vector<int64_t> rows2{2, 5, 3, 5, 3};
  std::unique_ptr<phi::SelectedRows> selected_rows2{
      new phi::SelectedRows(rows2, height)};
  auto* in2_value = selected_rows2->mutable_value();
  in2_value->mutable_data<float>(
      phi::make_ddim({static_cast<int64_t>(rows2.size()), row_numel}),
      gpu_place);
  set_const(ctx, in2_value, 1.0);

  std::unique_ptr<phi::SelectedRows> output{new phi::SelectedRows()};
  output->set_height(height);
  phi::funcs::scatter::MergeAdd<phi::GPUContext, float> merge_add_functor;

  std::vector<const phi::SelectedRows*> inputs;
  inputs.push_back(selected_rows1.get());
  inputs.push_back(selected_rows2.get());
  merge_add_functor(ctx, inputs, output.get());

  phi::DenseTensor output_cpu;
  paddle::framework::TensorCopy(output->value(), cpu_place, ctx, &output_cpu);
  ctx.Wait();

  EXPECT_EQ(output->height(), height);
  EXPECT_EQ(output->value().dims(), phi::make_ddim({3, row_numel}));

  std::vector<int64_t> ret_rows{2, 3, 5};
  EXPECT_EQ(output->rows(), ret_rows);

  auto* out_data = output_cpu.data<float>();
  for (size_t i = 0; i < ret_rows.size(); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(row_numel); ++j) {
      EXPECT_EQ(out_data[i * row_numel + j], ret_rows[i]);
    }
  }
}

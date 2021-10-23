/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/pten_utils.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable.h"

TEST(PtenUtils, FluidTensorToPtenTensor) {
  // 1. create tensor
  paddle::framework::LoDTensor x;
  paddle::framework::Tensor x2;
  x.Resize({2});
  x.mutable_data<float>(paddle::platform::CPUPlace());
  x.data<float>()[0] = 0.2;
  x.data<float>()[1] = 0.5;

  // 2. test API
  auto dense_x = paddle::framework::MakeTensorImpl<pten::DenseTensor>(
      x, x.place(), x.type());

  // 3. check result
  std::vector<float> expect_value = {0.2, 0.5};
  ASSERT_EQ(dense_x->data<float>()[0], expect_value[0]);
  ASSERT_EQ(dense_x->data<float>()[1], expect_value[1]);
  ASSERT_EQ(dense_x->backend(), pten::Backend::CPU);
  ASSERT_EQ(dense_x->data_type(), pten::DataType::FLOAT32);
}

TEST(PtenUtils, VarToPtenTensor) {
  // 1. create Variable
  paddle::framework::Variable v;
  auto selected_rows = v.GetMutable<paddle::framework::SelectedRows>();
  paddle::framework::Tensor* value = selected_rows->mutable_value();
  auto* data = value->mutable_data<int>(paddle::framework::make_ddim({1, 1}),
                                        paddle::platform::CPUPlace());
  data[0] = 123;
  pten::Backend expect_backend = pten::Backend::CPU;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  expect_backend = pten::Backend::CUDA;
#endif
  auto tensor_def = pten::TensorArgDef(expect_backend, pten::DataLayout::NCHW,
                                       pten::DataType::INT32);
  // 2. test API
  auto tensor_x = paddle::framework::InputVariableToPtenTensor(v, tensor_def);
  // 3. check result
  ASSERT_EQ(tensor_x->backend(), expect_backend);
  ASSERT_EQ(tensor_x->data_type(), pten::DataType::INT32);
}

TEST(PtenUtils, PtenTensorToFluidTensor) {
  pten::DenseTensor dense_tensor(
      pten::TensorMeta(paddle::framework::make_ddim({1, 1}), pten::Backend::CPU,
                       pten::DataType::FLOAT32, pten::DataLayout::ANY),
      pten::TensorStatus());
  auto* data_ptr = dense_tensor.mutable_data<float>();
  data_ptr[0] = 0.5;
  // share allocation into fluid Tensor
  paddle::framework::Tensor tensor;
  paddle::framework::LoDTensor lod_tensor;
  paddle::framework::ShareTensorImpl(&dense_tensor, &tensor);
  paddle::framework::ShareTensorImpl(&dense_tensor, &lod_tensor);
  // compare
  ASSERT_EQ(tensor.data<float>()[0], 0.5);
  ASSERT_EQ(lod_tensor.data<float>()[0], 0.5);
}

TEST(PtenUtils, TransPtenKernelKeyToOpKernelType) {
  pten::KernelKey kernel_key(pten::Backend::CPU, pten::DataLayout::NCHW,
                             pten::DataType::FLOAT32);
  auto op_kernel_type =
      paddle::framework::TransPtenKernelKeyToOpKernelType(kernel_key);
  ASSERT_EQ(op_kernel_type.data_type_, paddle::framework::proto::VarType::FP32);
  ASSERT_EQ(op_kernel_type.data_layout_, paddle::framework::DataLayout::kNCHW);
  ASSERT_TRUE(paddle::platform::is_cpu_place(op_kernel_type.place_));
  ASSERT_EQ(op_kernel_type.library_type_,
            paddle::framework::LibraryType::kPlain);

#ifdef PADDLE_WITH_MKLDNN
  pten::KernelKey kernel_key_mkldnn(
      pten::Backend::MKLDNN, pten::DataLayout::NCHW, pten::DataType::FLOAT32);
  op_kernel_type =
      paddle::framework::TransPtenKernelKeyToOpKernelType(kernel_key_mkldnn);
  ASSERT_EQ(op_kernel_type.data_type_, paddle::framework::proto::VarType::FP32);
  ASSERT_EQ(op_kernel_type.data_layout_, paddle::framework::DataLayout::kNCHW);
  ASSERT_TRUE(paddle::platform::is_cpu_place(op_kernel_type.place_));
  ASSERT_EQ(op_kernel_type.library_type_,
            paddle::framework::LibraryType::kMKLDNN);
#endif

#ifdef PADDLE_WITH_CUDA
  pten::KernelKey kernel_key_cudnn(pten::Backend::CUDNN, pten::DataLayout::NCHW,
                                   pten::DataType::FLOAT32);
  op_kernel_type =
      paddle::framework::TransPtenKernelKeyToOpKernelType(kernel_key_cudnn);
  ASSERT_EQ(op_kernel_type.data_type_, paddle::framework::proto::VarType::FP32);
  ASSERT_EQ(op_kernel_type.data_layout_, paddle::framework::DataLayout::kNCHW);
  ASSERT_TRUE(paddle::platform::is_gpu_place(op_kernel_type.place_));
  ASSERT_EQ(op_kernel_type.library_type_,
            paddle::framework::LibraryType::kCUDNN);
#endif
}

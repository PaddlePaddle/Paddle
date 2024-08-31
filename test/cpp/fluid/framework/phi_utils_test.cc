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

#include "paddle/fluid/framework/phi_utils.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/variable.h"

TEST(PhiUtils, TransPhiKernelKeyToOpKernelType) {
  phi::KernelKey kernel_key(
      phi::Backend::CPU, phi::DataLayout::NCHW, phi::DataType::FLOAT32);
  auto op_kernel_type =
      paddle::framework::TransPhiKernelKeyToOpKernelType(kernel_key);
  ASSERT_EQ(op_kernel_type.data_type_, paddle::framework::proto::VarType::FP32);
  ASSERT_EQ(op_kernel_type.data_layout_, phi::DataLayout::kNCHW);
  ASSERT_TRUE(phi::is_cpu_place(op_kernel_type.place_));
  ASSERT_EQ(op_kernel_type.library_type_,
            paddle::framework::LibraryType::kPlain);

#ifdef PADDLE_WITH_DNNL
  phi::KernelKey kernel_key_onednn(
      phi::Backend::ONEDNN, phi::DataLayout::NCHW, phi::DataType::FLOAT32);
  op_kernel_type =
      paddle::framework::TransPhiKernelKeyToOpKernelType(kernel_key_onednn);
  ASSERT_EQ(op_kernel_type.data_type_, paddle::framework::proto::VarType::FP32);
  ASSERT_EQ(op_kernel_type.data_layout_, phi::DataLayout::kNCHW);
  ASSERT_TRUE(phi::is_cpu_place(op_kernel_type.place_));
  ASSERT_EQ(op_kernel_type.library_type_,
            paddle::framework::LibraryType::kMKLDNN);
#endif

#ifdef PADDLE_WITH_CUDA
  phi::KernelKey kernel_key_cudnn(
      phi::Backend::GPUDNN, phi::DataLayout::NCHW, phi::DataType::FLOAT32);
  op_kernel_type =
      paddle::framework::TransPhiKernelKeyToOpKernelType(kernel_key_cudnn);
  ASSERT_EQ(op_kernel_type.data_type_, paddle::framework::proto::VarType::FP32);
  ASSERT_EQ(op_kernel_type.data_layout_, phi::DataLayout::kNCHW);
  ASSERT_TRUE(phi::is_gpu_place(op_kernel_type.place_));
  ASSERT_EQ(op_kernel_type.library_type_,
            paddle::framework::LibraryType::kCUDNN);
#endif
}

TEST(PhiUtils, TransOpKernelTypeToPhiKernelKey) {
  paddle::framework::OpKernelType op_kernel_type(
      paddle::framework::proto::VarType::FP32,
      phi::CPUPlace(),
      phi::DataLayout::kNCHW);
  auto kernel_key =
      paddle::framework::TransOpKernelTypeToPhiKernelKey(op_kernel_type);
  ASSERT_EQ(kernel_key.dtype(), phi::DataType::FLOAT32);
  ASSERT_EQ(kernel_key.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(kernel_key.backend(), phi::Backend::CPU);

#ifdef PADDLE_WITH_DNNL
  paddle::framework::OpKernelType op_kernel_type_onednn(
      paddle::framework::proto::VarType::FP32,
      phi::CPUPlace(),
      phi::DataLayout::ONEDNN,
      paddle::framework::LibraryType::kMKLDNN);
  auto kernel_key_onednn =
      paddle::framework::TransOpKernelTypeToPhiKernelKey(op_kernel_type_onednn);
  ASSERT_EQ(kernel_key_onednn.dtype(), phi::DataType::FLOAT32);
  ASSERT_EQ(kernel_key_onednn.layout(), phi::DataLayout::ONEDNN);
  ASSERT_EQ(kernel_key_onednn.backend(), phi::Backend::ONEDNN);
#endif

#ifdef PADDLE_WITH_CUDA
  paddle::framework::OpKernelType op_kernel_type_cudnn(
      paddle::framework::proto::VarType::FP32,
      phi::CPUPlace(),
      phi::DataLayout::kNCHW,
      paddle::framework::LibraryType::kCUDNN);
  auto kernel_key_cudnn =
      paddle::framework::TransOpKernelTypeToPhiKernelKey(op_kernel_type_cudnn);
  ASSERT_EQ(kernel_key_cudnn.dtype(), phi::DataType::FLOAT32);
  ASSERT_EQ(kernel_key_cudnn.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(kernel_key_cudnn.backend(), phi::Backend::GPUDNN);
#endif
}

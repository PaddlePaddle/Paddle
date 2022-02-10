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
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/variable.h"

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

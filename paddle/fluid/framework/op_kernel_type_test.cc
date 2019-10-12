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

#include "paddle/fluid/framework/op_kernel_type.h"
#include <gtest/gtest.h>
#include <iostream>

TEST(OpKernelType, ToString) {
  using OpKernelType = paddle::framework::OpKernelType;
  using DataType = paddle::framework::proto::VarType;
  using CPUPlace = paddle::platform::CPUPlace;
  using DataLayout = paddle::framework::DataLayout;
  using LibraryType = paddle::framework::LibraryType;

  OpKernelType op_kernel_type(DataType::FP32, CPUPlace(), DataLayout::kNCHW,
                              LibraryType::kCUDNN);

  ASSERT_EQ(paddle::framework::KernelTypeToString(op_kernel_type),
            "data_type[float]:data_layout[NCHW]:place[CPUPlace]:library_type["
            "CUDNN]");

  using CUDAPlace = paddle::platform::CUDAPlace;
  OpKernelType op_kernel_type2(DataType::FP16, CUDAPlace(0), DataLayout::kNCHW,
                               LibraryType::kCUDNN);
  ASSERT_EQ(paddle::framework::KernelTypeToString(op_kernel_type2),
            "data_type[::paddle::platform::float16]:data_layout[NCHW]:place["
            "CUDAPlace(0)]:library_"
            "type[CUDNN]");
}

TEST(OpKernelType, Hash) {
  using OpKernelType = paddle::framework::OpKernelType;
  using DataType = paddle::framework::proto::VarType;
  using CPUPlace = paddle::platform::CPUPlace;
  using CUDAPlace = paddle::platform::CUDAPlace;
  using DataLayout = paddle::framework::DataLayout;
  using LibraryType = paddle::framework::LibraryType;

  OpKernelType op_kernel_type_1(DataType::FP32, CPUPlace(), DataLayout::kNCHW,
                                LibraryType::kCUDNN);
  OpKernelType op_kernel_type_2(DataType::FP32, CUDAPlace(0), DataLayout::kNCHW,
                                LibraryType::kCUDNN);

  OpKernelType::Hash hasher;
  ASSERT_NE(hasher(op_kernel_type_1), hasher(op_kernel_type_2));
}

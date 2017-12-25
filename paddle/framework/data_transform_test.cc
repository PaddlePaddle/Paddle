/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/data_transform.h"
#include <gtest/gtest.h>

using OpKernelType = paddle::framework::OpKernelType;
using DataType = paddle::framework::proto::DataType;
using CPUPlace = paddle::platform::CPUPlace;
using GPUPlace = paddle::platform::GPUPlace;
using DataLayout = paddle::framework::DataLayout;
using LibraryType = paddle::framework::LibraryType;
using DataTransformFnMap = paddle::framework::DataTransformFnMap;
using DataTransformationFN = paddle::framework::DataTransformationFN;

namespace frw = paddle::framework;

namespace paddle {
namespace framework {
void fn1(const frw::Tensor& in, frw::Tensor* out) {}
}  // namespace framework
}  // namespace paddle

TEST(DataTransform, Register) {
  OpKernelType kernel_type_1(DataType::FP32, CPUPlace(), DataLayout::kNCHW,
                             LibraryType::kCUDNN);
  OpKernelType kernel_type_2(DataType::FP32, GPUPlace(0), DataLayout::kNCHW,
                             LibraryType::kCUDNN);

  DataTransformationFN fn = frw::fn1;
  DataTransformFnMap::Instance().Insert(kernel_type_1, kernel_type_2, fn);

  ASSERT_EQ(DataTransformFnMap::Instance().Map().size(), 1UL);
}
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
using CUDAPlace = paddle::platform::CUDAPlace;
using DataLayout = paddle::framework::DataLayout;
using LibraryType = paddle::framework::LibraryType;
using DataTransformFnMap = paddle::framework::DataTransformFnMap;
using DataTransformationFN = paddle::framework::DataTransformationFN;

namespace frw = paddle::framework;

namespace paddle {
namespace framework {
OpKernelType kernel_type_1(DataType::FP32, CPUPlace(), DataLayout::kNCHW,
                           LibraryType::kCUDNN);
OpKernelType kernel_type_2(DataType::FP32, CUDAPlace(0), DataLayout::kNCHW,
                           LibraryType::kCUDNN);
OpKernelType kernel_type_3(DataType::FP16, CUDAPlace(0), DataLayout::kNCHW,
                           LibraryType::kCUDNN);
void type1_to_type2(const frw::Tensor& in, frw::Tensor* out) {}

}  // namespace framework
}  // namespace paddle

REGISTER_DATA_TRANSFORM_FN(test, frw::kernel_type_1, frw::kernel_type_2,
                           frw::type1_to_type2);
REGISTER_DATA_TRANSFORM_FN(test1, frw::kernel_type_2, frw::kernel_type_3,
                           frw::type1_to_type2);

TEST(DataTransform, Register) {
  auto& instance = DataTransformFnMap::Instance();

  ASSERT_EQ(instance.Map().size(), 2UL);

  DataTransformationFN fn = frw::type1_to_type2;

  instance.Insert(frw::kernel_type_1, frw::kernel_type_3, fn);

  ASSERT_EQ(instance.Map().size(), 3UL);
}

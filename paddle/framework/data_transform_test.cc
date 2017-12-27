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

namespace paddle {
namespace framework {

using namespace platform;

int test_value = 0;

OpKernelType kernel_type_1(proto::DataType::FP32, CPUPlace(), DataLayout::kNCHW,
                           LibraryType::kCUDNN);
OpKernelType kernel_type_2(proto::DataType::FP32, CUDAPlace(0),
                           DataLayout::kNCHW, LibraryType::kCUDNN);
OpKernelType kernel_type_3(proto::DataType::FP16, CUDAPlace(0),
                           DataLayout::kNCHW, LibraryType::kCUDNN);

void type1_to_type2(const std::vector<const platform::DeviceContext*>& ctx,
                    const Variable& in, Variable* out) {
  test_value++;
}

void type2_to_type3(const std::vector<const platform::DeviceContext*>& ctx,
                    const Variable& in, Variable* out) {
  test_value--;
}

void type1_to_type3(const std::vector<const platform::DeviceContext*>& ctx,
                    const Variable& in, Variable* out) {
  test_value += 2;
}

}  // namespace framework
}  // namespace paddle

namespace frw = paddle::framework;

REGISTER_DATA_TRANSFORM_FN(frw::kernel_type_1, frw::kernel_type_2,
                           frw::type1_to_type2);
REGISTER_DATA_TRANSFORM_FN(frw::kernel_type_2, frw::kernel_type_3,
                           frw::type2_to_type3);
REGISTER_DATA_TRANSFORM_FN(frw::kernel_type_1, frw::kernel_type_3,
                           frw::type1_to_type3);

TEST(DataTransform, Register) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  auto& instance = DataTransformFnMap::Instance();
  ASSERT_EQ(instance.Map().size(), 3UL);
  std::vector<const DeviceContext*> ctx;
  paddle::framework::Variable in;
  paddle::framework::Variable out;

  instance.Get(std::make_pair(frw::kernel_type_1, frw::kernel_type_2))(ctx, in,
                                                                       &out);
  ASSERT_EQ(test_value, 1);
  instance.Get(std::make_pair(frw::kernel_type_2, frw::kernel_type_3))(ctx, in,
                                                                       &out);
  ASSERT_EQ(test_value, 0);
  instance.Get(std::make_pair(frw::kernel_type_1, frw::kernel_type_3))(ctx, in,
                                                                       &out);
  ASSERT_EQ(test_value, 2);
}

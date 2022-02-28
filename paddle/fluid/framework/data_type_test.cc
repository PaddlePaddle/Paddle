// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/framework/data_type.h"

#include <string>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/tensor.h"

TEST(DataType, float16) {
  using paddle::framework::Tensor;
  using paddle::platform::CPUPlace;
  using paddle::platform::float16;
  namespace f = paddle::framework;
  f::proto::VarType::Type dtype = f::proto::VarType::FP16;

  Tensor tensor;
  CPUPlace cpu;
  tensor.mutable_data(cpu, f::TransToPtenDataType(dtype));

  // test fp16 tensor
  EXPECT_EQ(f::TransToProtoVarType(tensor.dtype()),
            f::ToDataType(typeid(float16)));

  // test fp16 size
  EXPECT_EQ(f::SizeOfType(dtype), 2u);

  // test debug info
  std::string type = "::paddle::platform::float16";
  EXPECT_STREQ(f::DataTypeToString(dtype).c_str(), type.c_str());
}

TEST(DataType, bfloat16) {
  using paddle::framework::Tensor;
  using paddle::platform::CPUPlace;
  using paddle::platform::bfloat16;
  namespace f = paddle::framework;
  f::proto::VarType::Type dtype = f::proto::VarType::BF16;

  Tensor tensor;
  CPUPlace cpu;
  tensor.mutable_data(cpu, f::TransToPtenDataType(dtype));

  // test bf16 tensor
  EXPECT_EQ(f::TransToProtoVarType(tensor.dtype()),
            f::ToDataType(typeid(bfloat16)));

  // test bf16 size
  EXPECT_EQ(f::SizeOfType(dtype), 2u);

  // test debug info
  std::string type = "::paddle::platform::bfloat16";
  EXPECT_STREQ(f::DataTypeToString(dtype).c_str(), type.c_str());
}

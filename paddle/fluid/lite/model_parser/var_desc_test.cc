// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/model_parser/cpp/var_desc.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/model_parser/compatible_pb.h"
#include "paddle/fluid/lite/model_parser/pb/var_desc.h"

namespace paddle {
namespace lite {

using VarType = lite::VarDescAPI::VarDataType;

template <typename VarDesc>
void TestX() {
  VarDesc desc("a");

  desc.SetName("test");
  auto name = desc.Name();
  ASSERT_EQ(name, "test");

  desc.SetPersistable(false);
  auto persistable = desc.Persistable();
  ASSERT_EQ(persistable, false);

  desc.SetVarType(VarType::LOD_TENSOR);
  auto type = desc.GetVarType();
  ASSERT_EQ(type, VarType::LOD_TENSOR);

  desc.SetVarDataType(VarType::FP32);
  auto data_type = desc.GetVarDataType();
  ASSERT_EQ(data_type, VarType::FP32);

  desc.SetShape(std::vector<int64_t>({1, 2, 3}));
  auto shape = desc.GetShape();
  ASSERT_EQ(shape.size(), 3);
  ASSERT_EQ(shape[0], 1);
  ASSERT_EQ(shape[1], 2);
  ASSERT_EQ(shape[2], 3);
}

TEST(VarDesc, Basic) {
  TestX<pb::VarDesc>();
  TestX<cpp::VarDesc>();
}

TEST(VarDesc, PbToCpp) {
  pb::VarDesc desc("a");

  desc.SetName("test");
  desc.SetPersistable(false);
  desc.SetVarType(VarType::LOD_TENSOR);
  desc.SetVarDataType(VarType::FP32);
  desc.SetShape(std::vector<int64_t>({1, 2, 3}));

  cpp::VarDesc cpp_desc;

  TransformVarDescPbToCpp(desc, &cpp_desc);
  {
    auto& desc = cpp_desc;
    auto name = desc.Name();
    ASSERT_EQ(name, "test");

    auto persistable = desc.Persistable();
    ASSERT_EQ(persistable, false);

    auto type = desc.GetVarType();
    ASSERT_EQ(type, VarType::LOD_TENSOR);

    auto data_type = desc.GetVarDataType();
    ASSERT_EQ(data_type, VarType::FP32);

    auto shape = desc.GetShape();
    ASSERT_EQ(shape.size(), 3);
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 2);
    ASSERT_EQ(shape[2], 3);
  }

  desc.SetVarType(VarType::FP32);
  TransformVarDescPbToCpp(desc, &cpp_desc);
  {
    auto& desc = cpp_desc;
    auto name = desc.Name();
    ASSERT_EQ(name, "test");

    auto persistable = desc.Persistable();
    ASSERT_EQ(persistable, false);

    auto type = desc.GetVarType();
    ASSERT_EQ(type, VarType::FP32);

    auto data_type = desc.GetVarDataType();
    ASSERT_EQ(data_type, VarType::UNK);

    auto shape = desc.GetShape();
    ASSERT_EQ(shape.size(), 1);
    ASSERT_EQ(shape[0], 1);
  }
}

}  // namespace lite
}  // namespace paddle

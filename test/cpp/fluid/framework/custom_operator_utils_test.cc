// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/custom_operator.h"

namespace paddle {
namespace framework {

TEST(ParseAttrTypeToEnumTest, ScalarTypes) {
  EXPECT_EQ(ParseAttrTypeToEnum("bool"), CustomAttrType::BOOL);
  EXPECT_EQ(ParseAttrTypeToEnum("int"), CustomAttrType::INT);
  EXPECT_EQ(ParseAttrTypeToEnum("float"), CustomAttrType::FLOAT);
  EXPECT_EQ(ParseAttrTypeToEnum("double"), CustomAttrType::DOUBLE);
  EXPECT_EQ(ParseAttrTypeToEnum("int64_t"), CustomAttrType::INT64);
  EXPECT_EQ(ParseAttrTypeToEnum("std::string"), CustomAttrType::STRING);
}

TEST(ParseAttrTypeToEnumTest, VectorTypes) {
  EXPECT_EQ(ParseAttrTypeToEnum("std::vector<int>"), CustomAttrType::VEC_INT);
  EXPECT_EQ(ParseAttrTypeToEnum("std::vector<float>"),
            CustomAttrType::VEC_FLOAT);
  EXPECT_EQ(ParseAttrTypeToEnum("std::vector<int64_t>"),
            CustomAttrType::VEC_INT64);
  EXPECT_EQ(ParseAttrTypeToEnum("std::vector<std::string>"),
            CustomAttrType::VEC_STRING);
}

}  // namespace framework
}  // namespace paddle

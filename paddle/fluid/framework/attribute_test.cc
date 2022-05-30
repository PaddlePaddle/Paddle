//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/program_desc.h"

#include "gtest/gtest.h"
#include "paddle/utils/any.h"

TEST(Attribute, GetAttrValueToAny) {
  paddle::framework::Attribute x_int(100);
  auto rlt_int = paddle::framework::GetAttrValue(x_int);
  EXPECT_EQ(paddle::any_cast<int>(rlt_int), 100);

  float float_value = 3.14;
  paddle::framework::Attribute x_float(float_value);
  auto rlt_float = paddle::framework::GetAttrValue(x_float);
  EXPECT_NEAR(paddle::any_cast<float>(rlt_float), 3.14, 1e-6);

  std::string str_value("test");
  paddle::framework::Attribute x_str(str_value);
  auto rlt_str = paddle::framework::GetAttrValue(x_str);
  EXPECT_EQ(paddle::any_cast<std::string>(rlt_str), "test");

  std::vector<int> vec_int_var(2, 100);
  paddle::framework::Attribute x_vec_int = vec_int_var;
  auto rlt_vec_int = paddle::framework::GetAttrValue(x_vec_int);
  auto vec_int = paddle::any_cast<std::vector<int>>(rlt_vec_int);
  EXPECT_EQ(vec_int.size(), 2UL);
  EXPECT_EQ(vec_int[0], 100);
  EXPECT_EQ(vec_int[1], 100);

  std::vector<float> vec_float_var(2, 3.14);
  paddle::framework::Attribute x_vec_float = vec_float_var;
  auto rlt_vec_float = paddle::framework::GetAttrValue(x_vec_float);
  auto vec_float = paddle::any_cast<std::vector<float>>(rlt_vec_float);
  EXPECT_EQ(vec_float.size(), 2UL);
  EXPECT_NEAR(vec_float[0], 3.14, 1e-6);
  EXPECT_NEAR(vec_float[1], 3.14, 1e-6);

  std::vector<std::string> vec_str_var(2, "test");
  paddle::framework::Attribute x_vec_str = vec_str_var;
  auto rlt_vec_str = paddle::framework::GetAttrValue(x_vec_str);
  auto vec_str = paddle::any_cast<std::vector<std::string>>(rlt_vec_str);
  EXPECT_EQ(vec_str.size(), 2UL);
  EXPECT_EQ(vec_str[0], "test");
  EXPECT_EQ(vec_str[1], "test");

  paddle::framework::Attribute x_bool(true);
  auto rlt_bool = paddle::framework::GetAttrValue(x_bool);
  EXPECT_EQ(paddle::any_cast<bool>(rlt_bool), true);

  std::vector<bool> vec_bool_var(2, true);
  paddle::framework::Attribute x_vec_bool = vec_bool_var;
  auto rlt_vec_bool = paddle::framework::GetAttrValue(x_vec_bool);
  auto vec_bool = paddle::any_cast<std::vector<bool>>(rlt_vec_bool);
  EXPECT_EQ(vec_bool.size(), 2UL);
  EXPECT_EQ(vec_bool[0], true);
  EXPECT_EQ(vec_bool[1], true);

  paddle::framework::ProgramDesc prog;
  paddle::framework::proto::BlockDesc proto_block;
  paddle::framework::BlockDesc block_desc(&prog, &proto_block);
  paddle::framework::Attribute x_block_desc(&block_desc);
  auto rlt_block_desc = paddle::framework::GetAttrValue(x_block_desc);
  auto block_desc_ptr =
      paddle::any_cast<paddle::framework::BlockDesc*>(rlt_block_desc);
  EXPECT_NE(block_desc_ptr, nullptr);

  std::vector<paddle::framework::BlockDesc*> vec_block_desc_var;
  vec_block_desc_var.emplace_back(&block_desc);
  paddle::framework::Attribute x_vec_block_desc(vec_block_desc_var);
  auto rlt_vec_block_desc = paddle::framework::GetAttrValue(x_vec_block_desc);
  auto vec_block_desc =
      paddle::any_cast<std::vector<paddle::framework::BlockDesc*>>(
          rlt_vec_block_desc);
  EXPECT_EQ(vec_block_desc.size(), 1UL);
  EXPECT_NE(vec_block_desc[0], nullptr);

  int64_t int64_value = 100;
  paddle::framework::Attribute x_int64(int64_value);
  auto rlt_int64 = paddle::framework::GetAttrValue(x_int64);
  EXPECT_EQ(paddle::any_cast<int64_t>(rlt_int64), 100);

  std::vector<int64_t> vec_int64_var(2, 100);
  paddle::framework::Attribute x_vec_int64 = vec_int64_var;
  auto rlt_vec_int64 = paddle::framework::GetAttrValue(x_vec_int64);
  auto vec_int64 = paddle::any_cast<std::vector<int64_t>>(rlt_vec_int64);
  EXPECT_EQ(vec_int64.size(), 2UL);
  EXPECT_EQ(vec_int64[0], 100);
  EXPECT_EQ(vec_int64[1], 100);

  std::vector<double> vec_double_var(2, 3.14);
  paddle::framework::Attribute x_vec_double = vec_double_var;
  auto rlt_vec_double = paddle::framework::GetAttrValue(x_vec_double);
  auto vec_double = paddle::any_cast<std::vector<double>>(rlt_vec_double);
  EXPECT_EQ(vec_double.size(), 2UL);
  EXPECT_NEAR(vec_double[0], 3.14, 1e-6);
  EXPECT_NEAR(vec_double[1], 3.14, 1e-6);
}

/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/tensorrt/plugin_arg_mapping_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(ArgMappingContexTest, BasicFunction) {
  paddle::framework::proto::OpDesc op;
  op.set_type("imaged_op");
  auto *input_var = op.add_inputs();
  input_var->set_parameter("X");
  *input_var->add_arguments() = "input";

  auto *output_var = op.add_outputs();
  output_var->set_parameter("Out");
  *output_var->add_arguments() = "output";

  auto *attr = op.add_attrs();
  attr->set_name("int_attr");
  attr->set_type(paddle::framework::proto::AttrType::INT);
  attr->set_i(1);

  attr = op.add_attrs();
  attr->set_name("float_attr");
  attr->set_type(paddle::framework::proto::AttrType::FLOAT);
  attr->set_f(1.0);

  attr = op.add_attrs();
  attr->set_name("string_attr");
  attr->set_type(paddle::framework::proto::AttrType::STRING);
  attr->set_s("1");

  attr = op.add_attrs();
  attr->set_name("bool_attr");
  attr->set_type(paddle::framework::proto::AttrType::BOOLEAN);
  attr->set_b(true);

  attr = op.add_attrs();
  attr->set_name("ints_attr");
  attr->set_type(paddle::framework::proto::AttrType::INTS);
  attr->add_ints(1);
  attr->add_ints(2);

  attr = op.add_attrs();
  attr->set_name("floats_attr");
  attr->set_type(paddle::framework::proto::AttrType::FLOATS);
  attr->add_floats(1.0);
  attr->add_floats(2.0);

  attr = op.add_attrs();
  attr->set_name("strings_attr");
  attr->set_type(paddle::framework::proto::AttrType::STRINGS);
  attr->add_strings("1");
  attr->add_strings("2");

  attr = op.add_attrs();
  attr->set_name("bools_attr");
  attr->set_type(paddle::framework::proto::AttrType::BOOLEANS);
  attr->add_bools(true);
  attr->add_bools(true);

  framework::OpDesc op_desc(op, nullptr);
  PluginArgumentMappingContext context(&op_desc);

  EXPECT_EQ(context.HasInput("X"), true);
  EXPECT_EQ(context.HasOutput("Out"), true);
  EXPECT_EQ(context.HasAttr("int_attr"), true);

  int int_attr = any_cast<int>(context.Attr("int_attr"));
  EXPECT_EQ(int_attr, 1);

  float flaot_attr = any_cast<float>(context.Attr("float_attr"));
  EXPECT_EQ(flaot_attr, 1);

  std::string string_attr = any_cast<std::string>(context.Attr("string_attr"));
  EXPECT_EQ(string_attr, "1");

  bool bool_attr = any_cast<bool>(context.Attr("bool_attr"));
  EXPECT_EQ(bool_attr, true);

  std::vector<int> ints_attr =
      any_cast<std::vector<int>>(context.Attr("ints_attr"));
  EXPECT_EQ(ints_attr[0], 1);
  EXPECT_EQ(ints_attr[1], 2);

  std::vector<float> floats_attr =
      any_cast<std::vector<float>>(context.Attr("floats_attr"));
  EXPECT_EQ(floats_attr[0], 1.0);
  EXPECT_EQ(floats_attr[1], 2.0);

  std::vector<std::string> strings_attr =
      any_cast<std::vector<std::string>>(context.Attr("strings_attr"));
  EXPECT_EQ(strings_attr[0], "1");
  EXPECT_EQ(strings_attr[1], "2");

  std::vector<bool> bools_attr =
      any_cast<std::vector<bool>>(context.Attr("bools_attr"));
  EXPECT_EQ(bools_attr[0], true);
  EXPECT_EQ(bools_attr[1], true);

  EXPECT_EQ(context.InputSize("X"), true);
  EXPECT_EQ(context.OutputSize("Out"), true);
  EXPECT_EQ(context.IsDenseTensorInput("X"), false);
  EXPECT_EQ(context.IsDenseTensorInputs("X"), false);
  EXPECT_EQ(context.IsSelectedRowsInput("X"), false);
  EXPECT_EQ(context.IsDenseTensorVectorInput("X"), false);

  EXPECT_EQ(context.IsDenseTensorOutput("Out"), false);
  EXPECT_EQ(context.IsSelectedRowsOutput("Out"), false);
  EXPECT_EQ(context.IsForInferShape(), false);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

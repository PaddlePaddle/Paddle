/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/anakin/convert/elementwise.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

static void test_elementwise_op(const std::string &op_type) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("x", {1, 1, 2, 2});
  validator.DeclInputVar("y", {1, 1, 2, 2});
  validator.DeclOutputVar("out", {1, 1, 2, 2});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType(op_type);
  desc.SetInput("X", {"x"});
  desc.SetInput("Y", {"y"});
  desc.SetOutput("Out", {"out"});

  int axis = -1;
  desc.SetAttr("axis", axis);

  validator.SetOp(*desc.Proto());
  validator.Execute(1);
}

TEST(elementwise_op, native_add) { test_elementwise_op("elementwise_add"); }
TEST(elementwise_op, native_mul) { test_elementwise_op("elementwise_mul"); }

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(elementwise_add);
USE_ANAKIN_CONVERTER(elementwise_add);
USE_OP(elementwise_mul);
USE_ANAKIN_CONVERTER(elementwise_mul);

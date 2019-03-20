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
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(elementwise_op, native) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, scope);
  validator.DeclInputVar("elementwise_add_x", {1, 1, 2, 2});
  validator.DeclInputVar("elementwise_y", {1, 1, 2, 2});
  validator.DeclOutputVar("elementwise_out", {1, 1, 2, 2});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("elementwise_add");
  desc.SetInput("X", {"elementwise_add_x"});
  desc.SetInput("Y", {"elementwise_y"});
  desc.SetOutput("Out", {"elementwise_out"});

  int axis = -1;
  desc.SetAttr("axis", axis);

  validator.SetOp(*desc.Proto());
  validator.Execute(1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(elementwise_add);
USE_ANAKIN_CONVERTER(elementwise_add);

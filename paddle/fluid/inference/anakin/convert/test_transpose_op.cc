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

TEST(transpose_op, test) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("transpose-X", {2, 3, 4, 5});
  validator.DeclOutputVar("transpose-Out", {4, 2, 5, 3});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("transpose");
  desc.SetInput("X", {"transpose-X"});
  desc.SetOutput("Out", {"transpose-Out"});
  desc.SetAttr("axis", std::vector<int>({2, 0, 3, 1}));

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(3);
}

// test input shape's dims < 4
TEST(transpose_op, test2) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("transpose-X", {3, 4, 5});
  validator.DeclOutputVar("transpose-Out", {3, 5, 4});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("transpose");
  desc.SetInput("X", {"transpose-X"});
  desc.SetOutput("Out", {"transpose-Out"});
  desc.SetAttr("axis", std::vector<int>({0, 2, 1}));

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(transpose);
USE_ANAKIN_CONVERTER(transpose);

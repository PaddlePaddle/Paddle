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
#include "paddle/fluid/inference/anakin/convert/concat.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(concat_op, test) {
  std::unordered_set<std::string> parameters({""});
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("concat_x1", {1, 2, 1, 1});
  validator.DeclInputVar("concat_x2", {1, 3, 1, 1});
  validator.DeclInputVar("concat_x3", {1, 1, 1, 1});
  validator.DeclOutputVar("concat_out", {1, 6, 1, 1});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("concat");
  desc.SetInput("X", {"concat_x1", "concat_x2", "concat_x3"});
  desc.SetOutput("Out", {"concat_out"});

  int axis = 1;
  desc.SetAttr("axis", axis);

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

TEST(concat_op, test2) {
  std::unordered_set<std::string> parameters({""});
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("concat_x1", {1, 4});
  validator.DeclInputVar("concat_x2", {3, 4});
  validator.DeclInputVar("concat_x3", {2, 4});
  validator.DeclOutputVar("concat_out", {6, 4});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("concat");
  desc.SetInput("X", {"concat_x1", "concat_x2", "concat_x3"});
  desc.SetOutput("Out", {"concat_out"});

  int axis = 0;
  desc.SetAttr("axis", axis);

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
USE_OP(concat);
USE_ANAKIN_CONVERTER(concat);

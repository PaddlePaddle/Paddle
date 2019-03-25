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
#include "paddle/fluid/inference/anakin/convert/fc.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(fc_op, test) {
  auto fc_converter = OpRegister::instance()->Get("fc");
  ASSERT_TRUE(fc_converter != nullptr);
  // Registrar<FcOpConverter> register_fc("fc");
  // auto fc = std::make_shared<FcOpConverter>();

  std::unordered_set<std::string> parameters({"mul_y"});
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, scope);
  validator.DeclInputVar("mul_x", {1, 1, 1, 1});
  validator.DeclParamVar("mul_y", {1, 2});
  validator.DeclOutputVar("mul_out", {1, 1, 1, 2});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("mul");
  desc.SetInput("X", {"mul_x"});
  desc.SetInput("Y", {"mul_y"});
  desc.SetOutput("Out", {"mul_out"});
  int num_flatten_dims = 3;
  desc.SetAttr("x_num_col_dims", num_flatten_dims);
  validator.SetOp(*desc.Proto());

  validator.Execute(10);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(mul);

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
#include "paddle/fluid/inference/anakin/convert/conv2d.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(conv2d_op, test) {
  auto* conv2d_converter =
      Registry<AnakinOpConverter>::Global().Lookup("conv2d");
  ASSERT_TRUE(conv2d_converter != nullptr);
  std::unordered_set<std::string> parameters({"conv2d-Y"});
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("conv2d-X", {1, 3, 3, 3});
  validator.DeclParamVar("conv2d-Y", {4, 3, 1, 1});
  validator.DeclOutputVar("conv2d-Out", {1, 4, 3, 3});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("conv2d");
  desc.SetInput("Input", {"conv2d-X"});
  desc.SetInput("Filter", {"conv2d-Y"});
  desc.SetOutput("Output", {"conv2d-Out"});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({0, 0});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("dilations", dilations);
  desc.SetAttr("groups", groups);

  validator.SetOp(*desc.Proto());

  validator.Execute(3);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(conv2d);
USE_ANAKIN_CONVERTER(conv2d);

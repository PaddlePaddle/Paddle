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
#include "paddle/fluid/inference/anakin/convert/im2sequence.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(im2sequence_op, native) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);

  std::vector<int> kernels = {6, 1};
  std::vector<int> strides = {1, 1};
  std::vector<int> paddings = {0, 0, 0, 0};

  validator.DeclInputVar("x", {1, 1, 2, 2});
  validator.DeclOutputVar("out", {1, 1 * kernels[0] * kernels[1]});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("im2sequence");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"out"});

  desc.SetAttr("kernels", kernels);
  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);

  validator.SetOp(*desc.Proto());
  validator.Execute(1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(im2sequence);
USE_ANAKIN_CONVERTER(im2sequence);

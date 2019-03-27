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
#include "paddle/fluid/inference/anakin/convert/dropout.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(dropout_op, native) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, &scope);
  validator.DeclInputVar("x", {1, 1, 2, 2});
  validator.DeclOutputVar("out", {1, 1, 2, 2});
  validator.DeclOutputVar("mask", {1, 1, 2, 2});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("dropout");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"out"});
  desc.SetOutput("Mask", {"mask"});

  float dropout_prob = 0.5;
  desc.SetAttr("dropout_prob", dropout_prob);
  desc.SetAttr("is_test", true);

  validator.SetOp(*desc.Proto());
  std::unordered_set<std::string> neglected_output = {"mask"};
  validator.Execute(1, neglected_output);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(dropout);
USE_ANAKIN_CONVERTER(dropout);

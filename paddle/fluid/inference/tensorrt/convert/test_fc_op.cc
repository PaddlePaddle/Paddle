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
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(fc_op, test) {
  std::unordered_set<std::string> parameters({"mul-Y"});
  framework::Scope scope;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("mul-X", nvinfer1::Dims3(10, 1, 1));
  validator.DeclParamVar("mul-Y", nvinfer1::Dims2(10, 2));
  validator.DeclOutputVar("mul-Out", nvinfer1::Dims2(1, 2));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("mul");
  desc.SetInput("X", {"mul-X"});
  desc.SetInput("Y", {"mul-Y"});
  desc.SetOutput("Out", {"mul-Out"});

  validator.SetOp(*desc.Proto());

  validator.Execute(10);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
USE_OP_ITSELF(mul);

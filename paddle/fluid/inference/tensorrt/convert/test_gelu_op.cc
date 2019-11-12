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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {

void test_gelu() {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  TRTConvertValidation validator(5, parameters, scope, 1 << 15);

  // The ITensor's Dims should not contain the batch size.
  // So, the ITensor's Dims of input and output should be C * H * W.
  validator.DeclInputVar("gelu-X", nvinfer1::Dims3(3, 6, 7));
  validator.DeclOutputVar("gelu-Out", nvinfer1::Dims3(3, 6, 7));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("gelu");
  desc.SetInput("X", {"gelu-X"});
  desc.SetOutput("Out", {"gelu-Out"});

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(3);
}
TEST(GeluOpConverter, main) { test_gelu(); }
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(gelu);

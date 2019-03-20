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

TEST(PadConverter, main) {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("pad-X", nvinfer1::Dims3(3, 2, 2));
  validator.DeclOutputVar("pad-Out", nvinfer1::Dims3(3, 3, 5));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("pad");
  desc.SetInput("X", {"pad-X"});
  desc.SetOutput("Out", {"pad-Out"});

  std::vector<int> paddings = {0, 0, 0, 0, 0, 1, 1, 2};
  float pad_value = 0.0;
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("pad_value", pad_value);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(2);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(pad);

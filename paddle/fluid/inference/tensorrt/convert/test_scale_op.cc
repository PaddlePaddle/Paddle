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

void test_scale_converter(bool bias_after_scale) {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("scale-X", nvinfer1::DimsHW(3, 2));
  validator.DeclOutputVar("scale-Out", nvinfer1::DimsHW(3, 2));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("scale");
  desc.SetInput("X", {"scale-X"});
  desc.SetOutput("Out", {"scale-Out"});

  float scale = 2.0;
  float bias = 1.1;
  desc.SetAttr("scale", scale);
  desc.SetAttr("bias", bias);
  desc.SetAttr("bias_after_scale", bias_after_scale);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(2);
}

TEST(ScaleConverter, after) { test_scale_converter(true); }

TEST(ScaleConverter, before) { test_scale_converter(false); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(scale);

/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

TEST(nearest_interp_v2_op, test_swish) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("interp-X", nvinfer1::Dims3(3, 32, 32));
  validator.DeclOutputVar("interp-Out", nvinfer1::Dims3(3, 64, 64));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("nearest_interp_v2");
  desc.SetInput("X", {"interp-X"});
  desc.SetOutput("Out", {"interp-Out"});

  std::vector<float> scale({2.f, 2.f});

  desc.SetAttr("data_layout", "NCHW");
  desc.SetAttr("interp_method", "nearest");
  desc.SetAttr("align_corners", false);
  desc.SetAttr("scale", scale);
  desc.SetAttr("out_h", 0);
  desc.SetAttr("out_w", 0);

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(nearest_interp_v2);

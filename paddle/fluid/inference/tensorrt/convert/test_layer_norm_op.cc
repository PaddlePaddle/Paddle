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

TEST(layer_norm_op, test) {
  std::unordered_set<std::string> parameters(
      {"layer_norm_scale", "layer_norm_bias"});
  framework::Scope scope;
  TRTConvertValidation validator(5, parameters, scope, 1 << 15);

  std::vector<int> scale_bias_shape{1 * 2 * 2};
  std::vector<int> mean_var_shape{5};

  validator.DeclInputVar("layer_norm_X", nvinfer1::DimsCHW(1, 2, 2));
  validator.DeclParamVar("layer_norm_scale", scale_bias_shape);
  validator.DeclParamVar("layer_norm_bias", scale_bias_shape);
  validator.DeclOutputVar("layer_norm_Y", nvinfer1::DimsCHW(1, 2, 2));
  validator.DeclOutputVar("layer_norm_mean", mean_var_shape);
  validator.DeclOutputVar("layer_norm_variance", mean_var_shape);

  // Prepare Op description
  framework::OpDesc desc;

  desc.SetType("layer_norm");
  desc.SetInput("X", {"layer_norm_X"});
  desc.SetInput("Scale", {"layer_norm_scale"});
  desc.SetInput("Bias", {"layer_norm_bias"});
  desc.SetOutput("Y", {"layer_norm_Y"});
  desc.SetOutput("Mean", {"layer_norm_mean"});
  desc.SetOutput("Variance", {"layer_norm_variance"});

  float eps = 1e-5f;
  int begin_norm_axis = 1;

  desc.SetAttr("epsilon", eps);
  desc.SetAttr("begin_norm_axis", begin_norm_axis);

  validator.SetOp(*desc.Proto());

  std::unordered_set<std::string> neglected_output = {"layer_norm_variance",
                                                      "layer_norm_mean"};
  validator.Execute(3, neglected_output);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
USE_OP(layer_norm);

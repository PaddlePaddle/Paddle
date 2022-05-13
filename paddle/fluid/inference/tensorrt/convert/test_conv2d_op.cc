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

USE_OP_ITSELF(conv2d);
USE_OP_ITSELF(conv2d_transpose);

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(conv2d_op, test) {
  std::unordered_set<std::string> parameters({"conv2d-Y"});
  framework::Scope scope;
  TRTConvertValidation validator(5, parameters, scope, 1 << 15);

  validator.DeclInputVar("conv2d-X", nvinfer1::Dims3(2, 5, 5));
  validator.DeclParamVar("conv2d-Y", nvinfer1::Dims4(3, 2, 3, 3));
  validator.DeclOutputVar("conv2d-Out", nvinfer1::Dims3(3, 5, 5));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("conv2d");
  desc.SetInput("Input", {"conv2d-X"});
  desc.SetInput("Filter", {"conv2d-Y"});
  desc.SetOutput("Output", {"conv2d-Out"});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("dilations", dilations);
  desc.SetAttr("groups", groups);

  validator.SetOp(*desc.Proto());

  validator.Execute(3);
}

TEST(conv2d_transpose_op, test) {
  std::unordered_set<std::string> parameters({"deconv2d-Y"});
  framework::Scope scope;
  TRTConvertValidation validator(5, parameters, scope, 1 << 15);

  validator.DeclInputVar("deconv2d-X", nvinfer1::Dims3(3, 5, 5));
  validator.DeclParamVar("deconv2d-Y", nvinfer1::Dims4(3, 2, 3, 3));
  validator.DeclOutputVar("deconv2d-Out", nvinfer1::Dims3(2, 5, 5));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("conv2d_transpose");
  desc.SetInput("Input", {"deconv2d-X"});
  desc.SetInput("Filter", {"deconv2d-Y"});
  desc.SetOutput("Output", {"deconv2d-Out"});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("dilations", dilations);
  desc.SetAttr("groups", groups);

  validator.SetOp(*desc.Proto());

  validator.Execute(3);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

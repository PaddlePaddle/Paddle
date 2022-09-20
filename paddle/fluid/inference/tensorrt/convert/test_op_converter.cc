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

#include <gtest/gtest.h>  // NOLINT

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(OpConverter, ConvertBlock) {
  framework::ProgramDesc prog;
  auto* block = prog.MutableBlock(0);
  auto* conv2d_op = block->AppendOp();

  // init trt engine
  std::unique_ptr<TensorRTEngine> engine_;
  engine_.reset(new TensorRTEngine(5, 1 << 15));
  engine_->InitNetwork();

  engine_->DeclareInput(
      "conv2d-X", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3(2, 5, 5));

  conv2d_op->SetType("conv2d");
  conv2d_op->SetInput("Input", {"conv2d-X"});
  conv2d_op->SetInput("Filter", {"conv2d-Y"});
  conv2d_op->SetOutput("Output", {"conv2d-Out"});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  conv2d_op->SetAttr("strides", strides);
  conv2d_op->SetAttr("paddings", paddings);
  conv2d_op->SetAttr("dilations", dilations);
  conv2d_op->SetAttr("groups", groups);

  // init scope
  framework::Scope scope;
  std::vector<int> dim_vec = {3, 2, 3, 3};
  auto* x = scope.Var("conv2d-Y");
  auto* x_tensor = x->GetMutable<framework::LoDTensor>();
  x_tensor->Resize(phi::make_ddim(dim_vec));
  x_tensor->mutable_data<float>(platform::CUDAPlace(0));

  OpTeller::Global().SetOpConverterType("conv2d", OpConverterType::Default);
  OpConverter converter;
  converter.ConvertBlock(
      *block->Proto(), {"conv2d-Y"}, scope, engine_.get() /*TensorRTEngine*/);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_TRT_CONVERTER(conv2d)

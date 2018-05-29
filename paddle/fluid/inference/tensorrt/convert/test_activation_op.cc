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
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/io_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

USE_OP(relu);

namespace paddle {
namespace inference {
namespace tensorrt {

void Compare(const std::string op_type, float input, float expect) {
  framework::Scope scope;
  platform::CUDAPlace place;
  platform::CUDADeviceContext ctx(place);

  // init fluid op and variable
  auto x_var = scope.Var("X");
  auto x_tensor = x_var->GetMutable<framework::LoDTensor>();
  x_tensor->Resize({1, 1});
  x_tensor->mutable_data<float>(place);
  std::vector<float> init;
  init.push_back(input);
  framework::TensorFromVector(init, ctx, x_tensor);

  auto out_var = scope.Var("Out");
  auto out_tensor = out_var->GetMutable<framework::LoDTensor>();
  out_tensor->Resize({1, 1});
  out_tensor->mutable_data<float>(place);

  framework::OpDesc op_desc;
  op_desc.SetType(op_type);
  op_desc.SetInput("X", {"X"});
  op_desc.SetOutput("Out", {"Out"});

  auto op = framework::OpRegistry::CreateOp(*op_desc.Proto());

  // run fluid op
  op->Run(scope, place);
  // get fluid output
  std::vector<float> out1;
  framework::TensorToVector(*out_tensor, ctx, &out1);

  // init tensorrt op
  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));
  TensorRTEngine* engine = new TensorRTEngine(1, 1 << 10, &stream);
  engine->InitNetwork();
  engine->DeclareInput("X", nvinfer1::DataType::kFLOAT,
                       nvinfer1::DimsCHW{1, 1, 1});
  // convert op
  OpConverter op_converter;
  op_converter.ConvertOp(*op_desc.Proto(), engine);

  engine->DeclareOutput("Out");
  engine->FreezeNetwork();

  // convert LoDTensor to ITensor
  size_t size = x_tensor->memory_size();
  EngineIOConverter::ConvertInput(op_type, *x_tensor,
                                  engine->buffer("X").buffer, size, &stream);
  // run tensorrt Outp
  engine->Execute(1);
  // convert ITensor to LoDTensor
  EngineIOConverter::ConvertOutput(op_type, engine->buffer("Out").buffer,
                                   out_tensor, size, &stream);
  // get tensorrt output
  std::vector<float> out2;
  framework::TensorToVector(*out_tensor, ctx, &out2);

  // compare
  ASSERT_EQ(out1[0], out2[0]);
  ASSERT_EQ(out1[0], expect);

  delete engine;
  cudaStreamDestroy(stream);
}

TEST(OpConverter, ConvertRelu) {
  Compare("relu", 1, 1);   // relu(1) = 1
  Compare("relu", -5, 0);  // relu(-5) = 0
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(activation);

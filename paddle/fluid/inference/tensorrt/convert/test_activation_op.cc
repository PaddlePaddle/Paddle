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
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

USE_OP(relu);

namespace paddle {
namespace inference {
namespace tensorrt {

void Compare(float input, float expect) {
  framework::Scope scope;
  platform::CUDAPlace place;
  platform::CUDADeviceContext ctx(place);

  // init fluid op and variable
  auto x_var = scope.Var("X");
  auto x_tensor = x_var->GetMutable<framework::LoDTensor>();
  x_tensor->Resize({1, 1});
  std::vector<float> init;
  init.push_back(input);
  framework::TensorFromVector(init, ctx, x_tensor);

  auto out_var = scope.Var("Out");
  auto out_tensor = out_var->GetMutable<framework::LoDTensor>();
  out_tensor->Resize({1, 1});
  out_tensor->mutable_data<float>(place);

  framework::OpDesc op_desc;
  op_desc.SetType("relu");
  op_desc.SetInput("X", {"X"});
  op_desc.SetOutput("Out", {"Out"});

  auto relu_op = framework::OpRegistry::CreateOp(*op_desc.Proto());

  // run fluid op
  relu_op->Run(scope, place);
  std::vector<float> out1;
  framework::TensorToVector(*out_tensor, ctx, &out1);

  // init tensorrt op
  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));
  TensorRTEngine* engine = new TensorRTEngine(1, 1 << 10, &stream);
  engine->InitNetwork();
  engine->DeclareInput("X", nvinfer1::DataType::kFLOAT,
                       nvinfer1::DimsCHW{1, 1, 1});

  OpConverter op_converter;
  op_converter.ConvertOp(*op_desc.Proto(), engine);

  engine->DeclareOutput("Out");
  engine->FreezeNetwork();
  engine->SetInputFromCPU("X", &input, 1 * sizeof(float));

  // run tensorrt op
  engine->Execute(1);

  float out2;
  engine->GetOutputInCPU("Out", &out2, 1 * sizeof(float));

  ASSERT_EQ(out1[0], out2);
  ASSERT_EQ(out1[0], expect);

  delete engine;
  cudaStreamDestroy(stream);
}

TEST(OpConverter, ConvertRelu) {
  Compare(1, 1);   // relu(1) = 1
  Compare(-5, 0);  // relu(-5) = 0
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

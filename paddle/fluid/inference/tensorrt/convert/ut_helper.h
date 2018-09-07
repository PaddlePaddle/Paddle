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

/*
 * This file implements a UT framework to make the validation of transforming
 * Fluid Op to TRT Layer.
 */

#pragma once

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Get a random float value between [low, high]
 */
float random(float low, float high) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(1.0, 10.0);
  return dist(mt);
}

void RandomizeTensor(framework::LoDTensor* tensor, const platform::Place& place,
                     const platform::DeviceContext& ctx) {
  auto dims = tensor->dims();
  size_t num_elements = analysis::AccuDims(dims, dims.size());
  PADDLE_ENFORCE_GT(num_elements, 0);
  auto* data = tensor->mutable_data<float>(place);
  for (size_t i = 0; i < num_elements; i++) {
    *(data + i) = random(0., 1.);
  }
}

/*
 * Help to validate the correctness between Fluid Op and the corresponding TRT
 * layer.
 */
class TRTConvertValidation {
 public:
  TRTConvertValidation() = delete;

  TRTConvertValidation(int batch_size, int workspace_size = 1 << 10) {
    // create engine.
    engine_.reset(new TensorRTEngine(10, 1 << 10, &stream_));
    engine_->InitNetwork();

    PADDLE_ENFORCE_EQ(cudaStreamCreate(&stream_), 0);
  }

  // Declare a Variable as input with random initialization.
  void DeclInputVar(const std::string& name, const nvinfer1::Dims& dims) {
    DeclVar(name, dims);
    // Declare TRT inputs.
    engine_->DeclareInput(name, nvinfer1::DataType::kFLOAT, dims);
  }

  void DeclOutputVar(const std::string& name, const nvinfer1::Dims& dims) {
    DeclVar(name, dims);
  }

  void DeclVar(const std::string& name, const nvinfer1::Dims& dims) {
    platform::CPUPlace place;
    platform::CPUDeviceContext ctx(place);

    // Init Fluid tensor.
    std::vector<int> dim_vec(dims.nbDims);
    for (int i = 0; i < dims.nbDims; i++) {
      dim_vec[i] = dims.d[i];
    }
    auto* x = scope_.Var(name);
    auto* x_tensor = x->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dim_vec));
    RandomizeTensor(x_tensor, place, ctx);
  }

  void SetOp(const framework::proto::OpDesc& desc) {
    op_ = framework::OpRegistry::CreateOp(desc);

    OpConverter op_converter;
    op_converter.ConvertOp(desc, engine_.get());

    engine_->FreezeNetwork();

    // Declare outputs.
    op_desc_.reset(new framework::OpDesc(desc, nullptr, nullptr));

    // Set Inputs.
    for (const auto& input : op_desc_->InputArgumentNames()) {
      auto* var = scope_.FindVar(input);
      PADDLE_ENFORCE(var);
      auto tensor = var->GetMutable<framework::LoDTensor>();
      engine_->SetInputFromCPU(
          input, static_cast<void*>(tensor->data<float>()),
          sizeof(float) *
              analysis::AccuDims(tensor->dims(), tensor->dims().size()));
    }
  }

  void Execute(int batch_size) {
    // Execute Fluid Op
    // Execute TRT
    platform::CPUPlace place;
    platform::CPUDeviceContext ctx(place);
    engine_->Execute(batch_size);

    op_->Run(scope_, place);

    ASSERT_FALSE(op_desc_->OutputArgumentNames().empty());
    for (const auto& output : op_desc_->OutputArgumentNames()) {
      std::vector<float> fluid_out;
      std::vector<float> trt_out(200);
      engine_->GetOutputInCPU(output, &trt_out[0], 200 * sizeof(float));

      auto* var = scope_.FindVar(output);
      auto tensor = var->GetMutable<framework::LoDTensor>();
      framework::TensorToVector(*tensor, ctx, &fluid_out);
      // Compare two output
      ASSERT_FALSE(fluid_out.empty());
      for (size_t i = 0; i < fluid_out.size(); i++) {
        EXPECT_LT(std::abs(fluid_out[i] - trt_out[i]), 0.001);
      }
    }
  }

  framework::Scope& scope() { return scope_; }

 private:
  std::unique_ptr<TensorRTEngine> engine_;
  cudaStream_t stream_;
  framework::Scope scope_;
  std::unique_ptr<framework::OperatorBase> op_;
  std::unique_ptr<framework::OpDesc> op_desc_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

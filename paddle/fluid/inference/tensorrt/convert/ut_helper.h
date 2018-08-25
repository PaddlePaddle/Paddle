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

#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Get a random float value between [low, high]
 */
float random(float low, float high) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

void RandomizeTensor(framework::LoDTensor* tensor, const platform::Place& place,
                     const platform::DeviceContext& ctx) {
  auto dims = tensor->dims();
  size_t num_elements = analysis::AccuDims(dims, dims.size());
  PADDLE_ENFORCE_GT(num_elements, 0);

  platform::CPUPlace cpu_place;
  framework::LoDTensor temp_tensor;
  temp_tensor.Resize(dims);
  auto* temp_data = temp_tensor.mutable_data<float>(cpu_place);

  for (size_t i = 0; i < num_elements; i++) {
    *(temp_data + i) = random(0., 1.);
  }

  TensorCopySync(temp_tensor, place, tensor);
}

/*
 * Help to validate the correctness between Fluid Op and the corresponding TRT
 * layer.
 */
class TRTConvertValidation {
 public:
  TRTConvertValidation() = delete;

  TRTConvertValidation(int max_batch_size,
                       const std::unordered_set<std::string>& parameters,
                       framework::Scope& scope,  // NOLINT
                       int workspace_size = 1 << 10, bool if_add_batch = true)
      : parameters_(parameters),
        scope_(scope),
        if_add_batch_(if_add_batch),
        max_batch_size_(max_batch_size) {
    // create engine.
    engine_.reset(new TensorRTEngine(max_batch_size, workspace_size, &stream_));
    engine_->InitNetwork();

    PADDLE_ENFORCE_EQ(cudaStreamCreate(&stream_), 0);
  }

  // Declare a Variable as input with random initialization.
  void DeclInputVar(const std::string& name, const std::vector<int> tensor_dims,
                    const nvinfer1::Dims& trt_dims) {
    DeclVar(name, tensor_dims);
    engine_->DeclareInput(name, nvinfer1::DataType::kFLOAT, trt_dims);
  }

  void DeclInputVar(const std::string& name, const nvinfer1::Dims& dims) {
    DeclVar(name, dims);
    // Declare TRT inputs.
    engine_->DeclareInput(name, nvinfer1::DataType::kFLOAT, dims);
  }

  void DeclParamVar(const std::string& name, const std::vector<int> dim_vec) {
    DeclVar(name, dim_vec);
  }

  // Declare a parameter varaible in the scope.
  void DeclParamVar(const std::string& name, const nvinfer1::Dims& dims) {
    DeclVar(name, dims, true);
  }

  void DeclOutputVar(const std::string& name, const std::vector<int> dim_vec) {
    DeclVar(name, dim_vec);
  }

  void DeclOutputVar(const std::string& name, const nvinfer1::Dims& dims) {
    DeclVar(name, dims);
  }

  void DeclVar(const std::string& name, const std::vector<int> dim_vec) {
    platform::CUDAPlace place;
    platform::CUDADeviceContext ctx(place);

    auto* x = scope_.Var(name);
    auto* x_tensor = x->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dim_vec));
    RandomizeTensor(x_tensor, place, ctx);
  }
  // Declare a variable in a fluid Scope.
  void DeclVar(const std::string& name, const nvinfer1::Dims& dims,
               bool is_param = false) {
    // Init Fluid tensor.
    std::vector<int> dim_vec(dims.d, dims.d + dims.nbDims);
    // There is no batchsize in ITensor's shape, but We should add it to
    // tensor's shape of fluid. If the variable is not parameter and the
    // if_add_batch_ flag is true, add the max batchsize to dim_vec.
    if (is_param != true && if_add_batch_ == true)
      dim_vec.insert(dim_vec.begin(), max_batch_size_);

    DeclVar(name, dim_vec);
  }

  void SetOp(const framework::proto::OpDesc& desc) {
    op_ = framework::OpRegistry::CreateOp(desc);

    Singleton<OpConverter>::Global().ConvertOp(
        desc, parameters_, scope_, engine_.get(), true /*test_mode*/);

    engine_->FreezeNetwork();

    // Declare outputs.
    op_desc_.reset(new framework::OpDesc(desc, nullptr));

    // Set Inputs.
    for (const auto& input : op_desc_->InputArgumentNames()) {
      if (parameters_.count(input)) continue;
      auto* var = scope_.FindVar(input);
      PADDLE_ENFORCE(var);
      auto tensor = var->GetMutable<framework::LoDTensor>();

      engine_->SetInputFromGPU(
          input, static_cast<void*>(tensor->data<void>()),
          sizeof(float) *
              analysis::AccuDims(tensor->dims(), tensor->dims().size()));
    }
  }

  // We use the set 'neglected_output' here, because some Ops like batch norm,
  // the outputs specified in the op des are only used during training,
  // so we should neglect those output during inference.
  void Execute(int batch_size,
               std::unordered_set<std::string> neglected_output = {}) {
    // Execute Fluid Op
    PADDLE_ENFORCE_LE(batch_size, max_batch_size_);
    platform::CUDAPlace place;
    platform::CUDADeviceContext ctx(place);
    op_->Run(scope_, place);
    // Execute TRT.
    engine_->Execute(batch_size);
    cudaStreamSynchronize(*engine_->stream());

    ASSERT_FALSE(op_desc_->OutputArgumentNames().empty());
    const size_t output_space_size = 3000;
    for (const auto& output : op_desc_->OutputArgumentNames()) {
      if (neglected_output.count(output)) continue;
      std::vector<float> fluid_out;
      std::vector<float> trt_out(output_space_size);
      engine_->GetOutputInCPU(output, &trt_out[0], output_space_size);
      cudaStreamSynchronize(*engine_->stream());

      auto* var = scope_.FindVar(output);
      auto tensor = var->GetMutable<framework::LoDTensor>();
      framework::TensorToVector(*tensor, ctx, &fluid_out);

      size_t fluid_out_size = fluid_out.size();
      if (if_add_batch_ == true) {
        fluid_out_size =
            batch_size * (framework::product(tensor->dims()) / max_batch_size_);
      }
      // Compare two output
      ASSERT_FALSE(fluid_out.empty());
      for (size_t i = 0; i < fluid_out_size; i++) {
        // Loose the threshold for CI in different machine model.
        EXPECT_LT(std::abs(fluid_out[i] - trt_out[i]), 2e-5);
      }
    }
  }

  framework::Scope& scope() { return scope_; }

 private:
  std::unique_ptr<TensorRTEngine> engine_;
  cudaStream_t stream_;
  std::unique_ptr<framework::OperatorBase> op_;
  std::unique_ptr<framework::OpDesc> op_desc_;
  const std::unordered_set<std::string>& parameters_;
  framework::Scope& scope_;
  // The ITensor of trt does not cotain the batch size,
  // bug, in most cases, we need to set batch size for
  // fluid's tensor shape. This variable indicates
  // whether to add batch size to tensor shape of fluid.
  bool if_add_batch_;
  int max_batch_size_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

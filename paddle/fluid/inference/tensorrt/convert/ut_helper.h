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

#include <memory>
#include <string>
#include <unordered_set>
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
  static std::mt19937 mt(100);
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

void RandomizeTensor(phi::DenseTensor* tensor,
                     const phi::Place& place,
                     const phi::DeviceContext& ctx) {
  auto dims = tensor->dims();
  size_t num_elements = analysis::AccuDims(dims, dims.size());
  PADDLE_ENFORCE_GT(
      num_elements,
      0UL,
      common::errors::PermissionDenied("RandomizeTensor only can be used for "
                                       "tensor which dims is not zero."));

  phi::CPUPlace cpu_place;
  phi::DenseTensor temp_tensor;
  temp_tensor.Resize(dims);
  auto* temp_data = temp_tensor.mutable_data<float>(cpu_place);

  for (size_t i = 0; i < num_elements; i++) {
    *(temp_data + i) = random(0., 1.);
  }

  paddle::framework::TensorCopySync(temp_tensor, place, tensor);
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
                       int64_t workspace_size = 1 << 30,
                       bool if_add_batch = true)
      : parameters_(parameters),
        scope_(scope),
        if_add_batch_(if_add_batch),
        max_batch_size_(max_batch_size) {
    PADDLE_ENFORCE_EQ(cudaStreamCreate(&stream_),
                      0,
                      common::errors::External("cudaStreamCreate error."));
    TensorRTEngine::ConstructionParams params;
    params.max_batch_size = max_batch_size;
    params.max_workspace_size = workspace_size;
    engine_ = std::make_unique<TensorRTEngine>(params);
    engine_->InitNetwork();
  }

  // Declare a Variable as input with random initialization.
  void DeclInputVar(const std::string& name,
                    const std::vector<int> tensor_dims,
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

  // Declare a parameter variable in the scope.
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
    phi::GPUContext ctx(place_);

    auto* x = scope_.Var(name);
    auto* x_tensor = x->GetMutable<phi::DenseTensor>();
    x_tensor->Resize(common::make_ddim(dim_vec));
    RandomizeTensor(x_tensor, place_, ctx);
  }
  // Declare a variable in a fluid Scope.
  void DeclVar(const std::string& name,
               const nvinfer1::Dims& dims,
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
    op_desc_ = std::make_unique<framework::OpDesc>(desc, nullptr);
  }

  // We use the set 'neglected_output' here, because some Ops like batch norm,
  // the outputs specified in the op des are only used during training,
  // so we should neglect those output during inference.
  void Execute(int batch_size,
               std::unordered_set<std::string> neglected_output = {}) {
    // Execute Fluid Op
    PADDLE_ENFORCE_LE(batch_size,
                      max_batch_size_,
                      common::errors::InvalidArgument(
                          "Runtime batch_size should be less than or equal to "
                          "max_batch_size_. "
                          "But received batch_size:%d, max_batch_size_:%d",
                          batch_size,
                          max_batch_size_));
    phi::GPUContext ctx(place_);
    op_->Run(scope_, place_);
    cudaStreamSynchronize(stream_);
    std::vector<std::string> input_output_names;

    // Note: we need filter the parameter
    for (const auto& input : op_desc_->InputArgumentNames()) {
      if (parameters_.count(input)) continue;
      input_output_names.push_back(input);
    }

    // Collect the fluid outputs.
    std::vector<std::vector<float>> fluid_outs;
    for (const auto& output : op_desc_->OutputArgumentNames()) {
      if (neglected_output.count(output)) continue;
      input_output_names.push_back(output);
      std::vector<float> fluid_out;
      auto* var = scope_.FindVar(output);
      auto* tensor = var->GetMutable<phi::DenseTensor>();
      framework::TensorToVector(*tensor, ctx, &fluid_out);
      fluid_outs.push_back(fluid_out);
    }

    // Bind input and output for TRT.
    const int num_bindings = input_output_names.size();
    std::vector<void*> buffers(num_bindings);
#if IS_TRT_VERSION_GE(8600)
    std::unordered_map<std::string, int> tensor_index;
    for (int i = 0; i < engine_->engine()->getNbIOTensors(); ++i) {
      auto tensor_name = engine_->engine()->getIOTensorName(i);
      tensor_index[std::string(tensor_name)] = i;
    }
#endif
    for (const std::string& name : input_output_names) {
      auto* var = scope_.FindVar(name);
      auto* tensor = var->GetMutable<phi::DenseTensor>();
#if IS_TRT_VERSION_GE(10000)
      const int bind_index = tensor_index[std::string(name.c_str())];
#else
      const int bind_index = engine_->engine()->getBindingIndex(name.c_str());
#endif
      buffers[bind_index] =
          static_cast<void*>(tensor->mutable_data<float>(place_));
    }

    // Execute TRT.
    engine_->Execute(batch_size, &buffers, stream_);
    cudaStreamSynchronize(stream_);

    ASSERT_FALSE(op_desc_->OutputArgumentNames().empty());
    int index = 0;
    for (const auto& output : op_desc_->OutputArgumentNames()) {
      if (neglected_output.count(output)) continue;
      std::vector<float> trt_out;
      auto* var = scope_.FindVar(output);
      auto* tensor = var->GetMutable<phi::DenseTensor>();
      framework::TensorToVector(*tensor, ctx, &trt_out);

      size_t fluid_out_size = fluid_outs[index].size();
      if (if_add_batch_ == true) {
        fluid_out_size =
            batch_size * (common::product(tensor->dims()) / max_batch_size_);
      }

      for (size_t i = 0; i < fluid_out_size; i++) {
        // Loose the threshold for CI in different machine model.
        EXPECT_LT(std::abs(fluid_outs[index][i] - trt_out[i]), 2e-5);
      }
      index += 1;
    }
  }

  framework::Scope& scope() { return scope_; }

 private:
  phi::GPUPlace place_;
  std::unique_ptr<TensorRTEngine> engine_;
  cudaStream_t stream_;
  std::unique_ptr<framework::OperatorBase> op_;
  std::unique_ptr<framework::OpDesc> op_desc_;
  const std::unordered_set<std::string>& parameters_;
  framework::Scope& scope_;
  // The ITensor of trt does not contain the batch size,
  // bug, in most cases, we need to set batch size for
  // fluid's tensor shape. This variable indicates
  // whether to add batch size to tensor shape of fluid.
  bool if_add_batch_;
  int max_batch_size_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

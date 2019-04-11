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

#pragma once

#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/engine.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/enforce.h"

using anakin::Precision;
using anakin::saber::X86;

namespace paddle {
namespace inference {
namespace anakin {

/*
 * Get a random float value between [low, high]
 */
float random(float low, float high) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

void RandomizeTensor(framework::LoDTensor* tensor,
                     const platform::Place& place) {
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
 * Help to validate the correctness between Fluid Op and the corresponding
 * anakin
 * layer.
 */
template <typename TargetT, ::anakin::Precision PrecisionT>
class AnakinConvertValidation {
  using AnakinNvEngineT = AnakinEngine<TargetT, PrecisionT>;

 public:
  AnakinConvertValidation() = delete;

  AnakinConvertValidation(const std::unordered_set<std::string>& parameters,
                          framework::Scope* scope,
                          const platform::DeviceContext& ctx,
                          bool use_gpu = true)
      : parameters_(parameters), scope_(scope), ctx_(ctx), use_gpu_(use_gpu) {
    engine_.reset(new AnakinEngine<TargetT, PrecisionT>(true));
  }

  // Declare a Variable as input with random initialization.
  void DeclInputVar(const std::string& name,
                    const std::vector<int> tensor_dims) {
    DeclVar(name, tensor_dims);
    // should decalre anakin input here.
  }

  void DeclParamVar(const std::string& name, const std::vector<int> dim_vec) {
    DeclVar(name, dim_vec);
  }

  void DeclOutputVar(const std::string& name, const std::vector<int> dim_vec) {
    DeclVar(name, dim_vec);
    // should declare anakin output here.
  }

  void DeclVar(const std::string& name, const std::vector<int> dim_vec) {
    auto* x = scope_->Var(name);
    auto* x_tensor = x->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dim_vec));
    RandomizeTensor(x_tensor, ctx_.GetPlace());

    std::vector<int64_t> dim_vec_int64;
    for (auto& ele : dim_vec) {
      dim_vec_int64.push_back(static_cast<int64_t>(ele));
    }

    // Add var_desc to block_desc
    auto* block_desc = program_desc_.MutableBlock(framework::kRootBlockIndex);

    auto* var_desc = block_desc->Var(name);
    var_desc->SetShape(dim_vec_int64);
  }

  void SetOp(const framework::proto::OpDesc& desc) {
    op_ = framework::OpRegistry::CreateOp(desc);
    op_desc_.reset(new framework::OpDesc(desc, nullptr));
    // should init anakin engine here.

    auto& block_desc = program_desc_.Block(framework::kRootBlockIndex);
    Singleton<AnakinOpConverter<TargetT, PrecisionT>>::Global().ConvertOp(
        desc, block_desc, parameters_, *scope_, engine_.get(),
        true /*test_mode*/);
    engine_->Freeze();

    std::map<std::string, std::vector<int>> temp_max_input_shape;
    for (const auto& input : op_desc_->InputArgumentNames()) {
      if (parameters_.count(input)) continue;
      auto& t = inference::analysis::GetFromScope<framework::LoDTensor>(*scope_,
                                                                        input);
      auto t_shape = framework::vectorize2int(t.dims());
      while (t_shape.size() < 4) {
        t_shape.push_back(1);
      }
      engine_->SetInputShape(input, t_shape);
      temp_max_input_shape[input] = t_shape;
    }
    engine_->SetMaxInputShape(temp_max_input_shape);
    engine_->Optimize();
    engine_->InitNet();
  }

  // We use the set 'neglected_output' here, because some Ops like batch norm,
  // the outputs specified in the op des are only used during training,
  // so we should neglect those output during inference.
  void Execute(int batch_size,
               std::unordered_set<std::string> neglected_output = {}) {
    // Execute Fluid Op
    op_->Run(*scope_, ctx_.GetPlace());

    std::map<std::string, framework::LoDTensor*> inputs;
    for (const auto& input : op_desc_->InputArgumentNames()) {
      if (parameters_.count(input)) continue;
      auto* var = scope_->FindVar(input);
      auto tensor = var->GetMutable<framework::LoDTensor>();
      inputs.insert({input, tensor});
    }

    std::map<std::string, framework::LoDTensor*> outputs;
    std::vector<std::vector<float>> fluid_outputs;
    for (const auto& output : op_desc_->OutputArgumentNames()) {
      if (neglected_output.count(output)) continue;
      std::vector<float> fluid_out;
      auto* var = scope_->FindVar(output);
      auto tensor = var->GetMutable<framework::LoDTensor>();
      framework::TensorToVector(*tensor, ctx_, &fluid_out);
      fluid_outputs.push_back(fluid_out);

      outputs.insert({output, tensor});
    }

    if (!use_gpu_) {
      engine_->Execute(inputs, outputs);
    } else {
      cudaStream_t stream;
      PADDLE_ENFORCE_EQ(cudaStreamCreate(&stream), 0);
      engine_->Execute(inputs, outputs, stream);
    }

    int i_output = 0;
    for (const auto& output : op_desc_->OutputArgumentNames()) {
      if (neglected_output.count(output)) continue;
      std::vector<float> anakin_out;
      auto* var = scope_->FindVar(output);
      auto tensor = var->GetMutable<framework::LoDTensor>();
      framework::TensorToVector(*tensor, ctx_, &anakin_out);

      size_t anakin_out_size = anakin_out.size();
      auto fluid_out = fluid_outputs[i_output++];
      for (size_t i = 0; i < anakin_out_size; i++) {
        EXPECT_LT(std::abs(fluid_out[i] - anakin_out[i]), 1e-3);
      }
    }
  }

 private:
  std::unique_ptr<AnakinNvEngineT> engine_{nullptr};
  std::unique_ptr<framework::OperatorBase> op_;
  std::unique_ptr<framework::OpDesc> op_desc_;
  framework::ProgramDesc program_desc_;
  const std::unordered_set<std::string>& parameters_;
  framework::Scope* scope_;
  const platform::DeviceContext& ctx_;
  bool use_gpu_{true};
};

template class AnakinConvertValidation<::anakin::saber::NV,
                                       ::anakin::Precision::FP32>;
template class AnakinConvertValidation<::anakin::saber::X86,
                                       ::anakin::Precision::FP32>;

template class AnakinConvertValidation<::anakin::saber::NV,
                                       ::anakin::Precision::INT8>;
template class AnakinConvertValidation<::anakin::saber::X86,
                                       ::anakin::Precision::INT8>;
}  // namespace anakin
}  // namespace inference
}  // namespace paddle

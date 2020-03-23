//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
void PullSparseFunctor(
    const framework::ExecutionContext& ctx,
    const std::vector<uint64_t>* fea_keys_const,
    const std::vector<T*>* pull_result_ptr_const,
    const std::vector<::std::future<int32_t>>* pull_sparse_status_const) {
  const auto& inputs_const = ctx.MultiInput<framework::LoDTensor>("Ids");
  const auto& outputs_const = ctx.MultiOutput<framework::LoDTensor>("Out");
  uint32_t fea_dim = static_cast<uint32_t>(ctx.Attr<int>("EmbeddingDim"));
  uint64_t padding_id =static_cast<uint64_t>(ctx.Attr<int>("PaddingId"));
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));

  std::vector<uint64_t>* fea_keys =
      const_cast<std::vector<uint64_t>*>(fea_keys_const);
  std::vector<T*>* pull_result_ptr =
      const_cast<std::vector<T*>*>(pull_result_ptr_const);
  std::vector<::std::future<int32_t>>* pull_sparse_status =
      const_cast<std::vector<::std::future<int32_t>>*>(
          pull_sparse_status_const);
  std::vector<framework::LoDTensor*> inputs;
  inputs.reserve(inputs_const.size());
  for (const framework::LoDTensor* i : inputs_const) {
    inputs.push_back(const_cast<framework::LoDTensor*>(i));
  }
  std::vector<framework::LoDTensor*> outputs;
  outputs.reserve(outputs_const.size());
  for (const framework::LoDTensor* i : outputs_const) {
    outputs.push_back(const_cast<framework::LoDTensor*>(i));
  }

  auto fleet_ptr = framework::FleetWrapper::GetInstance();
  fleet_ptr->PullSparseToTensorSync(
      table_id, fea_dim, padding_id, ctx.GetPlace(), &inputs, &outputs,
      fea_keys, pull_result_ptr, pull_sparse_status);
}


template <typename T>
void PushSparseFunctor(
    const framework::ExecutionContext& ctx,
    const std::vector<uint64_t>* push_keys_const,
    const std::vector<std::vector<T>>* push_values_const,
    const std::vector<T>* fea_labels_const) {
  const auto& inputs_const = ctx.MultiInput<framework::LoDTensor>("Ids");
  const auto& outputs_const =
      ctx.MultiInput<framework::LoDTensor>(framework::GradVarName("Out"));
  uint32_t fea_dim = static_cast<uint32_t>(ctx.Attr<int>("EmbeddingDim"));
  std::string accesor = ctx.Attr<std::string>("AccessorClass");
  bool scale_sparse = ctx.Attr<bool>("ScaleSparseGrad");
  uint64_t padding_id =static_cast<uint64_t>(ctx.Attr<int>("PaddingId"));
  const std::string& label_name = ctx.Attr<std::string>("CtrLabelName");
  const framework::Scope& scope = ctx.scope();
  auto input_names = ctx.Attr<std::vector<std::string>>("InputNames");
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));
  std::vector<uint64_t>* push_keys =
      const_cast<std::vector<uint64_t>*>(push_keys_const);
  std::vector<std::vector<T>>* push_values =
      const_cast<std::vector<std::vector<T>>*>(push_values_const);
  std::vector<T>* fea_labels = const_cast<std::vector<T>*>(fea_labels_const);
  std::vector<framework::LoDTensor*> inputs;
  inputs.reserve(inputs_const.size());
  for (const framework::LoDTensor* i : inputs_const) {
    inputs.push_back(const_cast<framework::LoDTensor*>(i));
  }
  std::vector<framework::LoDTensor*> outputs;
  outputs.reserve(outputs_const.size());
  for (const framework::LoDTensor* i : outputs_const) {
    outputs.push_back(const_cast<framework::LoDTensor*>(i));
  }

  auto fleet_ptr = framework::FleetWrapper::GetInstance();
  fleet_ptr->PushSparseFromTensorWithLabelAsync(
      scope, table_id, fea_dim, padding_id, scale_sparse, accesor, label_name,
      ctx.GetPlace(), input_names, &inputs, &outputs, push_keys, push_values,
      fea_labels);
}

template <typename T>
class PullSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PullSparseFunctor<T>(ctx, &fea_keys_, &pull_result_ptr_,
                         &pull_sparse_status_);
  }
 protected:
   std::vector<uint64_t> fea_keys_;
   std::vector<T*> pull_result_ptr_;
   std::vector<::std::future<int32_t>> pull_sparse_status_;
};

template <typename T>
class PushSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushSparseFunctor<T>(ctx, &push_keys_, &push_values_, &fea_labels_);
  }

 protected:
  std::vector<uint64_t> push_keys_;
  std::vector<std::vector<T>> push_values_;
  std::vector<T> fea_labels_;
};
}  // namespace operators
}  // namespace paddle

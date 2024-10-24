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
void PullSparseFunctor(const framework::ExecutionContext& ctx) {
  auto inputs = ctx.MultiInput<phi::DenseTensor>("Ids");
  auto outputs = ctx.MultiOutput<phi::DenseTensor>("Out");
  uint32_t fea_dim = static_cast<uint32_t>(ctx.Attr<int>("EmbeddingDim"));
  uint64_t padding_id = static_cast<uint64_t>(ctx.Attr<int>("PaddingId"));
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));
  // note: GetInstance() is not thread-safe
  // we assume FleetWrapper has been already initialized
  auto fleet_ptr = framework::FleetWrapper::GetInstance();
  fleet_ptr->PullSparseToTensorSync(
      table_id, fea_dim, padding_id, ctx.GetPlace(), &inputs, &outputs);
}

template <typename T>
void PushSparseFunctor(const framework::ExecutionContext& ctx) {
  auto inputs = ctx.MultiInput<phi::DenseTensor>("Ids");
  auto grads = ctx.MultiInput<phi::DenseTensor>(framework::GradVarName("Out"));
  uint32_t fea_dim = static_cast<uint32_t>(ctx.Attr<int>("EmbeddingDim"));
  std::string accessor = ctx.Attr<std::string>("AccessorClass");
  bool scale_sparse = ctx.Attr<bool>("ScaleSparseGrad");
  uint64_t padding_id = static_cast<uint64_t>(ctx.Attr<int>("PaddingId"));
  const std::string& label_name = ctx.Attr<std::string>("CtrLabelName");
  const framework::Scope& scope = ctx.scope();
  auto input_names = ctx.Attr<std::vector<std::string>>("InputNames");
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));
  // note: GetInstance() is not thread-safe
  // we assume FleetWrapper has been already initialized
  auto fleet_ptr = framework::FleetWrapper::GetInstance();
  fleet_ptr->PushSparseFromTensorWithLabelAsync(scope,
                                                table_id,
                                                fea_dim,
                                                padding_id,
                                                scale_sparse,
                                                accessor,
                                                label_name,
                                                ctx.GetPlace(),
                                                input_names,
                                                &inputs,
                                                &grads);
}

template <typename T, typename DeviceContext>
class PullSparseV2CPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PullSparseFunctor<T>(ctx);
  }
};

template <typename T, typename DeviceContext>
class PushSparseV2CPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PushSparseFunctor<T>(ctx);
  }
};
}  // namespace operators
}  // namespace paddle

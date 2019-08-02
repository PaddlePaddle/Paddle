//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
template <typename T>
class PullBoxSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    printf("paddlebox: pull box sparse ff start..\n");
    auto inputs = ctx.MultiInput<framework::Tensor>("Ids");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");
    auto hidden_size = ctx.Attr<int>("size");
    printf("paddlebox: hidden size in op: %d\n", hidden_size);

    const auto slot_size = inputs.size();
    std::vector<std::vector<uint64_t>> all_keys(slot_size);
    std::vector<std::vector<float>> all_values(slot_size);
    for (auto i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
      const auto key_numel = slot->numel();

      printf("paddlebox: numel in the %d slot is %ld\n", i, key_numel);
      all_values[i].resize(hidden_size * key_numel);
      all_keys[i].resize(key_numel);
      memcpy(all_keys[i].data(), single_slot_keys,
             key_numel * sizeof(uint64_t));
    }

    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    box_ptr->PullSparsePara(ctx.scope(), ctx.GetPlace(), all_keys, &all_values);

    for (size_t i = 0; i < slot_size; ++i) {
      auto *output = outputs[i]->mutable_data<float>(ctx.GetPlace());
      memcpy(output, all_values[i].data(),
             all_values[i].size() * sizeof(float));
    }
  }
};

template <typename T>
class PullBoxSparseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<framework::Tensor>("Ids");
    auto d_output =
        ctx.MultiInput<framework::Tensor>(framework::GradVarName("Out"));

    auto hidden_size = ctx.Attr<int>("size");

    const auto slot_size = inputs.size();
    std::vector<std::vector<uint64_t>> all_keys(slot_size);
    std::vector<std::vector<float>> all_grad_values(slot_size);
    for (auto i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
      const float *grad_value = d_output[i]->data<float>();

      const auto key_numel = slot->numel();
      all_grad_values[i].resize(hidden_size * key_numel);
      all_keys[i].resize(key_numel);
      memcpy(all_keys[i].data(), single_slot_keys,
             key_numel * sizeof(uint64_t));
      memcpy(all_grad_values[i].data(), grad_value,
             hidden_size * key_numel * sizeof(float));
    }

    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    box_ptr->PushSparseGrad(ctx.scope(), ctx.GetPlace(), all_keys,
                            all_grad_values);
  }
};
}  // namespace operators
}  // namespace paddle

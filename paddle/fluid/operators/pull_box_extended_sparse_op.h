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
#include <vector>

#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
static void PullBoxExtendedSparseFunctor(
    const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<phi::DenseTensor>("Ids");
  auto outputs = ctx.MultiOutput<phi::DenseTensor>("Out");
  auto outputs_extend = ctx.MultiOutput<phi::DenseTensor>("OutExtend");
  const auto slot_size = inputs.size();
  std::vector<const uint64_t *> all_keys(slot_size);
  // BoxPS only supports float now
  std::vector<float *> all_values(slot_size * 2);
  std::vector<int64_t> slot_lengths(slot_size);
  for (size_t i = 0; i < slot_size; i++) {
    const auto *slot = inputs[i];
    const uint64_t *single_slot_keys =
        reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
    all_keys[i] = single_slot_keys;
    slot_lengths[i] = slot->numel();
    auto *output = outputs[i]->mutable_data<T>(ctx.GetPlace());
    auto *output_extend = outputs_extend[i]->mutable_data<T>(ctx.GetPlace());
    all_values[i] = reinterpret_cast<float *>(output);
    all_values[i + slot_size] = reinterpret_cast<float *>(output_extend);
  }
#ifdef PADDLE_WITH_BOX_PS
  auto emb_size = ctx.Attr<int>("emb_size");
  auto emb_extended_size = ctx.Attr<int>("emb_extended_size");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->PullSparse(ctx.GetPlace(),
                      all_keys,
                      all_values,
                      slot_lengths,
                      emb_size,
                      emb_extended_size);
#endif
}

template <typename T>
static void PushBoxExtendedSparseFunctor(
    const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<framework::LoDTensor>("Ids");
  auto d_output =
      ctx.MultiInput<phi::DenseTensor>(framework::GradVarName("Out"));
  auto d_output_extend =
      ctx.MultiInput<phi::DenseTensor>(framework::GradVarName("OutExtend"));
  const auto slot_size = inputs.size();
  std::vector<const uint64_t *> all_keys(slot_size);
  std::vector<const float *> all_grad_values(slot_size * 2);
  std::vector<int64_t> slot_lengths(slot_size);
  int batch_size = -1;
  for (size_t i = 0; i < slot_size; i++) {
    const auto *slot = inputs[i];
    const uint64_t *single_slot_keys =
        reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
    all_keys[i] = single_slot_keys;
    slot_lengths[i] = slot->numel();
    int cur_batch_size =
        slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
    if (batch_size == -1) {
      batch_size = cur_batch_size;
    } else {
      PADDLE_ENFORCE_EQ(batch_size,
                        cur_batch_size,
                        platform::errors::PreconditionNotMet(
                            "The batch size of all input slots should be same,"
                            "please cheack"));
    }
    const float *grad_value = d_output[i]->data<float>();
    const float *grad_value_extend = d_output_extend[i]->data<float>();
    all_grad_values[i] = reinterpret_cast<const float *>(grad_value);
    all_grad_values[i + slot_size] =
        reinterpret_cast<const float *>(grad_value_extend);
  }
#ifdef PADDLE_WITH_BOX_PS
  auto emb_size = ctx.Attr<int>("emb_size");
  auto emb_extended_size = ctx.Attr<int>("emb_extended_size");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->PushSparseGrad(ctx.GetPlace(),
                          all_keys,
                          all_grad_values,
                          slot_lengths,
                          emb_size,
                          emb_extended_size,
                          batch_size);
#endif
}

using LoDTensor = framework::LoDTensor;
template <typename T>
class PullBoxExtendedSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullBoxExtendedSparseFunctor<T>(ctx);
  }
};

template <typename T>
class PushBoxExtendedSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushBoxExtendedSparseFunctor<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/eager/gradient_accumulation.h"
#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/dtype.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

/**
 * Implementation of GradNodeBase, Edge and InputBuffer.
**/
namespace egr {

GradNodeBase::GradNodeBase(size_t bwd_in_slot_num, size_t bwd_out_slot_num) {
  bwd_in_meta_.resize(bwd_in_slot_num);
  bwd_out_meta_.resize(bwd_out_slot_num);
  adj_edges_.resize(bwd_out_slot_num);
}

void GradNodeBase::AddEdges(const std::vector<AutogradMeta*>& metas,
                            size_t slot_id) {
  PADDLE_ENFORCE_LT(slot_id, adj_edges_.size(),
                    "Given slot id is out of range of adj_edges outter size");
  for (const auto& meta : metas) {
    // adj_edges has as same rank as fwd inputs, and record it's output rank
    // from
    // its pre-ops
    adj_edges_[slot_id].emplace_back(meta->GetMutableGradNode(),
                                     meta->OutRankInfo());
  }
}

void GradNodeBase::AddEdges(const AutogradMeta& meta,
                            size_t slot_id) {
  PADDLE_ENFORCE_LT(slot_id, adj_edges_.size(),
                    "Given slot id is out of range of adj_edges outter size");
    adj_edges_[slot_id].emplace_back(meta.GetMutableGradNode(),
                                     meta.OutRankInfo());
}

const std::vector<GradSlotMeta>& GradNodeBase::InputMeta() const {
  return bwd_in_meta_;
}

const std::vector<GradSlotMeta>& GradNodeBase::OutputMeta() const {
  return bwd_out_meta_;
}

void GradNodeBase::SetGradInMeta(const std::vector<AutogradMeta*>& fwd_out,
                                      size_t slot_rank) {
  size_t slot_size = fwd_out.size();
  PADDLE_ENFORCE_LE(slot_rank, (bwd_in_meta_.size() - 1),
                    "Slot Rank should less equal than bwd_in_meta_ size.");
  auto& meta = bwd_in_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    "Bwd_in_meta should only be init once.");
  // Init stop gradient vector before use to avoid push back
  meta.Init(slot_size);
  for (size_t i = 0; i < slot_size; i++) {
    if (fwd_out[i]->StopGradient()) {
      // Set Stop Gradient only when its true, since all default value is false.
      meta.SetStopGradient(i, fwd_out[i]->StopGradient());
    }
  }
}

void GradNodeBase::SetMultiGradInMeta(const std::vector<AutogradMeta*>& fwd_out,
                                      size_t slot_rank) {
    SetGradInMeta(fwd_out, slot_rank);
}

void GradNodeBase::SetGradInMeta(const AutogradMeta& fwd_out,
                                 size_t slot_rank) {
  PADDLE_ENFORCE_LE(slot_rank, (bwd_in_meta_.size() - 1),
                    "Slot Rank should less equal than bwd_in_meta_ size.");
  auto& meta = bwd_in_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    "Bwd_in_meta should only be init once.");
  // Init stop gradient vector before use to avoid push back
  VLOG(7) << "Init bwd_in_meta_ with slot rank: " << slot_rank;
  meta.Init(1);
  meta.SetStopGradient(0, fwd_out.StopGradient());
}

void GradNodeBase::SetGradOutMeta(const std::vector<AutogradMeta*>& fwd_in,
                                       size_t slot_rank) {
  size_t slot_size = fwd_in.size();
  PADDLE_ENFORCE_LE(slot_rank, (bwd_out_meta_.size() - 1),
                    "Slot Rank should less equal than bwd_out_meta_ size.");
  auto& meta = bwd_out_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    "Bwd_out_meta should only be init once.");
  // Init stop gradient vector before use to avoid push back
  meta.Init(slot_size);
  for (size_t i = 0; i < slot_size; i++) {
    if (fwd_in[i]->StopGradient()) {
      // Set Stop Gradient only when its true, since all default value is false.
      meta.SetStopGradient(i, fwd_in[i]->StopGradient());
    }
  }
}

void GradNodeBase::SetMultiGradOutMeta(const std::vector<AutogradMeta*>& fwd_in,
                                       size_t slot_rank) {
    SetGradOutMeta(fwd_in, slot_rank);
}

void GradNodeBase::SetGradOutMeta(const AutogradMeta& fwd_in,
                                  size_t slot_rank) {
  PADDLE_ENFORCE_LE(slot_rank, (bwd_out_meta_.size() - 1),
                    "Slot Rank should less equal than bwd_out_meta_ size.");
  auto& meta = bwd_out_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    "Bwd_out_meta should only be init once.");
  // Init stop gradient vector before use to avoid push back
  meta.Init(1);
  meta.SetStopGradient(0, fwd_in.StopGradient());
}

void GradNodeBase::SetDefaultGradInOutMeta() {
  PADDLE_ENFORCE(
      (bwd_out_meta_.size() == 1) && (bwd_in_meta_.size() == 1),
      paddle::platform::errors::Fatal(
          "We can only support 1 in 1 out default grad meta setter"));
  // Default stop_gradient is false and slot id is 0, slot size is 1;
  bwd_out_meta_[0].Init(1);
  bwd_in_meta_[0].Init(1);
}

const std::vector<std::vector<Edge>>& GradNodeBase::GetEdges() const {
  return adj_edges_;
}

void GradNodeBase::RegisterGradientHook(
    size_t slot_id, size_t rank,
    const std::function<pt::Tensor(const pt::Tensor&)>& hook) {
  gradient_hooks_.emplace_back(std::make_tuple(slot_id, rank, hook));
}

void GradNodeBase::RegisterReduceHook(const std::function<void(void)>& hook) {
  reduce_hooks_.emplace_back(hook);
}

std::vector<std::vector<pt::Tensor>> GradNodeBase::ApplyGradientHooks(
    const std::vector<std::vector<pt::Tensor>>& tensors) {
  std::vector<std::vector<pt::Tensor>> outs(tensors.size());
  for (auto& tuple : gradient_hooks_) {
    size_t slot_id = std::get<0>(tuple);
    size_t rank = std::get<1>(tuple);
    std::function<pt::Tensor(const pt::Tensor&)>& hook = std::get<2>(tuple);

    PADDLE_ENFORCE(slot_id < tensors.size(),
                   paddle::platform::errors::Fatal(
                       "Slot_id from registered hook should be smaller than "
                       "slot size of grad_tensors"));

    PADDLE_ENFORCE(rank < tensors[slot_id].size(),
                   paddle::platform::errors::Fatal(
                       "rank of slot %d from registered hook should be smaller "
                       "than rank size of grad_tensors",
                       slot_id));

    std::vector<pt::Tensor>& slot_out = outs[slot_id];
    slot_out.resize(tensors[slot_id].size());
    pt::Tensor& out = slot_out[rank];
    if (!out.defined() || !out.initialized()) {
      out = hook(tensors[slot_id][rank]);
    } else {
      // TODO(jiabin): Why this?
      out = hook(out);
    }
  }

  for (size_t i = 0; i < outs.size(); i++) {
    if (outs[i].empty() && (!tensors[i].empty())) {
      outs[i].resize(tensors[i].size());
    }
    // TODO(Jiabin): Optimize this if we only add hook slot by slot
    for (size_t j = 0; j < outs[i].size(); j++) {
      if (!outs[i][j].defined() || !outs[i][j].initialized()) {
        outs[i][j] = tensors[i][j];
      }
    }
  }

  return outs;
}

void GradNodeBase::ApplyReduceHooks() {
  for (auto& hook : reduce_hooks_) {
    hook();
  }
}

void InputBuffer::add(size_t slot_id, size_t rank, const pt::Tensor& t,
                      bool fill_one) {
  // TODO(jiabin): We need to deal with empty input_buffer with slot size not
  // empty;
  PADDLE_ENFORCE(
      slot_id < buffer_.size(),
      paddle::platform::errors::Fatal("Invalid slot_id for InputBuffer::add() "
                                      "which exceeds size of buffer"));
  VLOG(6) << "Add Tensor for buffer_ slot: " << slot_id
          << ", size: " << buffer_[slot_id].size();
  if (buffer_[slot_id].empty()) {
    VLOG(6) << "Pass add Tensor for buffer_ slot: " << slot_id
            << " since its buffer_ is empty ";
    return;
  }
  PADDLE_ENFORCE(rank < buffer_[slot_id].size(),
                 paddle::platform::errors::Fatal(
                     "Invalid rank for InputBuffer::add() which exceeds size "
                     "of buffer slot %d, got slot size is: %d rank is: %d",
                     slot_id, buffer_[slot_id].size(), rank));
  pt::Tensor& buffer_tensor = buffer_[slot_id][rank];
  if (!fill_one) {
    if (!buffer_tensor.defined() || !buffer_tensor.initialized()) {
      // Simply copy tensor->impl
      buffer_tensor = t;

    } else {
      // Accumulation
      TensorAdd(t, &buffer_tensor);
    }
  } else {
    // Create new tensor->impl and fill it with 1.0
    auto t_impl = t.impl();

    // Fill 1.0
    FillConstAPI(1.0, t_impl->dims(), t_impl->backend(), t_impl->type(),
                 t_impl->layout(), &buffer_tensor);
  }
}

}  // namespace egr

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
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/dense_tensor.h"

#include "paddle/fluid/framework/var_type.h"
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
  // adj_edges has the same num as backward outputs
  adj_edges_.resize(bwd_out_slot_num);
}

void GradNodeBase::AddEdges(std::vector<AutogradMeta*>* metas, size_t slot_id) {
  PADDLE_ENFORCE_LT(
      slot_id, adj_edges_.size(),
      paddle::platform::errors::InvalidArgument(
          "Given slot id is out of range of adj_edges outter size, "
          "adj_edges is designed to has the same size of grad "
          "inputs's slot num."));
  for (const auto& meta : *metas) {
    // adj_edges has as same rank as fwd inputs, and record it's output rank
    // from
    // its pre-ops
    if (meta && !meta->StopGradient()) {
      auto node = meta->GetMutableGradNode();
      if (node) {
        adj_edges_[slot_id].emplace_back(meta->GetMutableGradNode(),
                                         meta->OutRankInfo());
      } else {
        meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>());
        adj_edges_[slot_id].emplace_back(meta->GetMutableGradNode(),
                                         meta->OutRankInfo());
      }
    }
  }
}

void GradNodeBase::AddEdges(AutogradMeta* meta, size_t slot_id) {
  PADDLE_ENFORCE_LT(
      slot_id, adj_edges_.size(),
      paddle::platform::errors::InvalidArgument(
          "Given slot id is out of range of adj_edges outter size, "
          "adj_edges is designed to has the same size of grad "
          "inputs's slot num."));
  if (meta && !meta->StopGradient()) {
    VLOG(6) << "Add Edges for slot: " << slot_id;
    auto node = meta->GetMutableGradNode();
    if (node) {
      adj_edges_[slot_id].emplace_back(meta->GetMutableGradNode(),
                                       meta->OutRankInfo());
    } else {
      meta->SetGradNode(std::make_shared<egr::GradNodeAccumulation>());
      adj_edges_[slot_id].emplace_back(meta->GetMutableGradNode(),
                                       meta->OutRankInfo());
    }
  }
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
  PADDLE_ENFORCE_LE(
      slot_rank, (bwd_in_meta_.size() - 1),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_in_meta_ size, since "
          "bwd_in_meta_ is designed to hold as same num as backward "
          "inputs."));
  auto& meta = bwd_in_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    paddle::platform::errors::PreconditionNotMet(
                        "Bwd_in_meta should only be init once, addition "
                        "initialization for it is forbidden. If you got this "
                        "error, it indicates bugs in framework."));
  // Init stop gradient vector before use to avoid push back
  meta.Init(slot_size);
  for (size_t i = 0; i < slot_size; i++) {
    PADDLE_ENFORCE_NOT_NULL(fwd_out[i],
                            paddle::platform::errors::PreconditionNotMet(
                                "Bwd_in_meta should only be called while "
                                "autograd_meta is not null. If you got this "
                                "error, it indicates bugs in framework."));
    if (fwd_out[i]->StopGradient()) {
      // Set Stop Gradient only when its true or non-initialized autograd_meta,
      // since all default value is false.
      meta.SetStopGradient(i, fwd_out[i]->StopGradient());
    }
  }
}

void GradNodeBase::SetGradInMeta(const AutogradMeta& fwd_out,
                                 size_t slot_rank) {
  PADDLE_ENFORCE_LE(
      slot_rank, (bwd_in_meta_.size() - 1),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_in_meta_ size, since "
          "bwd_in_meta_ is designed to hold as same num as backward "
          "inputs."));
  auto& meta = bwd_in_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    paddle::platform::errors::PreconditionNotMet(
                        "Bwd_in_meta should only be init once, Additional "
                        "initialization for it is forbidden. If you got this "
                        "error, it indicates bugs in framework."));
  // Init stop gradient vector before use to avoid push back
  VLOG(7) << "Init bwd_in_meta_ with slot rank: " << slot_rank;
  meta.Init(1);
  meta.SetStopGradient(0, fwd_out.StopGradient());
}

void GradNodeBase::SetGradOutMeta(const std::vector<AutogradMeta*>& fwd_in,
                                  size_t slot_rank) {
  size_t slot_size = fwd_in.size();
  PADDLE_ENFORCE_LE(
      slot_rank, (bwd_out_meta_.size() - 1),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& meta = bwd_out_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    paddle::platform::errors::PreconditionNotMet(
                        "Bwd_out_meta should only be init once. Additional "
                        "initialization for it is forbidden. If you got this "
                        "error, it indicates bugs in framework."));
  // Init stop gradient vector before use to avoid push back
  meta.Init(slot_size);
  for (size_t i = 0; i < slot_size; i++) {
    if (!fwd_in[i]) {
      meta.SetStopGradient(i, true);
      continue;
    }
    if (fwd_in[i]->StopGradient()) {
      // Set Stop Gradient only when its true or non-initialized autograd_meta,
      // since all default value is false.
      meta.SetStopGradient(i, fwd_in[i]->StopGradient());
    }
  }
}

void GradNodeBase::SetGradOutMeta(const AutogradMeta& fwd_in,
                                  size_t slot_rank) {
  PADDLE_ENFORCE_LE(
      (slot_rank + 1), bwd_out_meta_.size(),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& meta = bwd_out_meta_.at(slot_rank);
  PADDLE_ENFORCE_EQ(meta.IsInitialized(), false,
                    paddle::platform::errors::PreconditionNotMet(
                        "Bwd_out_meta should only be init once. Additional "
                        "initialization for it is forbidden. If you got this "
                        "error, it indicates bugs in framework."));
  // Init stop gradient vector before use to avoid push back
  meta.Init(1);
  meta.SetStopGradient(0, fwd_in.StopGradient());
}

void GradNodeBase::SetDefaultGradInOutMeta() {
  PADDLE_ENFORCE((bwd_out_meta_.size() == 1) && (bwd_in_meta_.size() == 1),
                 paddle::platform::errors::PreconditionNotMet(
                     "We can only support 1 input and 1 output in default grad "
                     "meta setter, other size of inputs and outputs should "
                     "create with Setter and Getters"));
  // Default stop_gradient is false and slot id is 0, slot size is 1;
  bwd_out_meta_[0].Init(1);
  bwd_in_meta_[0].Init(1);
}

const std::vector<std::vector<Edge>>& GradNodeBase::GetEdges() const {
  return adj_edges_;
}

void GradNodeBase::RegisterGradientHook(
    size_t slot_id, size_t rank,
    const std::function<egr::EagerTensor(const egr::EagerTensor&)>& hook) {
  gradient_hooks_.emplace_back(std::make_tuple(slot_id, rank, hook));
}

void GradNodeBase::RegisterReduceHook(const std::function<void(void)>& hook) {
  reduce_hooks_.emplace_back(hook);
}

std::vector<std::vector<egr::EagerTensor>> GradNodeBase::ApplyGradientHooks(
    const std::vector<std::vector<egr::EagerTensor>>& tensors) {
  std::vector<std::vector<egr::EagerTensor>> outs(tensors.size());
  for (auto& tuple : gradient_hooks_) {
    size_t slot_id = std::get<0>(tuple);
    size_t rank = std::get<1>(tuple);
    std::function<egr::EagerTensor(const egr::EagerTensor&)>& hook =
        std::get<2>(tuple);

    PADDLE_ENFORCE(slot_id < tensors.size(),
                   paddle::platform::errors::Fatal(
                       "Slot_id from registered hook should be smaller than "
                       "slot size of grad_tensors"));

    PADDLE_ENFORCE(rank < tensors[slot_id].size(),
                   paddle::platform::errors::Fatal(
                       "rank of slot %d from registered hook should be smaller "
                       "than rank size of grad_tensors",
                       slot_id));

    std::vector<egr::EagerTensor>& slot_out = outs[slot_id];
    slot_out.resize(tensors[slot_id].size());
    egr::EagerTensor& out = slot_out[rank];
    if (!out.defined() || !out.initialized()) {
      VLOG(8) << "Run Hook for tensor: " << tensors[slot_id][rank].name();
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
}  // namespace egr

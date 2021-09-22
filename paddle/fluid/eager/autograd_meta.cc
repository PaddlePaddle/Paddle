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

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/eager/nodes/accumulation_node.h"

/**
 * Implementation of AutogradMeta and AbstractAutogradMeta.
**/

namespace egr {

AutogradMeta* EagerUtils::autograd_meta(pt::Tensor* target) {
  auto* p_autograd_meta = target->get_autograd_meta();
  if (!p_autograd_meta) {
    auto p_autograd_meta_ptr = std::make_shared<AutogradMeta>();
    p_autograd_meta = p_autograd_meta_ptr.get();
    target->set_autograd_meta(p_autograd_meta_ptr);
  }
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

AutogradMeta* EagerUtils::unsafe_autograd_meta(const pt::Tensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  PADDLE_ENFORCE(p_autograd_meta,
                 paddle::platform::errors::Fatal(
                     "Null autograd_meta gotten from unsafe_autograd_meta()"));
  return static_cast<AutogradMeta*>(p_autograd_meta);
}
std::vector<AutogradMeta*> EagerUtils::multi_autograd_meta(
    std::vector<pt::Tensor>* targets) {
  std::vector<AutogradMeta*> ret;
  ret.reserve(targets->size());

  // for multi_autograd_meta we can tolerent it has nullptr.
  for (auto& t : (*targets)) {
    auto* p_autograd_meta = autograd_meta(&t);
    ret.push_back(static_cast<AutogradMeta*>(p_autograd_meta));
  }
  return ret;
}

std::pair<size_t, size_t> EagerUtils::OutRankInfo(const pt::Tensor& target) {
  return unsafe_autograd_meta(target)->OutRankInfo();
}

std::shared_ptr<GradNodeBase> EagerUtils::grad_node(const pt::Tensor& target) {
  return unsafe_autograd_meta(target)->GetMutableGradNode();
}

bool EagerUtils::IsLeafTensor(const pt::Tensor& target) {
  std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(target);
  if (std::dynamic_pointer_cast<GradNodeAccumulation>(grad_node)) {
    return true;
  }

  return false;
}

void EagerUtils::PassStopGradient(AutogradMeta** outs, size_t outs_num,
                                  bool generate_grad) {
  for (size_t i = 0; i < outs_num; ++i) {
    if (!outs[i]) {
      // TODO(jiabin): Add Tensor name here when we supported.
      VLOG(0) << "Tensor is NULL";
      continue;
    }
    outs[i]->SetStopGradient(generate_grad);
  }
}

bool EagerUtils::ComputeRequireGrad(AutogradMeta** ins, size_t ins_num,
                                    AutogradMeta** outs, size_t outs_num,
                                    bool trace_backward) {
  if (!trace_backward) return false;

  for (size_t i = 0; i < ins_num; ++i) {
    auto ins_stop_gradient = ins[i]->StopGradient();
    if (!ins_stop_gradient) {
      PassStopGradient(outs, outs_num, ins_stop_gradient);
      return true;
    }
  }
  return false;
}

void EagerUtils::SetHistory(std::vector<AutogradMeta*>* autograd_metas,
                            const std::shared_ptr<GradNodeBase>& grad_node) {
  for (const auto& autograd_meta : *autograd_metas) {
    autograd_meta->SetGradNode(grad_node);
  }
}

void EagerUtils::SetHistory(AutogradMeta* autograd_meta,
                            const std::shared_ptr<GradNodeBase>& grad_node) {
    autograd_meta->SetGradNode(grad_node);
}

pt::Tensor EagerUtils::CreateTensorWithValue(const pt::DDim& ddim,
                                             const pt::Backend& backend,
                                             const pt::DataType& dtype,
                                             const pt::DataLayout& layout,
                                             double value, bool is_leaf) {
  pt::Tensor out = pt::Tensor();
  // Init tensor's autograo_meta
  auto meta = autograd_meta(&out);
  FillConstAPI(value, ddim, backend, dtype, layout, &out);

  if (is_leaf) {
    auto accumulation_node = std::make_shared<GradNodeAccumulation>();
    meta->SetGradNode(accumulation_node);
    meta->SetStopGradient(false);
  }

  return out;
}

void EagerUtils::SetMultiOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                         size_t slot_id) {
  // Set OutRankInfo from 0 to size of targets
  for (size_t i = 0; i < targets->size(); i++) {
    (*targets)[i]->SetSingleOutRankWithSlot(slot_id, i);
  }
}
void EagerUtils::SetOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                         size_t slot_id) {
    SetMultiOutRankWithSlot(targets, slot_id);
}
void EagerUtils::SetOutRankWithSlot(AutogradMeta* target,
                                         size_t slot_id) {
    target->SetSingleOutRankWithSlot(slot_id, 0);
}

}  // namespace egr

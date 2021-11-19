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

#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"

#include "paddle/pten/api/all.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/core/tensor_meta.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/variable.h"

namespace egr {
/* ---- Tensor -> Var ---- */
std::vector<std::shared_ptr<egr::EagerTensor>> SyncToVars(
    const egr::EagerTensor& tensor) {
  // TODO(jiabin): No const cast here. We should call SyncToVar in Python_C
  // wrapper
  const_cast<EagerTensor*>(&tensor)->SyncToVar(
      paddle::framework::proto::VarType_Type_LOD_TENSOR);
  return {std::make_shared<EagerTensor>(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> SyncToVars(
    const std::vector<egr::EagerTensor>& tensors) {
  // TODO(jiabin): No const cast here. We should call SyncToVar in Python_C
  // wrapper
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    const_cast<EagerTensor*>(&(tensors[i]))
        ->SyncToVar(paddle::framework::proto::VarType_Type_LOD_TENSOR);
    res.emplace_back(new EagerTensor(tensors[i]));
  }
  return res;
}

/* ---- VarBase -> Tensor ---- */
std::vector<std::shared_ptr<egr::EagerTensor>> SyncToTensors(
    const egr::EagerTensor& tensor) {
  // TODO(jiabin): No const cast here. We should call SyncToTensor in Python_C
  // wrapper
  const_cast<EagerTensor*>(&tensor)->SyncToTensor();
  return {std::make_shared<EagerTensor>(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> SyncToTensors(
    const std::vector<egr::EagerTensor>& tensors) {
  // TODO(jiabin): No const cast here. We should call SyncToTensor in Python_C
  // wrapper
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    const_cast<EagerTensor*>(&(tensors[i]))->SyncToTensor();
    res.emplace_back(new EagerTensor(tensors[i]));
  }
  return res;
}

std::vector<std::shared_ptr<EagerTensor>> ConstructDuplicableOutput(
    const size_t num) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(
        new EagerTensor(Controller::Instance().GenerateUniqueName()));
  }
  return res;
}

std::vector<egr::EagerTensor> GetOutputs(
    const std::vector<std::shared_ptr<EagerTensor>>& outs) {
  std::vector<egr::EagerTensor> res;
  res.reserve(outs.size());
  for (const auto& out : outs) {
    PADDLE_ENFORCE_NOT_NULL(out.get(),
                            "Eager Tensor %s is null and cannot be copied.",
                            out->name());
    res.emplace_back((*(out.get())));
  }
  return res;
}

egr::EagerTensor GetOutput(const std::shared_ptr<EagerTensor>& out) {
  PADDLE_ENFORCE_NOT_NULL(
      out.get(), "Eager Tensor %s is null and cannot be copied.", out->name());
  return EagerTensor((*(out.get())));
}

/**
 * Implementation of Eager Utils.
**/

AutogradMeta* EagerUtils::autograd_meta(egr::EagerTensor* target) {
  auto* p_autograd_meta = target->get_autograd_meta();
  if (!p_autograd_meta) {
    auto p_autograd_meta_ptr = std::make_shared<AutogradMeta>();
    p_autograd_meta = p_autograd_meta_ptr.get();
    target->set_autograd_meta(p_autograd_meta_ptr);
  }
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

AutogradMeta* EagerUtils::unsafe_autograd_meta(const egr::EagerTensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  PADDLE_ENFORCE(p_autograd_meta,
                 paddle::platform::errors::Fatal(
                     "Null autograd_meta gotten from unsafe_autograd_meta()"));
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::vector<AutogradMeta*> EagerUtils::unsafe_autograd_meta(
    std::vector<egr::EagerTensor>* targets) {
  std::vector<AutogradMeta*> metas;
  for (const egr::EagerTensor& t : *targets) {
    metas.push_back(unsafe_autograd_meta(t));
  }
  return metas;
}

std::vector<AutogradMeta*> EagerUtils::multi_autograd_meta(
    std::vector<egr::EagerTensor>* targets) {
  std::vector<AutogradMeta*> ret;
  ret.reserve(targets->size());

  // for multi_autograd_meta we can tolerent it has nullptr.
  for (auto& t : (*targets)) {
    auto* p_autograd_meta = autograd_meta(&t);
    ret.push_back(static_cast<AutogradMeta*>(p_autograd_meta));
  }
  return ret;
}

std::pair<size_t, size_t> EagerUtils::OutRankInfo(
    const egr::EagerTensor& target) {
  return unsafe_autograd_meta(target)->OutRankInfo();
}

std::shared_ptr<GradNodeBase> EagerUtils::grad_node(
    const egr::EagerTensor& target) {
  return unsafe_autograd_meta(target)->GetMutableGradNode();
}

bool EagerUtils::IsLeafTensor(const egr::EagerTensor& target) {
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
      EagerUtils::PassStopGradient(outs, outs_num, ins_stop_gradient);
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

egr::EagerTensor EagerUtils::CreateTensorWithValue(
    const pten::DDim& ddim, const paddle::platform::Place& place,
    const pten::DataType& dtype, const pten::DataLayout& layout, float value,
    bool is_leaf) {
  paddle::experimental::Tensor tensor = paddle::experimental::full(
      paddle::framework::vectorize(ddim), paddle::experimental::Scalar(value),
      dtype, pten::TransToPtenBackend(place), layout);

  egr::EagerTensor out = egr::EagerTensor();
  out.set_tensor(std::make_shared<paddle::experimental::Tensor>(tensor));
  auto meta = autograd_meta(&out);

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
void EagerUtils::SetOutRankWithSlot(AutogradMeta* target, size_t slot_id) {
  target->SetSingleOutRankWithSlot(slot_id, 0);
}

}  // namespace egr

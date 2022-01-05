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
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/tensor_wrapper.h"

#include "paddle/pten/api/all.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/core/tensor_meta.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/variable.h"

PADDLE_DEFINE_EXPORTED_bool(retain_grad_for_all_tensor, true,
                            "retain grad for all tensor");

namespace egr {
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
    const std::vector<egr::EagerTensor>& targets) {
  std::vector<AutogradMeta*> metas;
  metas.reserve(targets.size());
  for (const egr::EagerTensor& t : targets) {
    metas.emplace_back(unsafe_autograd_meta(t));
  }
  return metas;
}

AutogradMeta* EagerUtils::nullable_autograd_meta(
    const egr::EagerTensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  if (!p_autograd_meta) return nullptr;

  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::vector<AutogradMeta*> EagerUtils::nullable_autograd_meta(
    const std::vector<egr::EagerTensor>& targets) {
  std::vector<AutogradMeta*> metas;
  metas.reserve(targets.size());
  for (const egr::EagerTensor& t : targets) {
    metas.emplace_back(nullable_autograd_meta(t));
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

void EagerUtils::SetOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                    size_t slot_id) {
  // Set OutRankInfo from 0 to size of targets
  for (size_t i = 0; i < targets->size(); i++) {
    (*targets)[i]->SetSingleOutRankWithSlot(slot_id, i);
  }
}
void EagerUtils::SetOutRankWithSlot(AutogradMeta* target, size_t slot_id) {
  target->SetSingleOutRankWithSlot(slot_id, 0);
}

/* ---- Tensor -> Var ---- */
std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToVars(
    const egr::EagerTensor& tensor) {
  // TODO(jiabin): No const cast here. We should call SyncToVar in Python_C
  // wrapper
  const_cast<EagerTensor*>(&tensor)->SyncToVar(
      paddle::framework::proto::VarType_Type_LOD_TENSOR);
  return {std::make_shared<EagerTensor>(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToVars(
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

static std::shared_ptr<egr::EagerTensor> TrySyncToVar(
    egr::EagerTensor* tensor) {
  if (tensor->initialized() || tensor->Var().IsInitialized()) {
    tensor->SyncToVar(paddle::framework::proto::VarType_Type_LOD_TENSOR);
  }
  return std::shared_ptr<egr::EagerTensor>(tensor,
                                           [&](egr::EagerTensor* ptr) {});
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::TrySyncToVars(
    egr::EagerTensor* tensor) {
  return {TrySyncToVar(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::TrySyncToVars(
    std::vector<egr::EagerTensor>* tensors) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors->size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(TrySyncToVar(&(*tensors)[i]));
  }
  return res;
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::TrySyncToVars(
    const std::vector<egr::EagerTensor*>& tensors) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(TrySyncToVar(tensors[i]));
  }
  return res;
}

/* ---- VarBase -> Tensor ---- */
std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToTensors(
    const egr::EagerTensor& tensor) {
  // TODO(jiabin): No const cast here. We should call SyncToTensor in Python_C
  // wrapper
  const_cast<EagerTensor*>(&tensor)->SyncToTensor();
  return {std::make_shared<EagerTensor>(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToTensors(
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

std::vector<std::shared_ptr<EagerTensor>> EagerUtils::ConstructDuplicableOutput(
    const size_t num) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(
        new EagerTensor(egr::Controller::Instance().GenerateUniqueName()));
  }
  return res;
}

std::vector<egr::EagerTensor> EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerTensor>>& outs) {
  std::vector<egr::EagerTensor> res;
  res.reserve(outs.size());
  for (const auto& out : outs) {
    PADDLE_ENFORCE_NOT_NULL(
        out.get(), paddle::platform::errors::Fatal(
                       "Eager Tensor %s is null and cannot be copied. "
                       "We are tring to Get Output tensor from its "
                       "shared_ptr, this error may indicate some outputs "
                       "are nullptr",
                       out->name()));
    res.emplace_back((*(out.get())));
  }
  return res;
}

egr::EagerTensor EagerUtils::GetOutput(
    const std::shared_ptr<EagerTensor>& out) {
  PADDLE_ENFORCE_NOT_NULL(
      out.get(), paddle::platform::errors::Fatal(
                     "Eager Tensor %s is null and cannot be copied. We "
                     "are tring to Get Output tensor from its shared_ptr, "
                     "this error may indicate output is nullptr",
                     out->name()));
  return EagerTensor((*(out.get())));
}

EagerTensor EagerUtils::RecoverTensorWrapper(
    TensorWrapper* tw, const std::shared_ptr<GradNodeBase>& grad_node) {
  return tw->recover(grad_node);
}

std::vector<EagerTensor> EagerUtils::RecoverTensorWrapper(
    std::vector<TensorWrapper>* tw,
    const std::shared_ptr<GradNodeBase>& grad_node) {
  std::vector<EagerTensor> ret;
  for (auto& t : *tw) {
    ret.emplace_back(t.recover(grad_node));
  }
  return ret;
}

void EagerUtils::CheckAndRetainGrad(const egr::EagerTensor& tensor) {
  VLOG(6) << "Check RetainGradForTensor: " << tensor.name();
  if (FLAGS_retain_grad_for_all_tensor) {
    VLOG(6) << "RetainGradForTensor: " << tensor.name();
    egr::egr_utils_api::RetainGradForTensor(tensor);
  }
}

void EagerUtils::CheckAndRetainGrad(
    const std::vector<egr::EagerTensor>& tensors) {
  if (FLAGS_retain_grad_for_all_tensor) {
    for (auto& tensor : tensors) {
      VLOG(6) << "RetainGradForTensor: " << tensor.name();
      egr::egr_utils_api::RetainGradForTensor(tensor);
    }
  }
}

}  // namespace egr

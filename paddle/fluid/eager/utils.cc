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

AutogradMeta* EagerUtils::autograd_meta(paddle::experimental::Tensor* target) {
  auto* p_autograd_meta = target->get_autograd_meta();
  if (!p_autograd_meta) {
    auto p_autograd_meta_ptr = std::make_shared<AutogradMeta>();
    p_autograd_meta = p_autograd_meta_ptr.get();
    target->set_autograd_meta(p_autograd_meta_ptr);
  }
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

AutogradMeta* EagerUtils::unsafe_autograd_meta(
    const paddle::experimental::Tensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  PADDLE_ENFORCE(p_autograd_meta,
                 paddle::platform::errors::Fatal(
                     "Null autograd_meta gotten from unsafe_autograd_meta()"));
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::vector<AutogradMeta*> EagerUtils::unsafe_autograd_meta(
    const std::vector<paddle::experimental::Tensor>& targets) {
  std::vector<AutogradMeta*> metas;
  metas.reserve(targets.size());
  for (const paddle::experimental::Tensor& t : targets) {
    metas.emplace_back(unsafe_autograd_meta(t));
  }
  return metas;
}

AutogradMeta* EagerUtils::nullable_autograd_meta(
    const paddle::experimental::Tensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  if (!p_autograd_meta) return nullptr;

  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::vector<AutogradMeta*> EagerUtils::nullable_autograd_meta(
    const std::vector<paddle::experimental::Tensor>& targets) {
  std::vector<AutogradMeta*> metas;
  metas.reserve(targets.size());
  for (const paddle::experimental::Tensor& t : targets) {
    metas.emplace_back(nullable_autograd_meta(t));
  }
  return metas;
}

std::vector<AutogradMeta*> EagerUtils::autograd_meta(
    std::vector<paddle::experimental::Tensor>* targets) {
  std::vector<AutogradMeta*> ret;
  ret.reserve(targets->size());

  // for autograd_meta we can tolerent it has nullptr.
  for (size_t i = 0; i < targets->size(); i++) {
    auto* p_autograd_meta = autograd_meta(&((*targets)[i]));
    ret.emplace_back(p_autograd_meta);
  }
  return ret;
}

std::pair<size_t, size_t> EagerUtils::OutRankInfo(
    const paddle::experimental::Tensor& target) {
  return unsafe_autograd_meta(target)->OutRankInfo();
}

std::shared_ptr<GradNodeBase> EagerUtils::grad_node(
    const paddle::experimental::Tensor& target) {
  auto* meta = nullable_autograd_meta(target);
  if (meta) {
    return meta->GetMutableGradNode();
  } else {
    return nullptr;
  }
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

std::shared_ptr<egr::EagerTensor> EagerUtils::TrySyncToVar(
    const paddle::experimental::Tensor& tensor) {
  return std::make_shared<egr::EagerTensor>(tensor);
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::TrySyncToVars(
    const paddle::experimental::Tensor& tensor) {
  return {TrySyncToVar(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::TrySyncToVars(
    paddle::experimental::Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      paddle::platform::errors::Fatal(
          "Should Not Pass Empty tensor pointer in, since only output can "
          "reach this, please check output value and make sure it's not null"));
  return {TrySyncToVar(*tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::TrySyncToVars(
    const std::vector<paddle::experimental::Tensor*>& tensors) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    auto* tensor = tensors[i];
    PADDLE_ENFORCE_NOT_NULL(
        tensor, paddle::platform::errors::Fatal(
                    "Tensor is null and cannot be copied. "
                    "We are tring to TrySyncToVars tensor from its "
                    "shared_ptr, this error may indicate some outputs "
                    "are nullptr"));
    res.emplace_back(TrySyncToVar(*tensor));
  }
  return res;
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::TrySyncToVars(
    const std::vector<paddle::experimental::Tensor>& tensors) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(TrySyncToVar(tensors[i]));
  }
  return res;
}

std::vector<std::shared_ptr<EagerTensor>> EagerUtils::CreateVars(
    const size_t num) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(
        new EagerTensor(egr::Controller::Instance().GenerateUniqueName()));
  }
  return res;
}

std::vector<paddle::experimental::Tensor> EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerTensor>>& outs) {
  std::vector<paddle::experimental::Tensor> res;
  res.reserve(outs.size());
  for (const auto& out : outs) {
    PADDLE_ENFORCE_NOT_NULL(
        out.get(), paddle::platform::errors::Fatal(
                       "Eager Tensor %s is null and cannot be copied. "
                       "We are tring to Get Output tensor from its "
                       "shared_ptr, this error may indicate some outputs "
                       "are nullptr",
                       out->name()));
    res.emplace_back(out->GetTensorBase(), out->name());
  }
  return res;
}

paddle::experimental::Tensor EagerUtils::GetOutput(
    const std::shared_ptr<EagerTensor>& out) {
  PADDLE_ENFORCE_NOT_NULL(
      out.get(), paddle::platform::errors::Fatal(
                     "Eager Tensor %s is null and cannot be copied. We "
                     "are tring to Get Output tensor from its shared_ptr, "
                     "this error may indicate output is nullptr",
                     out->name()));
  return paddle::experimental::Tensor(out->GetTensorBase(), out->name());
}

void EagerUtils::OverwriteOutputs(const std::shared_ptr<EagerTensor>& out,
                                  paddle::experimental::Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor, paddle::platform::errors::Fatal(
                  "Tensor is null and cannot be copied. "
                  "We are tring to OverwriteOutput from its "
                  "shared_ptr, this error may indicate some outputs "
                  "are nullptr"));
  tensor->set_impl(out->GetTensorBase());
}

void EagerUtils::OverwriteOutputs(
    const std::vector<std::shared_ptr<EagerTensor>>& outs,
    const std::vector<paddle::experimental::Tensor*>& tensors) {
  PADDLE_ENFORCE_EQ(
      outs.size(), tensors.size(),
      paddle::platform::errors::Fatal(
          "We are tring to OverwriteOutputs which passed in and it expected "
          "elements num of outs and origin outputs are equal, but we got outs "
          "size of: %d, and tensors passed in size is: %d",
          outs.size(), tensors.size()));
  for (size_t i = 0; i < outs.size(); i++) {
    OverwriteOutputs(outs[i], tensors[i]);
  }
}

void EagerUtils::OverwriteOutputs(const paddle::experimental::Tensor& out,
                                  paddle::experimental::Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor, paddle::platform::errors::Fatal(
                  "Tensor is null and cannot be copied. "
                  "We are tring to OverwriteOutput from its "
                  "shared_ptr, this error may indicate some outputs "
                  "are nullptr"));
  *tensor = out;
}
void EagerUtils::OverwriteOutputs(
    const std::vector<paddle::experimental::Tensor>& outs,
    const std::vector<paddle::experimental::Tensor*>& tensors) {
  for (size_t i = 0; i < outs.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        tensors[i], paddle::platform::errors::Fatal(
                        "Tensor is null and cannot be copied. "
                        "We are tring to OverwriteOutput from its "
                        "shared_ptr, this error may indicate some outputs "
                        "are nullptr"));
    *tensors[i] = outs[i];
  }
}

paddle::experimental::Tensor EagerUtils::RecoverTensorWrapper(
    TensorWrapper* tw, const std::shared_ptr<GradNodeBase>& grad_node) {
  return tw->recover(grad_node);
}

std::vector<paddle::experimental::Tensor> EagerUtils::RecoverTensorWrapper(
    std::vector<TensorWrapper>* tw,
    const std::shared_ptr<GradNodeBase>& grad_node) {
  std::vector<paddle::experimental::Tensor> ret;
  for (auto& t : *tw) {
    ret.emplace_back(t.recover(grad_node));
  }
  return ret;
}

void EagerUtils::CheckAndRetainGrad(
    const paddle::experimental::Tensor& tensor) {
  VLOG(6) << "Check RetainGradForTensor: " << tensor.name();
  if (FLAGS_retain_grad_for_all_tensor) {
    VLOG(6) << "RetainGradForTensor: " << tensor.name();
    egr::egr_utils_api::RetainGradForTensor(tensor);
  }
}

void EagerUtils::CheckAndRetainGrad(
    const std::vector<paddle::experimental::Tensor>& tensors) {
  if (FLAGS_retain_grad_for_all_tensor) {
    for (auto& tensor : tensors) {
      VLOG(6) << "RetainGradForTensor: " << tensor.name();
      egr::egr_utils_api::RetainGradForTensor(tensor);
    }
  }
}

}  // namespace egr

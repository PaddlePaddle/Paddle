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
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/tensor_wrapper.h"

#include "paddle/phi/api/all.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_meta.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/phi_utils.h"
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

paddle::experimental::Tensor* EagerUtils::mutable_grad(
    const paddle::experimental::Tensor& target) {
  auto* meta = nullable_autograd_meta(target);
  if (meta) {
    return meta->MutableGrad();
  } else {
    return nullptr;
  }
}

void EagerUtils::SetHistory(std::vector<AutogradMeta*>* autograd_metas,
                            const std::shared_ptr<GradNodeBase>& grad_node) {
  for (const auto& autograd_meta : *autograd_metas) {
    if (autograd_meta->GradNode()) {
      VLOG(7) << "Should not set grad node twice, original node is:"
              << autograd_meta->GradNode()->name()
              << "current is: " << grad_node->name();
    }
    autograd_meta->SetGradNode(grad_node);
  }
}

void EagerUtils::SetHistory(AutogradMeta* autograd_meta,
                            const std::shared_ptr<GradNodeBase>& grad_node) {
  if (autograd_meta->GradNode()) {
    VLOG(7) << "Should not set grad node twice, original node is:"
            << autograd_meta->GradNode()->name()
            << "current is: " << grad_node->name();
  }
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

std::shared_ptr<egr::EagerVariable> EagerUtils::TrySyncToVar(
    const paddle::experimental::Tensor& tensor) {
  return std::make_shared<egr::EagerVariable>(tensor);
}

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    const paddle::experimental::Tensor& tensor) {
  return {TrySyncToVar(tensor)};
}

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    paddle::experimental::Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      paddle::platform::errors::Fatal(
          "Should Not Pass Empty tensor pointer in, since only output can "
          "reach this, please check output value and make sure it's not null"));
  return {TrySyncToVar(*tensor)};
}

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    const std::vector<paddle::experimental::Tensor*>& tensors) {
  std::vector<std::shared_ptr<EagerVariable>> res;
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

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    const std::vector<paddle::experimental::Tensor>& tensors) {
  std::vector<std::shared_ptr<EagerVariable>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(TrySyncToVar(tensors[i]));
  }
  return res;
}

std::vector<std::shared_ptr<EagerVariable>> EagerUtils::CreateVars(
    const size_t num) {
  std::vector<std::shared_ptr<EagerVariable>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(
        new EagerVariable(egr::Controller::Instance().GenerateUniqueName()));
  }
  return res;
}

void EagerUtils::ModifyInplaceInput(
    const std::shared_ptr<EagerVariable>& inplace_variable,
    paddle::experimental::Tensor* inplace_tensor) {
  // Only modify the meta information of the inplace tensor, because
  // EagerVariable cannot modify Tensor's meta information after inplace
  // op (such as ``reshape``) is executed.
  PADDLE_ENFORCE_NOT_NULL(inplace_tensor,
                          paddle::platform::errors::Fatal(
                              "Inplace Tensor is null and cannot be modified. "
                              "We are tring to Modify Inplace Input from its "
                              "shared_ptr, this error may indicate the inplace "
                              " input is nullptr"));
  if (phi::DenseTensor::classof(inplace_variable->GetTensorBase().get())) {
    phi::DenseTensor* variable_dense_tensor =
        static_cast<phi::DenseTensor*>(inplace_variable->GetTensorBase().get());
    phi::DenseTensor* tensor_dense_tensor =
        static_cast<phi::DenseTensor*>(inplace_tensor->impl().get());
    tensor_dense_tensor->set_meta(variable_dense_tensor->meta());
  }
}

std::vector<paddle::experimental::Tensor> EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerVariable>>& outs) {
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
    const std::shared_ptr<EagerVariable>& out) {
  PADDLE_ENFORCE_NOT_NULL(
      out.get(), paddle::platform::errors::Fatal(
                     "Eager Tensor %s is null and cannot be copied. We "
                     "are tring to Get Output tensor from its shared_ptr, "
                     "this error may indicate output is nullptr",
                     out->name()));
  return paddle::experimental::Tensor(out->GetTensorBase(), out->name());
}

void EagerUtils::GetOutput(const std::shared_ptr<EagerVariable>& out,
                           paddle::experimental::Tensor* out_var) {
  PADDLE_ENFORCE_NOT_NULL(
      out_var, paddle::platform::errors::Fatal(
                   "Tensor is null and cannot be copied. "
                   "We are tring to OverwriteOutput from its "
                   "shared_ptr, this error may indicate some outputs "
                   "are nullptr"));
  out_var->set_impl(out->GetTensorBase());
  out_var->set_name(out->name());
}

void EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerVariable>>& outs,
    std::vector<paddle::experimental::Tensor>* result) {
  for (size_t i = 0; i < outs.size(); i++) {
    result->emplace_back(outs[i]->GetTensorBase());
  }
}

void EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerVariable>>& outs,
    const std::vector<paddle::experimental::Tensor*>& out_var) {
  for (size_t i = 0; i < outs.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        out_var[i], paddle::platform::errors::Fatal(
                        "Tensor is null and cannot be copied. "
                        "We are tring to OverwriteOutput from its "
                        "shared_ptr, this error may indicate some outputs "
                        "are nullptr"));
    out_var[i]->set_impl(outs[i]->GetTensorBase());
  }
}

void EagerUtils::GetOutputs(const std::shared_ptr<EagerVariable>& out,
                            std::vector<paddle::experimental::Tensor>* result) {
  result->emplace_back(out->GetTensorBase());
}

void EagerUtils::GetOutputs(
    const std::shared_ptr<EagerVariable>& out,
    const std::vector<paddle::experimental::Tensor*>& out_var) {
  PADDLE_ENFORCE_NOT_NULL(
      out_var[0], paddle::platform::errors::Fatal(
                      "Tensor is null and cannot be copied. "
                      "We are tring to OverwriteOutput from its "
                      "shared_ptr, this error may indicate some outputs "
                      "are nullptr"));
  out_var[0]->set_impl(out->GetTensorBase());
}

void EagerUtils::Output2Result(
    const std::vector<paddle::experimental::Tensor*>& out_var,
    std::vector<paddle::experimental::Tensor>* result) {
  result->reserve(out_var.size());
  for (size_t i = 0; i < out_var.size(); i++) {
    result->emplace_back(*out_var[i]);
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

std::shared_ptr<egr::GradNodeBase> EagerUtils::GetGradAccumulationNode(
    const paddle::experimental::Tensor& tensor) {
  auto* autograd_ptr = nullable_autograd_meta(tensor);
  if (!autograd_ptr) {
    return nullptr;
  }
  auto node_ptr = autograd_ptr->GetMutableGradNode();
  if (node_ptr && node_ptr.get()) {
    if (!autograd_ptr->StopGradient()) {
      auto accumulation_ptr =
          std::dynamic_pointer_cast<GradNodeAccumulation>(node_ptr);
      if (accumulation_ptr) {
        return accumulation_ptr;
      } else {
        // Current GradNode is not a egr::GradNodeAccumulation
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "GetGradAccumulationNode should only be called on leaf tensor, but "
            "target tensor: %s has GradNode which is not a "
            "GradNodeAccumulation, and this should not happend unless target "
            "tensor is modified by some ops and calling set history for it.",
            tensor.name()));
      }
    } else {
      // Current Tensor does not have grad since it's stop_gradient is true;
      return nullptr;
    }
  } else {
    if (!autograd_ptr->StopGradient()) {
      VLOG(6) << "Add GradNodeAccumulation for tensor: " << tensor.name();
      autograd_ptr->SetGradNode(
          std::make_shared<egr::GradNodeAccumulation>(autograd_ptr));
      return autograd_ptr->GetMutableGradNode();
    } else {
      return nullptr;
    }
  }
}

void EagerUtils::FillZeroForEmptyGradInputs(
    std::vector<std::vector<paddle::experimental::Tensor>>* in_grads,
    const std::vector<std::vector<GradSlotMeta>>& grad_in_metas) {
  for (size_t i = 0; i < in_grads->size(); i++) {
    for (size_t j = 0; j < (*in_grads)[i].size(); j++) {
      paddle::experimental::Tensor& grad = (*in_grads)[i][j];
      if (!grad.is_initialized()) {
        const GradSlotMeta& grad_in_meta = grad_in_metas[i][j];
        PADDLE_ENFORCE(
            grad_in_meta.HasTensorMeta(),
            paddle::platform::errors::Fatal(
                "Unable to fill empty grad inputs due to empty GradSlotMeta"));

        const auto& tensor_meta = grad_in_meta.GetTensorMeta();
        phi::Place place = grad_in_meta.GetPlace();

        auto tensor_with_zero = paddle::experimental::full(
            phi::vectorize(tensor_meta.dims), 0.0, tensor_meta.dtype, place);
        grad.set_impl(tensor_with_zero.impl());
      }
    }
  }
}

}  // namespace egr

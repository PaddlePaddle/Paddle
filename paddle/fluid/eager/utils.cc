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

namespace egr {
/**
 * Implementation of Eager Utils.
 **/

AutogradMeta* EagerUtils::autograd_meta(paddle::Tensor* target) {
  auto* p_autograd_meta = target->get_autograd_meta();
  if (!p_autograd_meta) {
    auto p_autograd_meta_ptr = std::make_shared<AutogradMeta>();
    p_autograd_meta = p_autograd_meta_ptr.get();
    target->set_autograd_meta(p_autograd_meta_ptr);
  }
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

AutogradMeta* EagerUtils::unsafe_autograd_meta(const paddle::Tensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  PADDLE_ENFORCE(p_autograd_meta,
                 paddle::platform::errors::Fatal(
                     "Null autograd_meta gotten from unsafe_autograd_meta()"));
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::vector<AutogradMeta*> EagerUtils::unsafe_autograd_meta(
    const std::vector<paddle::Tensor>& targets) {
  std::vector<AutogradMeta*> metas;
  metas.reserve(targets.size());
  for (const paddle::Tensor& t : targets) {
    metas.emplace_back(unsafe_autograd_meta(t));
  }
  return metas;
}

AutogradMeta* EagerUtils::nullable_autograd_meta(const paddle::Tensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  if (!p_autograd_meta) return nullptr;

  return static_cast<AutogradMeta*>(p_autograd_meta);
}

AutogradMeta* EagerUtils::nullable_autograd_meta(
    const paddle::optional<paddle::Tensor>& target) {
  if (target.get_ptr() != nullptr) {
    return EagerUtils::nullable_autograd_meta(*(target.get_ptr()));
  }
  return nullptr;
}

std::vector<AutogradMeta*> EagerUtils::nullable_autograd_meta(
    const std::vector<paddle::Tensor>& targets) {
  std::vector<AutogradMeta*> metas;
  metas.reserve(targets.size());
  for (const paddle::Tensor& t : targets) {
    metas.emplace_back(nullable_autograd_meta(t));
  }
  return metas;
}

std::vector<AutogradMeta*> EagerUtils::nullable_autograd_meta(
    const std::vector<paddle::Tensor*>& targets) {
  std::vector<AutogradMeta*> metas;
  metas.reserve(targets.size());
  for (const paddle::Tensor* t : targets) {
    metas.emplace_back(nullable_autograd_meta(*t));
  }
  return metas;
}

std::vector<AutogradMeta*> EagerUtils::autograd_meta(
    std::vector<paddle::Tensor>* targets) {
  std::vector<AutogradMeta*> ret;
  ret.reserve(targets->size());

  // for autograd_meta we can tolerent it has nullptr.
  for (auto& target : *targets) {
    auto* p_autograd_meta = autograd_meta(&target);
    ret.emplace_back(p_autograd_meta);
  }
  return ret;
}

std::vector<AutogradMeta*> EagerUtils::autograd_meta(
    std::vector<paddle::Tensor*>* targets) {
  std::vector<AutogradMeta*> ret;
  ret.reserve(targets->size());

  // for autograd_meta we can tolerent it has nullptr.
  for (auto& target : *targets) {
    auto* p_autograd_meta = autograd_meta(target);
    ret.emplace_back(p_autograd_meta);
  }
  return ret;
}

std::pair<size_t, size_t> EagerUtils::OutRankInfo(
    const paddle::Tensor& target) {
  return unsafe_autograd_meta(target)->OutRankInfo();
}

std::shared_ptr<GradNodeBase> EagerUtils::grad_node(
    const paddle::Tensor& target) {
  auto* meta = nullable_autograd_meta(target);
  if (meta) {
    return meta->GetMutableGradNode();
  } else {
    return nullptr;
  }
}

paddle::Tensor* EagerUtils::mutable_grad(const paddle::Tensor& target) {
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
              << " current is: " << grad_node->name();
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

bool EagerUtils::IsLeafTensor(const paddle::Tensor& target) {
  std::shared_ptr<GradNodeBase> grad_node_ptr = grad_node(target);
  if (!grad_node_ptr ||
      std::dynamic_pointer_cast<GradNodeAccumulation>(grad_node_ptr)) {
    return true;
  }

  return false;
}

void EagerUtils::CheckInplace(const paddle::Tensor& target,
                              const AutogradMeta* autograd_meta,
                              bool require_any_grad) {
  if (require_any_grad && autograd_meta) {
    PADDLE_ENFORCE_EQ(!autograd_meta->StopGradient() && IsLeafTensor(target),
                      false,
                      paddle::platform::errors::InvalidArgument(
                          "Leaf Var (%s) that doesn't stop gradient "
                          "can't use inplace strategy.",
                          target.name()));
  }
}

std::shared_ptr<egr::EagerVariable> EagerUtils::TrySyncToVar(
    const paddle::Tensor& tensor) {
  return std::make_shared<egr::EagerVariable>(tensor);
}

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    const paddle::Tensor& tensor) {
  return {TrySyncToVar(tensor)};
}

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    paddle::Tensor* tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      paddle::platform::errors::Fatal(
          "Should Not Pass Empty tensor pointer in, since only output can "
          "reach this, please check output value and make sure it's not null"));
  return {TrySyncToVar(*tensor)};
}

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    const std::vector<paddle::Tensor*>& tensors) {
  std::vector<std::shared_ptr<EagerVariable>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    auto* tensor = tensors[i];
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        paddle::platform::errors::Fatal(
            "Tensor is null and cannot be copied. "
            "We are tring to TrySyncToVars tensor from its "
            "shared_ptr, this error may indicate some outputs "
            "are nullptr"));
    res.emplace_back(TrySyncToVar(*tensor));
  }
  return res;
}

std::vector<std::shared_ptr<egr::EagerVariable>> EagerUtils::TrySyncToVars(
    const std::vector<paddle::Tensor>& tensors) {
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

void EagerUtils::HandleViewBetweenInputAndOutput(
    const std::shared_ptr<EagerVariable>& input_var,
    const std::shared_ptr<EagerVariable>& view_output_var) {
  PADDLE_ENFORCE_EQ(
      input_var->Var().IsInitialized(),
      true,
      paddle::platform::errors::InvalidArgument(
          "Tensor %s has not been initialized!", input_var->name()));

  if (phi::DenseTensor::classof(input_var->GetTensorBase().get())) {
    auto input_dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(input_var->GetTensorBase());
    PADDLE_ENFORCE_EQ(
        input_dense_tensor->IsInitialized(),
        true,
        paddle::platform::errors::InvalidArgument(
            "DenseTensor %s has not been initialized!", input_var->name()));

    auto* view_output_tensor =
        view_output_var->MutableVar()->GetMutable<phi::DenseTensor>();
    view_output_tensor->ShareBufferWith(*input_dense_tensor);
    view_output_tensor->ShareInplaceVersionCounterWith(*input_dense_tensor);

    VLOG(3) << "Perform View between Output Var(" << view_output_var->name()
            << ") and Input Var(" << input_var->name()
            << "), share allocation and inplace version.";
  }
}

void EagerUtils::HandleViewBetweenInputAndOutput(
    const paddle::Tensor& input_tensor, paddle::Tensor* view_output_tensor) {
  PADDLE_ENFORCE_EQ(
      input_tensor.initialized(),
      true,
      paddle::platform::errors::InvalidArgument(
          "Tensor %s has not been initialized!", input_tensor.name()));

  if (input_tensor.is_dense_tensor()) {
    auto input_dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(input_tensor.impl());
    if (view_output_tensor->impl() == nullptr) {
      view_output_tensor->set_impl(std::make_shared<phi::DenseTensor>());
    }
    auto view_output_dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(view_output_tensor->impl());
    view_output_dense_tensor->ShareBufferWith(*input_dense_tensor);
    view_output_dense_tensor->ShareInplaceVersionCounterWith(
        *input_dense_tensor);

    VLOG(4) << "Perform View between Output Tensor("
            << view_output_tensor->name() << ") and Input Tensor("
            << input_tensor.name()
            << "), share allocation and inplace version.";
  }
}

std::vector<paddle::Tensor> EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerVariable>>& outs) {
  std::vector<paddle::Tensor> res;
  res.reserve(outs.size());
  for (const auto& out : outs) {
    PADDLE_ENFORCE_NOT_NULL(
        out.get(),
        paddle::platform::errors::Fatal(
            "Eager Tensor %s is null and cannot be copied. "
            "We are tring to Get Output tensor from its "
            "shared_ptr, this error may indicate some outputs "
            "are nullptr",
            out->name()));
    res.emplace_back(out->GetTensorBase(), out->name());
  }
  return res;
}

paddle::Tensor EagerUtils::GetOutput(
    const std::shared_ptr<EagerVariable>& out) {
  PADDLE_ENFORCE_NOT_NULL(
      out.get(),
      paddle::platform::errors::Fatal(
          "Eager Tensor %s is null and cannot be copied. We "
          "are tring to Get Output tensor from its shared_ptr, "
          "this error may indicate output is nullptr",
          out->name()));
  return paddle::Tensor(out->GetTensorBase(), out->name());
}

void EagerUtils::GetOutput(const std::shared_ptr<EagerVariable>& out,
                           paddle::Tensor* out_var) {
  PADDLE_ENFORCE_NOT_NULL(
      out_var,
      paddle::platform::errors::Fatal(
          "Tensor is null and cannot be copied. "
          "We are tring to OverwriteOutput from its "
          "shared_ptr, this error may indicate some outputs "
          "are nullptr"));
  out_var->set_impl(out->GetTensorBase());
  out_var->set_name(out->name());
}

void EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerVariable>>& outs,
    std::vector<paddle::Tensor>* result) {
  for (const auto& out : outs) {
    result->emplace_back(out->GetTensorBase());
  }
}

void EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerVariable>>& outs,
    const std::vector<paddle::Tensor*>& out_var) {
  for (size_t i = 0; i < outs.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        out_var[i],
        paddle::platform::errors::Fatal(
            "Tensor is null and cannot be copied. "
            "We are tring to OverwriteOutput from its "
            "shared_ptr, this error may indicate some outputs "
            "are nullptr"));
    out_var[i]->set_impl(outs[i]->GetTensorBase());
  }
}

void EagerUtils::GetOutputs(const std::shared_ptr<EagerVariable>& out,
                            std::vector<paddle::Tensor>* result) {
  result->emplace_back(out->GetTensorBase());
}

void EagerUtils::GetOutputs(const std::shared_ptr<EagerVariable>& out,
                            const std::vector<paddle::Tensor*>& out_var) {
  PADDLE_ENFORCE_NOT_NULL(
      out_var[0],
      paddle::platform::errors::Fatal(
          "Tensor is null and cannot be copied. "
          "We are tring to OverwriteOutput from its "
          "shared_ptr, this error may indicate some outputs "
          "are nullptr"));
  out_var[0]->set_impl(out->GetTensorBase());
}

void EagerUtils::Output2Result(const std::vector<paddle::Tensor*>& out_var,
                               std::vector<paddle::Tensor>* result) {
  result->reserve(out_var.size());
  for (auto* item : out_var) {
    result->emplace_back(*item);
  }
}

paddle::Tensor EagerUtils::RecoverTensorWrapper(TensorWrapper* tw) {
  return tw->recover();
}

std::vector<paddle::Tensor> EagerUtils::RecoverTensorWrapper(
    std::vector<TensorWrapper>* tw) {
  std::vector<paddle::Tensor> ret;
  for (auto& t : *tw) {
    ret.emplace_back(t.recover());
  }
  return ret;
}

std::shared_ptr<egr::GradNodeBase> EagerUtils::GetGradAccumulationNode(
    const paddle::Tensor& tensor) {
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

void EagerUtils::FillZeroForEmptyOptionalGradInput(
    std::vector<paddle::Tensor>* in_grads,
    const std::vector<GradSlotMeta>& grad_in_metas) {
  for (size_t i = 0; i < in_grads->size(); i++) {
    paddle::Tensor& grad = (*in_grads)[i];
    if (!grad.initialized() && grad_in_metas[i].HasTensorMeta()) {
      auto tensor_with_zero = paddle::experimental::full(
          phi::vectorize(grad_in_metas[i].GetTensorMeta().dims),
          0.0,
          grad_in_metas[i].GetTensorMeta().dtype,
          grad_in_metas[i].GetPlace());
      grad.set_impl(tensor_with_zero.impl());
    }
  }
}

void EagerUtils::FillZeroForEmptyGradInput(paddle::Tensor* in_grad,
                                           const GradSlotMeta& grad_in_meta) {
  if (!in_grad->initialized()) {
    PADDLE_ENFORCE(
        grad_in_meta.HasTensorMeta(),
        paddle::platform::errors::Fatal(
            "Unable to fill empty grad inputs due to empty GradSlotMeta"));
    const auto& tensor_meta = grad_in_meta.GetTensorMeta();
    auto tensor_with_zero =
        paddle::experimental::full(phi::vectorize(tensor_meta.dims),
                                   0.0,
                                   tensor_meta.dtype,
                                   grad_in_meta.GetPlace());
    in_grad->set_impl(tensor_with_zero.impl());
  }
}

void EagerUtils::FillZeroForEmptyOptionalGradInput(
    paddle::Tensor* in_grad, const GradSlotMeta& grad_in_meta) {
  if (!in_grad->initialized() && grad_in_meta.HasTensorMeta()) {
    const auto& tensor_meta = grad_in_meta.GetTensorMeta();
    auto tensor_with_zero =
        paddle::experimental::full(phi::vectorize(tensor_meta.dims),
                                   0.0,
                                   tensor_meta.dtype,
                                   grad_in_meta.GetPlace());
    in_grad->set_impl(tensor_with_zero.impl());
  }
}

void EagerUtils::FillZeroForEmptyGradInput(
    std::vector<paddle::Tensor>* in_grads,
    const std::vector<GradSlotMeta>& grad_in_metas) {
  for (size_t i = 0; i < in_grads->size(); i++) {
    FillZeroForEmptyGradInput(&in_grads->at(i), grad_in_metas[i]);
  }
}

std::string EagerUtils::GradNodeStr(const egr::GradNodeBase& node) {
  if (VLOG_IS_ON(6)) {
    const char* GRAD_NODE_TEMPLATE =
        "BackwardOutMeta: [ %s ], BackwardInMeta: [ %s ]";
    const char* GRAD_SLOT_META_TEMPLATE = " {SlotSize: [%d]: %s} ";
    const char* SLOT_INFO_TEMPLATE =
        "SlotID: %s, StopGradients: %s, Edges[ %s ]";
    auto out_metas = node.OutputMeta();
    auto in_metas = node.InputMeta();
    std::string out_slot_str = "";
    std::string in_slot_str = "";
    const char* EDGE_INFO_TEMPLATE = " { [%d, %d]: [%s, %s] }, ";
    std::string slot_str = "";
    for (size_t i = 0; i < out_metas.size(); i++) {
      std::string edges_str = "";
      std::string sg_str = "";
      for (const GradSlotMeta& meta : out_metas[i]) {
        const egr::Edge& edge = meta.GetEdge();
        if (edge.IsInitialized()) {
          edges_str += paddle::string::Sprintf(EDGE_INFO_TEMPLATE,
                                               edge.GetEdgeRankInfo().first,
                                               edge.GetEdgeRankInfo().second,
                                               edge.GetGradNode(),
                                               edge.GetGradNode()->name());
        } else {
          edges_str += paddle::string::Sprintf("{ NULL Edge }");
        }
        sg_str += meta.IsStopGradient() ? "1, " : "0, ";
      }
      out_slot_str +=
          paddle::string::Sprintf(SLOT_INFO_TEMPLATE, i, sg_str, edges_str);
    }
    std::string out_meta_str = paddle::string::Sprintf(
        GRAD_SLOT_META_TEMPLATE, out_metas.size(), out_slot_str);

    for (size_t i = 0; i < in_metas.size(); i++) {
      std::string edges_str = "";
      std::string sg_str = "";
      for (const GradSlotMeta& meta : in_metas[i]) {
        edges_str += paddle::string::Sprintf("{ NULL Edge }");
        sg_str += meta.IsStopGradient() ? "1, " : "0, ";
      }
      in_slot_str +=
          paddle::string::Sprintf(SLOT_INFO_TEMPLATE, i, sg_str, edges_str);
    }
    std::string in_meta_str =
        paddle::string::Sprintf(GRAD_SLOT_META_TEMPLATE, in_slot_str);
    return paddle::string::Sprintf(
        GRAD_NODE_TEMPLATE, out_meta_str, in_meta_str);
  } else if (VLOG_IS_ON(5)) {
    const char* GRAD_NODE_TEMPLATE =
        "BackwardOutMeta: [ %s ], BackwardInMeta: [ %s ]";
    const char* GRAD_SLOT_META_TEMPLATE = "SlotSize: %d";
    std::string out_meta_str = paddle::string::Sprintf(
        GRAD_SLOT_META_TEMPLATE, node.OutputMeta().size());
    std::string in_meta_str = paddle::string::Sprintf(GRAD_SLOT_META_TEMPLATE,
                                                      node.InputMeta().size());
    return paddle::string::Sprintf(
        GRAD_NODE_TEMPLATE, out_meta_str, in_meta_str);
  } else {
    return "[ Not specified grad node log level. ] ";
  }
}

std::string EagerUtils::GradNodeStr(const paddle::Tensor& t) {
  auto* ad_meta = nullable_autograd_meta(t);
  if (ad_meta && (ad_meta->GetMutableGradNode().get())) {
    return GradNodeStr((*ad_meta->GetMutableGradNode().get()));
  } else {
    return "None";
  }
}

/**
 * Print Input Output (level 0 means least info, level 2 means most info)
 * **/
std::string EagerUtils::TensorStr(const paddle::Tensor& t) {
  std::string tensor_name_str = "";
  if (t.name() == "") {
    tensor_name_str = "None";
  } else {
    tensor_name_str = t.name();
  }
  const char* TENSOR_INFO_TEMPLATE =
      "Type: %s, Dtype: %s, Place: %s, Shape: %s, DistAttr: %s";
  std::string tensor_info_str = "";
  if (t.defined()) {
    if (t.is_dist_tensor()) {
      auto dist_t =
          std::static_pointer_cast<phi::distributed::DistTensor>(t.impl());
      if (t.initialized()) {
        tensor_info_str += paddle::string::Sprintf(
            TENSOR_INFO_TEMPLATE,
            t.impl()->type_info().name(),
            t.dtype(),
            t.place().DebugString(),
            paddle::string::Sprintf(
                "%s, Local Shape: %s", t.dims(), dist_t->local_dims()),
            dist_t->dist_attr());
      } else {
        tensor_info_str += paddle::string::Sprintf(TENSOR_INFO_TEMPLATE,
                                                   t.impl()->type_info().name(),
                                                   "Unknown",
                                                   "Unknown",
                                                   t.dims(),
                                                   dist_t->dist_attr());
      }
    } else {
      if (t.initialized()) {
        tensor_info_str += paddle::string::Sprintf(TENSOR_INFO_TEMPLATE,
                                                   t.impl()->type_info().name(),
                                                   t.dtype(),
                                                   t.place().DebugString(),
                                                   t.dims(),
                                                   "Unknown");
      } else {
        tensor_info_str += paddle::string::Sprintf(TENSOR_INFO_TEMPLATE,
                                                   t.impl()->type_info().name(),
                                                   "Unknown",
                                                   "Unknown",
                                                   "Unknown",
                                                   "Unknown");
      }
    }
  } else {
    tensor_info_str += "Unknown";
  }
  if (VLOG_IS_ON(11)) {
    const char* TENSOR_PRINT_TEMPLATE =
        "{Name: %s, Initialized: %d, Ptr: %d, "
        "TensorInfo: [ %s ], Value:[ %s ], ADInfo:[ %s ]}";
    auto* ad_meta = nullable_autograd_meta(t);
    if (ad_meta && (ad_meta->WeakGrad().lock().get())) {
      std::string ad_info_str = "";
      const char* AD_INFO_TEMPLATE =
          "Grad: [ %s ],  GradNode: [ %s ], StopGradient: [ %d ]";
      ad_info_str += paddle::string::Sprintf(AD_INFO_TEMPLATE,
                                             TensorStr(ad_meta->Grad()),
                                             GradNodeStr(t),
                                             ad_meta->StopGradient());
      auto* data_ptr = dynamic_cast<phi::DenseTensor*>(t.impl().get());
      if (t.is_initialized() && data_ptr) {
        return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                       tensor_name_str,
                                       t.initialized(),
                                       t.impl(),
                                       tensor_info_str,
                                       *data_ptr,
                                       ad_info_str);
      } else {
        return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                       tensor_name_str,
                                       t.initialized(),
                                       t.impl(),
                                       tensor_info_str,
                                       "None",
                                       ad_info_str);
      }
    } else {
      auto* data_ptr = dynamic_cast<phi::DenseTensor*>(t.impl().get());
      if (t.is_initialized() && data_ptr) {
        return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                       tensor_name_str,
                                       t.initialized(),
                                       t.impl(),
                                       tensor_info_str,
                                       *data_ptr,
                                       "None");
      } else {
        return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                       tensor_name_str,
                                       t.initialized(),
                                       t.impl(),
                                       tensor_info_str,
                                       "None",
                                       "None");
      }
    }
  } else if (VLOG_IS_ON(6)) {
    const char* TENSOR_PRINT_TEMPLATE =
        "{Name: %s, Initialized: %d, Ptr: %d,"
        "TensorInfo: [ %s ], ADInfo:[ %s ]}";
    auto* ad_meta = nullable_autograd_meta(t);
    if (ad_meta && (ad_meta->WeakGrad().lock().get())) {
      std::string ad_info_str = "";
      const char* AD_INFO_TEMPLATE =
          "Grad: [ %s ],  GradNode: [ %s ], StopGradient: [ %d ]";
      ad_info_str += paddle::string::Sprintf(AD_INFO_TEMPLATE,
                                             TensorStr(ad_meta->Grad()),
                                             GradNodeStr(t),
                                             ad_meta->StopGradient());
      return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                     tensor_name_str,
                                     t.initialized(),
                                     t.impl(),
                                     tensor_info_str,
                                     ad_info_str);
    } else {
      return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                     tensor_name_str,
                                     t.initialized(),
                                     t.impl(),
                                     tensor_info_str,
                                     "None");
    }
  } else if (VLOG_IS_ON(5)) {
    const char* TENSOR_PRINT_TEMPLATE =
        "{Name: %s, Initialized: %d , Ptr: %d, "
        "TensorInfo: [ %s ]}";
    return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                   tensor_name_str,
                                   t.initialized(),
                                   t.impl(),
                                   tensor_info_str);
  } else if (VLOG_IS_ON(4)) {
    const char* TENSOR_PRINT_TEMPLATE =
        "{ Name: %s, Initialized: %d, Ptr: %d }";
    return paddle::string::Sprintf(
        TENSOR_PRINT_TEMPLATE, tensor_name_str, t.initialized(), t.impl());
  } else {
    return "[ Not specified tensor log level ]";
  }
}

std::string EagerUtils::TensorStr(const std::vector<paddle::Tensor>& tensors) {
  std::string tensors_str = "";
  for (const auto& tensor : tensors) {
    tensors_str += TensorStr(tensor) + ", ";
  }
  return "[ " + tensors_str + " ]";
}

std::string EagerUtils::TensorStr(const paddle::optional<paddle::Tensor>& t) {
  if (!t.is_initialized()) {
    return "{ UnDefinedTensor }";
  } else {
    return TensorStr((*t.get_ptr()));
  }
}

std::string EagerUtils::TensorStr(
    const paddle::optional<std::vector<paddle::Tensor>>& tensors) {
  std::string tensors_str = "";
  if (!tensors.is_initialized()) {
    return "[ UnDefinedTensor List ]";
  } else {
    for (const auto& tensor : (*tensors.get_ptr())) {
      tensors_str += TensorStr(tensor) + ", ";
    }
    return "[ " + tensors_str + " ]";
  }
}
}  // namespace egr

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

#pragma once

#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/phi/api/all.h"

namespace egr {

class TensorWrapper;

/**
 * EagerUtils is utils used to do some static conversion or autograd
 * members access, this class is desinged to be a full static functional
 * utils class
 * **/

template <typename ElementType>
class IterHelper {
  virtual void visit(ElementType element) = 0;

  void visit(std::vector<ElementType>* elements) {
    for (auto element : *elements) visit(element);
  }

  template <typename... Args>
  void apply() {}

 public:
  template <typename T, typename... Args>
  void apply(T&& arg, Args&&... args) {
    visit(std::forward<T>(arg));
    return apply(std::forward<Args>(args)...);
  }
  virtual ~IterHelper() = default;
};

class ComputeRequireGradIter : public IterHelper<AutogradMeta*> {
 public:
  bool RequireGrad() { return require_grad_; }

 private:
  void visit(AutogradMeta* element) override {
    // Dispensable Tensors feeds in nullptr autograd_meta
    if (!element) return;

    bool stop_gradient = element->StopGradient();
    if (!stop_gradient) require_grad_ = true;
  }

  bool require_grad_ = false;
};

class PassStopGradientIter : public IterHelper<AutogradMeta*> {
 public:
  void SetStopGradient(bool stop_gradient) { stop_gradient_ = stop_gradient; }

 private:
  void visit(AutogradMeta* element) override {
    if (!element) {
      // TODO(jiabin): Add Tensor name here when we supported.
      VLOG(2) << "Tensor is NULL";
      return;
    }
    element->SetStopGradient(stop_gradient_);
  }

  bool stop_gradient_ = true;
};

class EagerUtils {
 public:
  /**
   * We have to use autograd_meta and multi_autograd_meta to initialize
   * autograd_meta for tensor, since we can't init it in
   * egr::EagerVariable's
   * constructor (it's abstract class there)
   *
   * **/
  static AutogradMeta* autograd_meta(paddle::experimental::Tensor* target);

  static std::vector<AutogradMeta*> autograd_meta(
      std::vector<paddle::experimental::Tensor>* targets);

  static std::vector<AutogradMeta*> autograd_meta(
      std::vector<paddle::experimental::Tensor*>* targets);

  static std::pair<size_t, size_t> OutRankInfo(
      const paddle::experimental::Tensor& target);

  static std::shared_ptr<GradNodeBase> grad_node(
      const paddle::experimental::Tensor& target);
  static paddle::experimental::Tensor* mutable_grad(
      const paddle::experimental::Tensor& target);

  // Set history is used to set backward info during forward process, it will
  // set forward var's autograd meta's grad node as current backward node.
  static void SetHistory(std::vector<AutogradMeta*>* autograd_metas,
                         const std::shared_ptr<GradNodeBase>& grad_node);
  static void SetHistory(AutogradMeta* autograd_meta,
                         const std::shared_ptr<GradNodeBase>& grad_node);

  // This is used for Set vector of tensors' rank
  static void SetOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                 size_t slot_id);
  static void SetOutRankWithSlot(AutogradMeta* target, size_t slot_id);

  // This method will return an AutogradMeta pointer unsafely.
  static AutogradMeta* nullable_autograd_meta(
      const paddle::experimental::Tensor& target);
  static AutogradMeta* nullable_autograd_meta(
      const paddle::optional<paddle::experimental::Tensor>& target);
  static std::vector<AutogradMeta*> nullable_autograd_meta(
      const std::vector<paddle::experimental::Tensor>& targets);
  static std::vector<AutogradMeta*> nullable_autograd_meta(
      const std::vector<paddle::experimental::Tensor*>& targets);
  static AutogradMeta* unsafe_autograd_meta(
      const paddle::experimental::Tensor& target);
  static std::vector<AutogradMeta*> unsafe_autograd_meta(
      const std::vector<paddle::experimental::Tensor>& targets);

  template <typename T, typename... Args>
  static bool ComputeRequireGrad(T trace_backward, Args&&... args) {
    if (!trace_backward) {
      VLOG(6) << "Do not require grad because trace_backward = false";
      return false;
    }

    auto iter = ComputeRequireGradIter();
    iter.apply(std::forward<Args>(args)...);

    return iter.RequireGrad();
  }

  template <typename T, typename... Args>
  static void PassStopGradient(T stop_gradient, Args&&... args) {
    auto iter = PassStopGradientIter();
    iter.SetStopGradient(stop_gradient);
    iter.apply(std::forward<Args>(args)...);
  }

  static void CheckInplace(const paddle::experimental::Tensor& target,
                           const AutogradMeta* autograd_meta,
                           bool require_any_grad) {
    if (require_any_grad && autograd_meta) {
      PADDLE_ENFORCE_EQ(!autograd_meta->StopGradient() &&
                            egr::egr_utils_api::IsLeafTensor(target),
                        false,
                        paddle::platform::errors::InvalidArgument(
                            "Leaf Var (%s) that doesn't stop gradient "
                            "can't use inplace strategy.",
                            target.name()));
    }
  }

  // View Strategy
  static void HandleViewBetweenInputAndOutput(
      const std::shared_ptr<EagerVariable>& input_var,
      const std::shared_ptr<EagerVariable>& view_output_var);
  static void HandleViewBetweenInputAndOutput(
      const paddle::experimental::Tensor& input_tensor,
      paddle::experimental::Tensor* view_output_tensor);

  // TensorWrapper Utils
  static paddle::experimental::Tensor RecoverTensorWrapper(TensorWrapper* tw);
  static std::vector<paddle::experimental::Tensor> RecoverTensorWrapper(
      std::vector<TensorWrapper>* tw);

  // Intermidate needed remove this once we don't need legacy
  // Inner Method
  static std::shared_ptr<egr::EagerVariable> TrySyncToVar(
      const paddle::experimental::Tensor& tensor);
  // Basic Input
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      const paddle::experimental::Tensor& tensor);
  // Basic Output
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      paddle::experimental::Tensor* tensor);
  // Multi Output
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      const std::vector<paddle::experimental::Tensor*>& tensors);
  // Multi Input
  static std::vector<std::shared_ptr<egr::EagerVariable>> TrySyncToVars(
      const std::vector<paddle::experimental::Tensor>& tensors);
  // Construct empty output
  static std::vector<std::shared_ptr<EagerVariable>> CreateVars(
      const size_t num);
  // Construct Tensor From var
  static std::vector<paddle::experimental::Tensor> GetOutputs(
      const std::vector<std::shared_ptr<EagerVariable>>& outs);
  static paddle::experimental::Tensor GetOutput(
      const std::shared_ptr<EagerVariable>& out);
  static void GetOutput(const std::shared_ptr<EagerVariable>& out,
                        paddle::experimental::Tensor* out_var);
  static void GetOutputs(
      const std::vector<std::shared_ptr<EagerVariable>>& outs,
      std::vector<paddle::experimental::Tensor>* result);
  static void GetOutputs(
      const std::vector<std::shared_ptr<EagerVariable>>& outs,
      const std::vector<paddle::experimental::Tensor*>& out_var);
  static void GetOutputs(const std::shared_ptr<EagerVariable>& out,
                         std::vector<paddle::experimental::Tensor>* result);
  static void GetOutputs(
      const std::shared_ptr<EagerVariable>& out,
      const std::vector<paddle::experimental::Tensor*>& out_var);

  static void Output2Result(
      const std::vector<paddle::experimental::Tensor*>& out_var,
      std::vector<paddle::experimental::Tensor>* result);

  // end Intermidate needed.

  static void CheckAndRetainGrad(const paddle::experimental::Tensor& tensor);
  static void CheckAndRetainGrad(
      const std::vector<paddle::experimental::Tensor>& tensors);
  static void CheckAndRetainGrad(
      const std::vector<paddle::experimental::Tensor*>& tensors);

  static std::shared_ptr<egr::GradNodeBase> GetGradAccumulationNode(
      const paddle::experimental::Tensor& tensor);

  /**
   * Fill Zero
   * **/
  static void FillZeroForEmptyOptionalGradInput(
      std::vector<paddle::experimental::Tensor>* in_grads,
      const std::vector<GradSlotMeta>& grad_in_metas);
  static void FillZeroForEmptyGradInput(paddle::experimental::Tensor* in_grad,
                                        const GradSlotMeta& grad_in_meta);
  static void FillZeroForEmptyOptionalGradInput(
      paddle::experimental::Tensor* in_grad, const GradSlotMeta& grad_in_meta);
  static void FillZeroForEmptyGradInput(
      std::vector<paddle::experimental::Tensor>* in_grads,
      const std::vector<GradSlotMeta>& grad_in_metas);
  /**
   * Print Input Output (level 0 means least info, level 2 means most info)
   * **/
  static const std::string TensorStr(const paddle::experimental::Tensor& t) {
    std::string tensor_name_str = "";
    if (t.name() == "") {
      tensor_name_str = "None";
    } else {
      tensor_name_str = t.name();
    }
    const char* TENSOR_INFO_TEMPLATE =
        "Type: %s, Dtype: %s, Place: %s, Shape: %s";
    std::string tensor_info_str = "";
    if (t.defined()) {
      if (t.initialized()) {
        tensor_info_str += paddle::string::Sprintf(TENSOR_INFO_TEMPLATE,
                                                   t.impl()->type_info().name(),
                                                   t.dtype(),
                                                   t.place().DebugString(),
                                                   t.dims());
      } else {
        tensor_info_str += paddle::string::Sprintf(TENSOR_INFO_TEMPLATE,
                                                   t.impl()->type_info().name(),
                                                   "Unknown",
                                                   "Unknown",
                                                   "Unknown");
      }
    } else {
      tensor_info_str += "Unknown";
    }
    if (VLOG_IS_ON(11)) {
      const char* TENSOR_PRINT_TEMPLATE =
          "{Name: %s, Initialized: %d, Ptr: %d "
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
          "{Name: %s, Initialized: %d, Ptr: %d "
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
          "{Name: %s, Initialized: %d , Ptr: %d "
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

  static const std::string GradNodeStr(const egr::GradNodeBase& node) {
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
      std::string in_meta_str = paddle::string::Sprintf(
          GRAD_SLOT_META_TEMPLATE, node.InputMeta().size());
      return paddle::string::Sprintf(
          GRAD_NODE_TEMPLATE, out_meta_str, in_meta_str);
    } else {
      return "[ Not specified grad node log level. ] ";
    }
  }

  static const std::string GradNodeStr(const paddle::experimental::Tensor& t) {
    auto* ad_meta = nullable_autograd_meta(t);
    if (ad_meta && (ad_meta->GetMutableGradNode().get())) {
      return GradNodeStr((*ad_meta->GetMutableGradNode().get()));
    } else {
      return "None";
    }
  }

  static const std::string TensorStr(
      const std::vector<paddle::experimental::Tensor>& tensors) {
    std::string tensors_str = "";
    for (const auto& tensor : tensors) {
      tensors_str += TensorStr(tensor) + ", ";
    }
    return "[ " + tensors_str + " ]";
  }

  static const std::string TensorStr(
      const paddle::optional<paddle::experimental::Tensor>& t) {
    if (!t.is_initialized()) {
      return "{ UnDefinedTensor }";
    } else {
      return TensorStr((*t.get_ptr()));
    }
  }

  static const std::string TensorStr(
      const paddle::optional<std::vector<paddle::experimental::Tensor>>&
          tensors) {
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
};

}  // namespace egr

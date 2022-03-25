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
    element->WeakSetStopGradient(stop_gradient_);
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
  static std::vector<AutogradMeta*> nullable_autograd_meta(
      const std::vector<paddle::experimental::Tensor>& targets);
  static AutogradMeta* unsafe_autograd_meta(
      const paddle::experimental::Tensor& target);
  static std::vector<AutogradMeta*> unsafe_autograd_meta(
      const std::vector<paddle::experimental::Tensor>& targets);

  template <typename T, typename... Args>
  static bool ComputeRequireGrad(T trace_backward, Args&&... args) {
    if (!trace_backward) return false;

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
                        false, paddle::platform::errors::InvalidArgument(
                                   "Leaf Var (%s) that doesn't stop gradient "
                                   "can't use inplace strategy.",
                                   target.name()));
    }
  }

  // TensorWrapper Utils
  static paddle::experimental::Tensor RecoverTensorWrapper(
      TensorWrapper* tw, const std::shared_ptr<GradNodeBase>& grad_node);
  static std::vector<paddle::experimental::Tensor> RecoverTensorWrapper(
      std::vector<TensorWrapper>* tw,
      const std::shared_ptr<GradNodeBase>& grad_node);

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
  static void ModifyInplaceInput(
      const std::shared_ptr<EagerVariable>& inplace_variable,
      paddle::experimental::Tensor* inplace_tensor);
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

  // end Intermidate needed

  static void CheckAndRetainGrad(const paddle::experimental::Tensor& tensor);
  static void CheckAndRetainGrad(
      const std::vector<paddle::experimental::Tensor>& tensors);
  static std::shared_ptr<egr::GradNodeBase> GetGradAccumulationNode(
      const paddle::experimental::Tensor& tensor);

  /**
    * Fill Zero
    * **/
  static void FillZeroForEmptyGradInputs(
      std::vector<std::vector<paddle::experimental::Tensor>>* out_grads,
      const std::vector<std::vector<GradSlotMeta>>& grad_out_metas);
};

}  // namespace egr

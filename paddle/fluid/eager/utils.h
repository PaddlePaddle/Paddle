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

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"

#include "paddle/pten/api/all.h"

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
   * egr::EagerTensor's
   * constructor (it's abstract class there)
   *
   * **/
  static AutogradMeta* autograd_meta(egr::EagerTensor* target);

  static std::vector<AutogradMeta*> multi_autograd_meta(
      std::vector<egr::EagerTensor>* targets);

  static std::pair<size_t, size_t> OutRankInfo(const egr::EagerTensor& target);

  static std::shared_ptr<GradNodeBase> grad_node(
      const egr::EagerTensor& target);

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
  static AutogradMeta* nullable_autograd_meta(const egr::EagerTensor& target);
  static std::vector<AutogradMeta*> nullable_autograd_meta(
      const std::vector<egr::EagerTensor>& targets);
  static AutogradMeta* unsafe_autograd_meta(const egr::EagerTensor& target);
  static std::vector<AutogradMeta*> unsafe_autograd_meta(
      const std::vector<egr::EagerTensor>& targets);

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

  // TensorWrapper Utils
  static egr::EagerTensor RecoverTensorWrapper(
      egr::TensorWrapper* tw, const std::shared_ptr<GradNodeBase>& grad_node);
  static std::vector<egr::EagerTensor> RecoverTensorWrapper(
      std::vector<egr::TensorWrapper>* tw,
      const std::shared_ptr<GradNodeBase>& grad_node);

  // Intermidate needed remove this once we don't need legacy
  static std::vector<std::shared_ptr<egr::EagerTensor>> TrySyncToVars(
      egr::EagerTensor* tensor);
  static std::vector<std::shared_ptr<egr::EagerTensor>> TrySyncToVars(
      std::vector<egr::EagerTensor>* tensors);
  static std::vector<std::shared_ptr<egr::EagerTensor>> TrySyncToVars(
      const std::vector<egr::EagerTensor*>& tensors);

  static std::vector<std::shared_ptr<egr::EagerTensor>> SyncToVars(
      const egr::EagerTensor& tensor);
  static std::vector<std::shared_ptr<egr::EagerTensor>> SyncToVars(
      const std::vector<egr::EagerTensor>& tensors);
  static std::vector<std::shared_ptr<egr::EagerTensor>> SyncToTensors(
      const egr::EagerTensor& tensor);
  static std::vector<std::shared_ptr<egr::EagerTensor>> SyncToTensors(
      const std::vector<egr::EagerTensor>& tensors);
  static std::vector<std::shared_ptr<EagerTensor>> ConstructDuplicableOutput(
      const size_t num);
  static std::vector<egr::EagerTensor> GetOutputs(
      const std::vector<std::shared_ptr<EagerTensor>>& outs);
  static egr::EagerTensor GetOutput(const std::shared_ptr<EagerTensor>& outs);

  static void CheckAndRetainGrad(const egr::EagerTensor& tensor);
  static void CheckAndRetainGrad(const std::vector<egr::EagerTensor>& tensors);
};

}  // namespace egr

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

#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/framework/variable.h"

namespace egr {

using AbstractAutogradMeta = paddle::experimental::AbstractAutogradMeta;
/**
 *
 * AutogradMeta is what record the backward info for tensor. When we run
 * computation
 * graph eagerly, we can not build a static paddle program like static mode do,
 * so we
 * need a new method to record forward info to trace backward when we finish all
 * forward
 * computation. This require our AutogradMeta class record following main
 * members
 *
 * 1. grad_op:
 * Grad_op indicate the grad operation of the forward op
 *
 * 2. grad:
 * Grad is the gradient of forward Tensor, which should be compute after
 * backward computation
 *
 * NOTE: grad should only be available when current tensor is a leaf tensor, and
 * for non-leaf
 * tensor grad is only available while user set `retain_grad` option as `true`.
 *
 * TODO(jiabin) : support hooks
 * 3. hooks:
 * Hooks are some computation logic which only attached with backward operation,
 * it registered
 * by user and run before accumulator.
 *
 * 4.overrided_stop_gradient_
 * This member is used to finish some auto-prune related work, which indicate
 * user set stop_gradient
 * should overrided the result indicated by framework. All non-parameter
 * tensor's stop_gradient
 * properties should be true. We will pass stop_gradient when we find one who
 * need it.
 *
 * NOTE: AutogradMeta is inherited from AbstractAutogradMeta which is defined
 * in tensor's deps,
 * we did this to avoid additional dependency on Autograd. In eager execution,
 * we will cast
 * AbstractAutogradMeta as AutogradMeta to use it.
 *
 * **/

// No other AutogradMeta class should be derivated from AbstractAutogradMeta.
// It's only used by
class AutogradMeta : public AbstractAutogradMeta {
 public:
  explicit AutogradMeta(const Edge& edge = Edge()) {
    out_rank_ = 0;
    out_slot_id_ = 0;
    grad_node_ = edge.GetMutableGradNode();
  }

  ~AutogradMeta() override = default;

  const egr::EagerTensor& Grad() const { return grad_; }

  egr::EagerTensor* MutableGrad() { return &grad_; }

  void SetGradNode(const std::shared_ptr<GradNodeBase>& grad_node) {
    PADDLE_ENFORCE_NOT_NULL(grad_node.get(),
                            "Should Not set NULL as GradNode pointer!");
    grad_node_ = grad_node;
  }

  std::shared_ptr<GradNodeBase> GetMutableGradNode() const {
    return grad_node_;
  }

  GradNodeBase* GradNode() const { return grad_node_.get(); }

  void SetSingleOutRankWithSlot(size_t slot_id, size_t rank) {
    out_slot_id_ = slot_id;
    out_rank_ = rank;
  }

  std::pair</* slot id */ size_t, /* rank in slot */ size_t> OutRankInfo()
      const {
    return std::make_pair(out_slot_id_, out_rank_);
  }

  bool IsInitialized() { return !grad_node_.get(); }

  bool StopGradient() const { return stop_gradient_ != 0; }

  int NumericStopGradient() const { return stop_gradient_; }

  void SetStopGradient(bool stop_gradient) {
    if (stop_gradient_ == -1) {
      stop_gradient_ = static_cast<int>(stop_gradient);
    } else {
      VLOG(0) << "Ignore Stop gradient conversion for Var: "
              << "Set value is: " << stop_gradient_;
    }
  }

  bool Persistable() const { return persistable_; }

  void SetPersistable(bool persistable) { persistable_ = persistable; }

 private:
  // TODO(jiabin) :Should we use pointer instead of object?
  egr::EagerTensor grad_;

  // GradNodeBase is base class of all grad op which is a
  // wrapper for grad op. This class will make grad op easy
  // to be traced.
  std::shared_ptr<GradNodeBase> grad_node_;

  /**
   * Why we need slot id here?
   * Because in paddle most of our operators inputs and outputs
   * are assemble in form of {"slot name", vector<tensor>}.
   * So its better for us to set a slot id to fit this format. **/
  size_t out_slot_id_;

  // output rank of forward op, this is a vital num, since
  // we are now trying to make our forward output is as same
  // sequence as backward input. In case of tracing backward
  // sequence we need to record output rank in slot here.
  size_t out_rank_;

  // TODO(jiabin) :Support hooks here and store it in AutogradMeta

  // Stop gradient flag to indicate should we compute backward
  int stop_gradient_{-1};

  bool persistable_{false};

  // TODO(jiabin) :Support Quantum here and add cache mechanism as
  // VarCache defined in VarBase
};

/**
 * AugradUtils is utils used to do some static conversion or autograd
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

template <typename T, typename... Args>
bool ComputeRequireGrad(T trace_backward, Args&&... args) {
  if (!trace_backward) return false;

  auto iter = ComputeRequireGradIter();
  iter.apply(std::forward<Args>(args)...);

  return iter.RequireGrad();
}

template <typename T, typename... Args>
void PassStopGradient(T generate_grad, Args&&... args) {
  auto iter = PassStopGradientIter();
  iter.SetStopGradient(generate_grad);
  iter.apply(std::forward<Args>(args)...);
}

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

  static bool ComputeRequireGrad(AutogradMeta** ins, size_t ins_num,
                                 AutogradMeta** outs, size_t outs_num,
                                 bool trace_backward);

  static void PassStopGradient(AutogradMeta** outs, size_t outs_num,
                               bool generate_grad);

  // If and only if the tensor holds an AccumulationNode
  // Then it's treated as a leaf tensor
  static bool IsLeafTensor(const egr::EagerTensor& target);

  // Set history is used to set backward info during forward process, it will
  // set forward var's autograd meta's grad node as current backward node.
  static void SetHistory(std::vector<AutogradMeta*>* autograd_metas,
                         const std::shared_ptr<GradNodeBase>& grad_node);
  static void SetHistory(AutogradMeta* autograd_meta,
                         const std::shared_ptr<GradNodeBase>& grad_node);

  static egr::EagerTensor CreateTensorWithValue(const pten::DDim& ddim,
                                                const pten::Backend& backend,
                                                const pten::DataType& dtype,
                                                const pten::DataLayout& layout,
                                                double value,
                                                bool is_leaf = true);
  // This is used for Set vector of tensors' rank
  static void SetMultiOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                      size_t slot_id);
  static void SetOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                 size_t slot_id);
  static void SetOutRankWithSlot(AutogradMeta* target, size_t slot_id);

  // This method will return an AutogradMeta pointer unsafely.
  static AutogradMeta* unsafe_autograd_meta(const egr::EagerTensor& target);
  static std::vector<AutogradMeta*> unsafe_autograd_meta(
      std::vector<egr::EagerTensor>* targets);
};

}  // namespace egr

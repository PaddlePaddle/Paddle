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
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"

namespace egr {

class GradNodeAccumulation : public GradNodeBase {
 public:
  // Constructor: configure fwd input tensors to grad node
  explicit GradNodeAccumulation(AutogradMeta* meta) : GradNodeBase(1, 1) {
    VLOG(6) << "Construct GradNodeAccumulation";
    if (meta) {
      weak_grad_ = meta->WeakGrad();
    }

    SetDefaultGradInOutMeta();
  }

  ~GradNodeAccumulation() override {
    VLOG(6) << "Destruct GradNodeAccumulation";
  }

  // Functor: perform backward computations
  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      std::vector<std::vector<paddle::experimental::Tensor>>& grads,  // NOLINT
      bool create_graph = false) override;

  void ClearTensorWrappers() override { VLOG(6) << "Do nothing here now"; }

  std::string name() { return "GradNodeAccumulation"; }

  /**
   * Register ReduceHook
   * **/
  void RegisterReduceHook(std::shared_ptr<TensorVoidHook>&& hook);

  /**
   * Apply ReduceHook here
   * **/
  inline bool ReduceHooksRegistered() { return reduce_hooks_.size() != 0; }
  void ApplyReduceHooks();

  std::shared_ptr<GradNodeBase> Copy() const override {
    return std::shared_ptr<GradNodeAccumulation>(
        new GradNodeAccumulation(nullptr));
  }

 private:
  std::weak_ptr<paddle::experimental::Tensor> weak_grad_;

  std::function<paddle::experimental::Tensor(
      const paddle::experimental::Tensor&)>
      retain_grad_hook_;

  std::vector<std::shared_ptr<TensorVoidHook>> reduce_hooks_;
};

}  // namespace egr

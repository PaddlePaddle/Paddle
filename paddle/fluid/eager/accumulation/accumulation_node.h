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
#include "paddle/utils/test_macros.h"

COMMON_DECLARE_int32(call_stack_level);

namespace egr {

class TEST_API GradNodeAccumulation : public GradNodeBase {
 public:
  // Constructor: configure fwd input tensors to grad node
  explicit GradNodeAccumulation(AutogradMeta* meta) : GradNodeBase(1, 1) {
    VLOG(5) << "Construct GradNodeAccumulation";
    if (meta) {
      weak_grad_ = meta->WeakGrad();
    }

    if (FLAGS_call_stack_level == 3) {
      this->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }

    SetDefaultGradInOutMeta();
  }

  ~GradNodeAccumulation() override {
    VLOG(5) << "Destruct GradNodeAccumulation";
  }

  // Functor: perform backward computations
  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override { VLOG(5) << "Do nothing here now"; }

  std::string name() override { return "GradNodeAccumulation"; }

  /**
   * Register ReduceHook
   * **/
  void RegisterReduceHook(std::shared_ptr<VoidHook>&& hook);

  /**
   * Apply ReduceHook here
   * **/
  inline bool ReduceHooksRegistered() { return reduce_hooks_.size() != 0; }
  void ApplyReduceHooks();

  std::shared_ptr<GradNodeBase> Copy() const override {
    return std::shared_ptr<GradNodeAccumulation>(
        new GradNodeAccumulation(nullptr));
  }

  void SetFakeEmpty(bool is_fake_empty) { is_fake_empty_ = is_fake_empty; }

 private:
  // TODO(Jiabin): remove this when we make our clear gradient really cleared;
  bool is_fake_empty_ = {false};
  std::weak_ptr<paddle::Tensor> weak_grad_;
  std::vector<std::shared_ptr<VoidHook>> reduce_hooks_;
  std::function<paddle::Tensor(const paddle::Tensor&)> retain_grad_hook_;
};

}  // namespace egr

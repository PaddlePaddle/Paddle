// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <utility>
#include <vector>

#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/layer.h"

namespace paddle {
namespace imperative {

class GradientAccumulator {
 public:
  explicit GradientAccumulator(std::shared_ptr<VariableWrapper> var) {
    if (var->Var().IsInitialized()) {
      const framework::Tensor* tensor = nullptr;
      if (var->Var().IsType<framework::LoDTensor>()) {
        tensor = &(var->Var().Get<framework::LoDTensor>());
      } else if (var->Var().IsType<framework::SelectedRows>()) {
        tensor = &(var->Var().Get<framework::SelectedRows>()).value();
      } else {
        PADDLE_THROW(platform::errors::PermissionDenied(
            "Only support LoDTensor and SelectedRows for gradient var"));
      }
      if (tensor && tensor->IsInitialized()) {
        VLOG(3) << "Gradient (" << var->Name() << ") has been initialized! It "
                                                  "will accumulate on previous "
                                                  "Gradient!";
        if (!var->HasGradNode()) {
          previous_var_ = var;
          var_ = std::make_shared<VariableWrapper>(var->Name());
          var_->SetType(var->Type());
          var_->SetDataType(var->DataType());
          var_->InnerSetOverridedStopGradient(
              var->InnerOverridedStopGradient());
        }
      }
    }
    if (!HasPreviousVar()) {
      var_ = var;
    }
  }

  virtual void Add(std::shared_ptr<VariableWrapper> var, size_t trace_id,
                   bool unchange_input = false) = 0;

  virtual ~GradientAccumulator() = default;

  inline void IncreaseRefCnt() {
    ++ref_cnt_;
    VLOG(6) << var_->Name() << " Increase ref_cnt_ to " << ref_cnt_;
  }

  inline void IncreaseCurCnt() {
    ++cur_cnt_;
    VLOG(6) << var_->Name() << " Increase cur_cnt to " << cur_cnt_
            << ", ref_cnt " << ref_cnt_;
  }

  inline size_t CurCnt() const { return cur_cnt_; }

  inline size_t RefCnt() const { return ref_cnt_; }

  inline bool AddGradCompleted() const {
    return cur_cnt_ == ref_cnt_ || ref_cnt_ == 1;
  }

  std::shared_ptr<VariableWrapper> Var() { return var_; }

  inline bool HasPreviousVar() const { return !previous_var_.expired(); }

  /* Hook related methods */
  inline bool HasPostHooks() const { return !post_hooks_.expired(); }

  void SetPostHooks(const std::shared_ptr<LeafVarHookPipeline>& hooks) {
    PADDLE_ENFORCE_NOT_NULL(
        hooks, platform::errors::InvalidArgument(
                   "The hook set to GradientAccumulator is nullptr."));

    auto shared_hooks = post_hooks_.lock();
    if (shared_hooks != hooks) {
      PADDLE_ENFORCE_EQ(
          shared_hooks, nullptr,
          platform::errors::PermissionDenied(
              "Cannot set post hooks twice to GradientAccumulator."));
      post_hooks_ = hooks;
    }
  }
  // void CallHooks() {}

  // Sum Gradient with Pre-Graph
  void AccumulateGrad();

  // call backward post hooks, such as reduce hook
  void CallBackwardPostHooks() {
    PADDLE_ENFORCE_NE(
        post_hooks_.expired(), true,
        platform::errors::NotFound(
            "The post hooks of GradientAccumulator for Tensor `%s` expired.",
            var_->Name()));
    auto shared_hooks = post_hooks_.lock();
    for (const auto& hook : shared_hooks->backward_hooks()) {
      VLOG(3) << "call gradient accumulator backward hooks.";
      (*hook)(var_.get());
    }
  }

 protected:
  std::weak_ptr<VariableWrapper> previous_var_;
  std::shared_ptr<VariableWrapper> var_;
  size_t ref_cnt_{0};
  size_t cur_cnt_{0};
  std::weak_ptr<LeafVarHookPipeline> post_hooks_;
};

class EagerGradientAccumulator : public GradientAccumulator {
 public:
  using GradientAccumulator::GradientAccumulator;

  void Add(std::shared_ptr<VariableWrapper> var, size_t trace_id,
           bool unchange_input) override;
};

class SortedGradientAccumulator : public GradientAccumulator {
 public:
  using GradientAccumulator::GradientAccumulator;

  void Add(std::shared_ptr<VariableWrapper> var, size_t trace_id,
           bool unchange_input) override;

 private:
  struct SavedVarInfo {
    SavedVarInfo(std::shared_ptr<VariableWrapper>&& v, size_t id,
                 bool enable_unchange_input)
        : var(std::move(v)),
          trace_id(id),
          unchange_input(enable_unchange_input) {}

    std::shared_ptr<VariableWrapper> var;
    size_t trace_id;
    bool unchange_input;
  };

  std::vector<SavedVarInfo> tmp_grad_vars_;
};

}  // namespace imperative
}  // namespace paddle

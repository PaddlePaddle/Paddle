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
  explicit GradientAccumulator(VariableWrapper* var) {
    // var may be initialized, so Synchronous VariableWrapper with Variable
    if (var && var->Var().IsInitialized()) {
      if (var->Var().IsType<framework::LoDTensor>()) {
        var->SetType(framework::proto::VarType::LOD_TENSOR);
      } else if (var->Var().IsType<framework::SelectedRows>()) {
        var->SetType(framework::proto::VarType::SELECTED_ROWS);
      } else {
        PADDLE_THROW(platform::errors::PermissionDenied(
            "Only support LoDTensor and SelectedRows for gradient var"));
      }
    }

    // inner_var_ record the grad of this auto-grad.
    // Only need to generate inner var for non-empty leaf-tensor.
    if (var->IsLeafGrad() && !var->IsEmpty()) {
      inner_var_ = std::make_shared<VariableWrapper>(var->Name());
      inner_var_->SetType(var->Type());
      inner_var_->SetDataType(var->DataType());
      inner_var_->InnerSetOverridedStopGradient(
          var->InnerOverridedStopGradient());
      VLOG(6) << " Create inner grad var for (" << var->Name()
              << ") to store result of this Graph";
    }

    // TODO(zhouwei): fix Tensor.clear_gradient() bug, remove this hard flag
    var->SetIsEmpty(false);

    // var_ is the final grad, processed by hooks and grad accumulation
    var_ = var;
  }

  // function that Sum Gradient with this Graph
  virtual void SumGrad(std::shared_ptr<VariableWrapper> var, size_t trace_id,
                       bool unchange_input = false) = 0;

  virtual ~GradientAccumulator() = default;

  inline void IncreaseRefCnt() {
    ++ref_cnt_;
    VLOG(6) << var_->Name() << " Increase total count to " << ref_cnt_;
  }

  inline void IncreaseCurCnt() {
    ++cur_cnt_;
    VLOG(6) << var_->Name() << " Increase current count to " << cur_cnt_
            << ", total count: " << ref_cnt_;
  }

  inline size_t CurCnt() const { return cur_cnt_; }

  inline size_t RefCnt() const { return ref_cnt_; }

  inline bool SumGradCompleted() const {
    return cur_cnt_ == ref_cnt_ || ref_cnt_ == 1;
  }

  std::shared_ptr<VariableWrapper>& InnerVar() { return inner_var_; }

  // return the var that will be calculated in this graph
  VariableWrapper* Var() {
    return inner_var_ != nullptr ? inner_var_.get() : var_;
  }

  inline bool HasInnerVar() const { return inner_var_ != nullptr; }

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
  // void CallHooks(){}
  //  ** inner_var_ **

  // function that Sum Gradient with Previous Graph
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
      (*hook)(var_);
    }
  }

 protected:
  VariableWrapper* var_;
  // NOTE: only gradient accumulater of leaf tensor should hold
  // inner_var_, So not hold it by other shared pointer.
  std::shared_ptr<VariableWrapper> inner_var_;
  size_t ref_cnt_{0};
  size_t cur_cnt_{0};
  std::weak_ptr<LeafVarHookPipeline> post_hooks_;
};

class EagerGradientAccumulator : public GradientAccumulator {
 public:
  using GradientAccumulator::GradientAccumulator;

  void SumGrad(std::shared_ptr<VariableWrapper> var, size_t trace_id,
               bool unchange_input) override;
};

class SortedGradientAccumulator : public GradientAccumulator {
 public:
  using GradientAccumulator::GradientAccumulator;

  void SumGrad(std::shared_ptr<VariableWrapper> var, size_t trace_id,
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

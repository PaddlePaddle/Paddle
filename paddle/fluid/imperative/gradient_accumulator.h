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
  explicit GradientAccumulator(VariableWrapper* var) : var_(var) {}

  virtual void Add(std::shared_ptr<VariableWrapper> var, size_t trace_id,
                   bool unchange_input = false) = 0;

  virtual ~GradientAccumulator() = default;

  inline void IncreaseRefCnt() { ++ref_cnt_; }

  inline size_t RefCnt() const { return ref_cnt_; }

  /* Hook related methods */
  inline bool HasReduceHook() const { return !reduce_hook_.expired(); }

  inline void SetReduceHook(
      const std::shared_ptr<LambdaGradAccumulatorPostHook>& hook) {
    if (!hook) {
      reduce_hook_.reset();
      return;
    }

    auto shared_hook = reduce_hook_.lock();
    if (shared_hook != hook) {
      PADDLE_ENFORCE_EQ(
          shared_hook, nullptr,
          platform::errors::PermissionDenied("Cannot set post hooks twice"));
      reduce_hook_ = hook;
    }
  }

  inline void CallReduceHook() {
    auto shared_hook = reduce_hook_.lock();
    (*shared_hook)(var_);
  }

 protected:
  VariableWrapper* var_;
  size_t ref_cnt_{0};

  std::weak_ptr<LambdaGradAccumulatorPostHook> reduce_hook_;
};

class EagerGradientAccumulator : public GradientAccumulator {
 public:
  using GradientAccumulator::GradientAccumulator;

  void Add(std::shared_ptr<VariableWrapper> var, size_t trace_id,
           bool unchange_input) override;

 private:
  inline bool AccumulateCompleted() const { return cur_cnt_ == ref_cnt_; }

  void IncreaseCurCnt() {
    ++cur_cnt_;
    VLOG(0) << "IncreaseCurCnt: cur_cnt " << cur_cnt_ << "ref_cnt " << ref_cnt_;
    // After all tmp gradient being accumulated to grad var, run hooks
    if (AccumulateCompleted() && HasReduceHook()) {
      CallReduceHook();
    }
  }

 private:
  size_t cur_cnt_{0};
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

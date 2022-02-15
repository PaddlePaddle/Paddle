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
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/pten/api/include/tensor.h"

namespace paddle {
namespace imperative {

class GradientAccumulator {
 public:
  explicit GradientAccumulator(VariableWrapper* var) {
    // var may be initialized, so Synchronous VariableWrapper with Variable
    if (var && var->Var().IsInitialized()) {
      if (var->Var().IsType<framework::LoDTensor>()) {
        var->SetType(framework::proto::VarType::LOD_TENSOR);
      } else if (var->Var().IsType<pten::SelectedRows>()) {
        var->SetType(framework::proto::VarType::SELECTED_ROWS);
      } else {
        PADDLE_THROW(platform::errors::PermissionDenied(
            "Only support LoDTensor and SelectedRows for gradient var"));
      }
    }

    // inner_var_ record the grad of this auto-grad.
    // Only need to generate inner var for leaf-tensor.
    if (var->IsLeafGrad()) {
      inner_var_ = std::make_shared<VariableWrapper>(var->Name());
      inner_var_->SetType(var->Type());
      inner_var_->SetDataType(var->DataType());
      inner_var_->SetForwardDataType(var->ForwardDataType());
      inner_var_->InnerSetOverridedStopGradient(
          var->InnerOverridedStopGradient());
      VLOG(6) << " Create inner grad var for (" << var->Name()
              << ") to store result of this Graph";
    }

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

  // function that Sum Gradient with Previous Graph
  void AccumulateGrad();

  /** [ Hook related methods ]
   *
   *  [Why need two types of VariableWrapperHook? ]
   *
   *    There are two types of gradient accumulation:
   *    1. Gradient accumulation in same batch
   *    2. Gradient accumulation across batchs
   *    The order of execution between Hooks and gradient accumulation:

   *      [ Gradient accumulation in same batch]
   *                        |
   *            [ leaf GradVarBase hooks ]
   *                        |
   *      [ Gradient accumulation across batchs ]
   *                        |
   *          [ Gradient reduce / allreduce hooks ]

   *    Because we currently intend to accumulate these two gradient
   *    accumulation in one GradientAccumulator, We must distinguish between
   *    two types of hooks.

   *    And the InplaceVariableWrapperHook does not allow users to register
   *    directly, and is currently only used to support the reduce strategy of
   *    parallel multi-card training.
   */

  void CallGradientHooks();

  void CallReduceHooks();

 protected:
  VariableWrapper* var_;
  // NOTE: only gradient accumulater of leaf tensor should hold
  // inner_var_, So not hold it by other shared pointer.
  std::shared_ptr<VariableWrapper> inner_var_;
  size_t ref_cnt_{0};
  size_t cur_cnt_{0};
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

template <typename ReturnVarType, typename VarType>
std::shared_ptr<ReturnVarType> SelectedRowsMerge(const VarType& src1,
                                                 const VarType& src2);

template <typename VarType>
void SelectedRowsAddToTensor(const VarType& src, VarType* dst);

template <typename VarType>
void SelectedRowsAddTensor(const VarType& src_selected_rows_var,
                           const VarType& src_tensor_var,
                           VarType* dst_tensor_var);

template <typename VarType>
void TensorAdd(const VarType& src, VarType* dst);

}  // namespace imperative
}  // namespace paddle

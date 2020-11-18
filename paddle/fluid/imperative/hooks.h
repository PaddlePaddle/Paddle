// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

class VariableWrapper;

/** [ Basic hook classes ]
 * s
 * @brief OpBasePreHook is executed before the grad OpBase is executed,
 *        taking the input of the current grad OpBase as input, and
 *        executing python hooks (user-defined) or C++ hooks (developer-defined)
 *        to achieve the purpose of custom operations on the interior VarBase
 *        gradient.
 *
 * @note  OpBasePreHook will not change the input gradient VarBase.
 *
 * @note  [Why need to be OpBase `PreHook`, why not `PostHook`?]
 *
 *        If set OpBase post hook, when the op executed end, the op's output
 *        gradient may not be the final state, because it may need other op's
 *        gradient output to accumulated to it. But before op can be executed,
 *        the gradient output must have been accumulated to final value.
 *
 * @note  [Why only can be used for interior VarBase?]
 *
 *        Because the leaf VarBase's GradVarBase has no GradOpNode, so leaf
 *        GradVarBase has no next OpBase to executed, so if need to deal with
 *        the leaf GradVarBase, cannot use OpBasePreHook. For this case, we
 *        deal with by GradAccumulatorPostHook.
 */
class OpBasePreHook {
 public:
  virtual ~OpBasePreHook() = default;
  virtual VariableWrapperList operator()(
      const VariableWrapperList& grad_inputs) = 0;
};

/**
 * @brief GradAccumulatorPostHook is the Hook that operates on the current
 *        gradientafter the GradientAccumulator has accumulated the gradient.
 *        Leaf GradVarBase has no next OpBase, if we want to register hook
 *        for it, we also need to wait until the leaf GradVarBase accumulation
 *        is completed, so we can add post hook to GradientAccumulator.
 *
 * @note  GradAccumulatorPostHook will change the grad VarBase value.
 *
 * @note  Only allow leaf VarBase hold GradientAccumulatorPostHook.
 */
class GradAccumulatorPostHook {
 public:
  virtual ~GradAccumulatorPostHook() = default;
  virtual void operator()(VariableWrapper* var) = 0;
};

/** [ Hook for cpp functions ]
 *
 * Here we design three C++ hooks；
 * 1. CppOpBasePreHook (Implement later):
 *    - used for developer-defined C++ interior VarBase hooks
 * 2. CppGradAccumulatorPostHook (Implement later):
 *    - used for developer-defined C++ leaf VarBase hooks
 * 3. LambdaGradAccumulatorPostHook:
 *    - used for VarBase reduce in parallel training
 *
 * @note  [Why need two types of GradAccumulatorPostHook? ]
 *
 *        There are two types of gradient accumulation:
 *        1. Gradient accumulation in same batch
 *        2. Gradient accumulation across batchs
 *        The order of execution between Hooks and gradient accumulation:
 *
 *          [ Gradient accumulation in same batch]
 *                            |
 *                [ leaf GradVarBase hooks ]
 *                            |
 *          [ Gradient accumulation across batchs ]
 *                            |
 *              [ Gradient reduce / allreduce]
 *
 *        Because we currently intend to accumulate these two gradient
 *        accumulation in one GradientAccumulator, We must distinguish between
 *        two types of hooks.
 *
 *        And the LambdaGradAccumulatorPostHook does not allow users to register
 *        directly, and is currently only used to support the reduce strategy of
 *        parallel multi-card training.
 */
class LambdaGradAccumulatorPostHook : public GradAccumulatorPostHook {
 public:
  explicit LambdaGradAccumulatorPostHook(
      std::function<void(VariableWrapper*)> fn)
      : fn_(std::move(fn)) {}

  void operator()(VariableWrapper* var) override { fn_(var); }

 private:
  std::function<void(VariableWrapper*)> fn_;
};

/* Hooks for python function: in pybind/imperative.cc */

/** Add Python Hooks later:
 * - PyOpBasePreHook (Implement later): used for user-defined interior python
 * VarBase hooks
 * - PyGradAccumulatorPostHook (Implement later): used for user-defined leaf
 * python VarBase hooks
 */

/** [ Hook Pipeline classes ]
 *
 * @note  [Why need hook pipeline classes?]
 *
 *        There are 2 purposes for adding Hook pipeline here:
 *
 *        1. Make the code implementation cleaner.
 *
 *          If there are no Hook pipeline, we need to add 3 hook vector into
 *          VariableWrapper, 1 hook vector into OpBase, 2 hook vector into
 *          GradientAccumulator, like:
 *
 *          - VariableWrapper:
 *            std::vector<std::shared_ptr<OpBasePreHook>>
 *              interior_var_hooks_;
 *            std::vector<std::shared_ptr<GradAccumulatorPostHook>>
 *              leaf_var_hooks_;
 *            std::vector<std::shared_ptr<GradAccumulatorPostHook>>
 *              backward_hooks_;
 *
 *          - OpBase:
 *            std::vector<std::weak_ptr<OpBasePreHook>>
 *              interior_var_hooks_;
 *
 *          - GradientAccumulator:
 *            std::vector<std::weak_ptr<GradAccumulatorPostHook>>
 *              leaf_var_hooks_;
 *            std::vector<std::weak_ptr<GradAccumulatorPostHook>>
 *              backward_hooks_;
 *
 *          This seems more complicated, and std::vector<std::weak_ptr<...>>
 *          is not easy to destruct.
 *
 *        2. Make the code easier to understand.
 *
 *          From these two packages, we can clearly understand that we
 *          have two types of Hooks, respectively for the interior
 *          gradient var and leaf gradient var inside the backward
 *          calculation graph.
 */

class InteriorVarHookPipeline {
 public:
  InteriorVarHookPipeline() = default;

  void add_hook(std::unique_ptr<OpBasePreHook>&& hook) {
    hooks_.emplace_back(std::move(hook));
  }

  const std::vector<std::unique_ptr<OpBasePreHook>>& hooks() const {
    return hooks_;
  }

  std::vector<std::unique_ptr<OpBasePreHook>>& hooks() { return hooks_; }

 private:
  std::vector<std::unique_ptr<OpBasePreHook>> hooks_;

  DISABLE_COPY_AND_ASSIGN(InteriorVarHookPipeline);
};

class LeafVarHookPipeline {
 public:
  LeafVarHookPipeline() = default;

  void add_hook(std::unique_ptr<GradAccumulatorPostHook>&& hook) {
    hooks_.emplace_back(std::move(hook));
  }

  const std::vector<std::unique_ptr<GradAccumulatorPostHook>>& hooks() const {
    return hooks_;
  }

  std::vector<std::unique_ptr<GradAccumulatorPostHook>>& hooks() {
    return hooks_;
  }

  void add_backward_hook(std::unique_ptr<GradAccumulatorPostHook>&& hook) {
    backward_hooks_.emplace_back(std::move(hook));
  }

  const std::vector<std::unique_ptr<GradAccumulatorPostHook>>& backward_hooks()
      const {
    return backward_hooks_;
  }

  std::vector<std::unique_ptr<GradAccumulatorPostHook>>& backward_hooks() {
    return backward_hooks_;
  }

 private:
  std::vector<std::unique_ptr<GradAccumulatorPostHook>> hooks_;
  // NOTE: the `backward` here means the `whole backward process`,
  // the `backward_hooks_` need to be executed after the `whole backward
  // process`.
  std::vector<std::unique_ptr<GradAccumulatorPostHook>> backward_hooks_;

  DISABLE_COPY_AND_ASSIGN(LeafVarHookPipeline);
};

}  // namespace imperative
}  // namespace paddle

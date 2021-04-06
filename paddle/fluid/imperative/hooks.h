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
namespace paddle {
namespace imperative {

class VariableWrapper;

/** [ Const VariableWrapper Hook: Pre hook functor of OpBase ]
 *
 * @brief This hook functor is executed before the grad OpBase is executed,
 *        taking the input of the current grad OpBase as input, and
 *        executing python hooks (user-defined) or C++ hooks (developer-defined)
 *        to achieve the purpose of custom operations on the interior VarBase
 *        gradient.
 *
 * @note  This hook functor will not change the input gradient VarBase.
 *
 * @note  [Why need to be OpBase `PreHook`, why not `PostHook`?]
 *
 *        1. We expect If set OpBase post hook, when the op executed end, the
 *        op's output gradient may not be the final state, because it may need
 *        other op's gradient output to accumulated to it. But before op can
 *        be executed, the gradient output must have been accumulated to final
 *        value.
 *        2. We don’t want the hook to change its input Tensor value, so now
 *        we can't call all hooks in GradAccumulator.
 *
 * @note  [Why only can be used for interior VarBase?]
 *
 *        Because the leaf VarBase's GradVarBase has no GradOpNode, so leaf
 *        GradVarBase has no next OpBase to executed, so if need to deal with
 *        the leaf GradVarBase, cannot use this hook functor. For this case, we
 *        deal with by other inplace hook method.
 */
class VariableWrapperHook {
 public:
  virtual ~VariableWrapperHook() = default;
  virtual std::shared_ptr<VariableWrapper> operator()(
      const std::shared_ptr<VariableWrapper>& var) = 0;
};

/** [ Inplace VariableWrapper Hook: Post hook functor of GradAccumulator ]
 *
 * @brief This hook functor is the Hook that operates on the current
 *        gradientafter the GradientAccumulator has accumulated the gradient.
 *        Leaf GradVarBase has no next OpBase, if we want to register hook
 *        for it, we also need to wait until the leaf GradVarBase accumulation
 *        is completed, so we can add post hook to GradientAccumulator.
 *
 * @note  This hook functor will change the grad VarBase value.
 *
 * @note  Only allow leaf VarBase hold call this hook functor.
 */
class InplaceVariableWrapperHook {
 public:
  virtual ~InplaceVariableWrapperHook() = default;
  virtual void operator()(VariableWrapper* var) = 0;
};

class LambdaInplaceVariableWrapperHook : public InplaceVariableWrapperHook {
 public:
  explicit LambdaInplaceVariableWrapperHook(
      std::function<void(VariableWrapper*)>&& fn)
      : fn_(std::move(fn)) {}

  void operator()(VariableWrapper* var) override { fn_(var); }

 private:
  std::function<void(VariableWrapper*)> fn_;
};

}  // namespace imperative
}  // namespace paddle

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

/** [ VariableWrapper Hook ]
 *
 * @brief This hook functor is executed before the grad OpBase is executed or
 *        after gradient accumulation completed in current batch.
 *        1. For interior var, VariableWrapper Hook take the input of the
 *        current grad OpBase as input.
 *        2. For leaf var, VariableWrapper Hook take the inner_var_ of
 *        GradientAccumulator as input.
 *
 * @note  This hook functor will not change the input gradient VariableWrapper,
 *        but if you copy the input VariableWrapper and change the value of
 *        Variable in VariableWrapper, the value of input will also be changed,
 *        because they shared same PlaceHolder.
 *
 * @note  [ Why need to be OpBase `PreHook`, why not `PostHook`? ]
 *
 *        We expect If set OpBase post hook, when the op executed end, the
 *        op's output gradient may not be the final state, because it may need
 *        other op's gradient output to accumulated to it. But before op can
 *        be executed, the gradient output must have been accumulated to final
 *        value.
 *
 * @note  [ Why Leaf gradient is special? ]
 *
 *        Because the leaf VarBase's GradVarBase has no GradOpNode, so leaf
 *        GradVarBase has no next OpBase to executed, so if need to deal with
 *        the leaf GradVarBase, we should call hooks after gradient accumulation
 *        completed.
 */
class VariableWrapperHook {
 public:
  virtual ~VariableWrapperHook() = default;
  virtual std::shared_ptr<VariableWrapper> operator()(
      const std::shared_ptr<VariableWrapper>& var) = 0;
};

class CppVariableWrapperHook : public VariableWrapperHook {
 public:
  explicit CppVariableWrapperHook(
      std::function<std::shared_ptr<VariableWrapper>(
          const std::shared_ptr<VariableWrapper>&)>&& fn)
      : fn_(std::move(fn)) {}

  std::shared_ptr<VariableWrapper> operator()(
      const std::shared_ptr<VariableWrapper>& var) override {
    return fn_(var);
  }

 private:
  std::function<std::shared_ptr<VariableWrapper>(
      const std::shared_ptr<VariableWrapper>&)>
      fn_;
};

}  // namespace imperative
}  // namespace paddle

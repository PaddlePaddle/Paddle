//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <utility>

#include "paddle/pten/core/tensor_base.h"
#include "paddle/utils/any.h"
#include "paddle/utils/small_vector.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace pten {

using DeviceContext = paddle::platform::DeviceContext;
using DataType = paddle::experimental::DataType;
using DataLayout = paddle::experimental::DataLayout;

/**
 * Note: KernelContext doesn't manage the life if DeviceContext and Tensor
 *
 * Note: KernelContext does not couple the concept of framework,
 *       its constructor can only take the members it needs as parameters,
 *       not Scope, RuntimeContext, etc. as parameters
 */
class KernelContext {
 public:
  explicit KernelContext(const DeviceContext& dev_ctx) : dev_ctx_(dev_ctx) {}
  KernelContext(const DeviceContext& dev_ctx,
                const paddle::SmallVector<std::shared_ptr<TensorBase>>& inputs,
                const paddle::SmallVector<std::shared_ptr<TensorBase>>& outputs,
                const paddle::SmallVector<paddle::any>& attrs)
      : dev_ctx_(dev_ctx), inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  template <typename CtxType>
  const CtxType& GetDeviceContext() const {
    return static_cast<const CtxType&>(dev_ctx_);
  }

  void EmplaceBackInput(std::shared_ptr<TensorBase> input) {
    inputs_.emplace_back(input);
    // Record the start and end index of the input
    int index = inputs_.size();
    input_range_.emplace_back(std::pair<int, int>(index, index + 1));
  }

  void EmplaceBackInputs(
      const paddle::SmallVector<std::shared_ptr<TensorBase>>& inputs) {
    for (auto in : inputs) {
      inputs_.emplace_back(in);
    }
    // Record the start and end index of the input
    int index = inputs_.size();
    input_range_.emplace_back(
        std::pair<int, int>(index, index + inputs.size()));
  }

  void EmplaceBackOutput(std::shared_ptr<TensorBase> output) {
    outputs_.emplace_back(output);
    // Record the start and end index of the input
    int index = outputs_.size();
    output_range_.emplace_back(std::pair<int, int>(index, index + 1));
  }

  void EmplaceBackOutputs(
      const paddle::SmallVector<std::shared_ptr<TensorBase>>& outputs) {
    for (auto out : outputs) {
      outputs_.emplace_back(out);
    }
    // Record the start and end index of the input
    int index = outputs_.size();
    output_range_.emplace_back(
        std::pair<int, int>(index, index + outputs.size()));
  }

  void EmplaceBackAttr(paddle::any attr) { attrs_.emplace_back(attr); }

  template <typename TensorType>
  const TensorType& InputAt(size_t idx) const {
    return static_cast<const TensorType&>(*(inputs_.at(idx)));
  }

  template <typename TensorType>
  TensorType* MutableOutputAt(size_t idx) {
    return static_cast<TensorType*>(outputs_.at(idx).get());
  }

  template <typename AttrType>
  AttrType AttrAt(size_t idx) const {
    try {
      return paddle::any_cast<AttrType>(attrs_.at(idx));
    } catch (paddle::bad_any_cast&) {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Attribute cast error in Op Kernel Context."));
    }
  }

 private:
  bool IsDuplicable() const { return input_range_.size() != inputs_.size(); }

 private:
  // DeviceContext base class
  const DeviceContext& dev_ctx_;

  // TODO(chenweihang): Tensor -> Tensor*, Tensor should by managed `scope`
  // Note: can't use API Tensor here, the inference don't use this API Tensor
  paddle::SmallVector<std::shared_ptr<TensorBase>> inputs_{};
  paddle::SmallVector<std::shared_ptr<TensorBase>> outputs_{};
  paddle::SmallVector<paddle::any> attrs_{};

  // Only contains input like list[Tensor] need `range`
  paddle::SmallVector<std::pair<int, int>> input_range_{{}};
  paddle::SmallVector<std::pair<int, int>> output_range_{{}};

  // Only static graph need `name`
  // TODO(chenweihang): replaced by paddle::string_view
  paddle::SmallVector<std::string> input_names_{{}};
  paddle::SmallVector<std::string> output_names_{{}};
};

}  // namespace pten

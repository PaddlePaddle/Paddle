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

#include <iterator>
#include <utility>

#include "paddle/pten/core/compat_utils.h"
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
  KernelContext() = default;
  explicit KernelContext(DeviceContext* dev_ctx) : dev_ctx_(dev_ctx) {}

  void SetDeviceContext(DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }

  template <typename CtxType>
  const CtxType& GetDeviceContext() const {
    return static_cast<const CtxType&>(*dev_ctx_);
  }

  void EmplaceBackInput(std::shared_ptr<TensorBase> input) {
    int index = inputs_.size();
    inputs_.emplace_back(std::move(input));
    // Record the start and end index of the input
    input_range_.emplace_back(std::pair<int, int>(index, index + 1));
  }

  void EmplaceBackInputs(
      paddle::SmallVector<std::shared_ptr<TensorBase>> inputs) {
    int index = inputs_.size();
    // Record the start and end index of the input
    input_range_.emplace_back(
        std::pair<int, int>(index, index + inputs.size()));
    inputs_.insert(inputs_.end(),
                   std::make_move_iterator(inputs.begin()),
                   std::make_move_iterator(inputs.end()));
  }

  void EmplaceBackOutput(std::shared_ptr<TensorBase> output) {
    int index = outputs_.size();
    outputs_.emplace_back(std::move(output));
    // Record the start and end index of the input
    output_range_.emplace_back(std::pair<int, int>(index, index + 1));
  }

  void EmplaceBackOutputs(
      paddle::SmallVector<std::shared_ptr<TensorBase>> outputs) {
    int index = outputs_.size();
    // Record the start and end index of the input
    output_range_.emplace_back(
        std::pair<int, int>(index, index + outputs.size()));
    outputs_.insert(outputs_.end(),
                    std::make_move_iterator(outputs.begin()),
                    std::make_move_iterator(outputs.end()));
  }

  void EmplaceBackAttr(paddle::any attr) {
    attrs_.emplace_back(std::move(attr));
  }

  template <typename TensorType>
  const TensorType& InputAt(size_t idx) const {
    return static_cast<const TensorType&>(*(inputs_.at(idx)));
  }

  template <typename TensorType>
  std::vector<TensorType> InputBetween(size_t start, size_t end) const {
    std::vector<TensorType> v;
    for (size_t i = start; i < end; ++i) {
      auto t = std::dynamic_pointer_cast<TensorType>(inputs_.at(i));
      v.emplace_back(std::move(*t.get()));
    }

    return v;
  }

  const std::pair<int, int>& InputRangeAt(size_t idx) const {
    return input_range_.at(idx);
  }

  const std::pair<int, int>& OutputRangeAt(size_t idx) const {
    return output_range_.at(idx);
  }

  std::pair<int, int>& MutableInputRangeAt(size_t idx) {
    return input_range_[idx];
  }

  std::pair<int, int>& MutableOutputRangeAt(size_t idx) {
    return output_range_[idx];
  }

  template <typename TensorType>
  TensorType* MutableInputAt(size_t idx) {
    return static_cast<TensorType*>(inputs_.at(idx).get());
  }

  template <typename TensorType>
  TensorType* MutableOutputAt(size_t idx) {
    return static_cast<TensorType*>(outputs_.at(idx).get());
  }

  template <typename TensorType>
  std::vector<TensorType*> MutableOutputBetween(size_t start, size_t end) {
    std::vector<TensorType*> v;
    for (size_t i = start; i < end; ++i) {
      v.emplace_back(static_cast<TensorType*>(outputs_.at(i).get()));
    }

    return v;
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

  // Temporary method: For compatible with fluid Tensor and improve performance
  // Only deal with DenseTensor now
  void ClearData() {
    for (auto& in : inputs_) {
      CompatibleDenseTensorUtils::ClearStorage(
          static_cast<DenseTensor*>(in.get()));
    }
    for (auto& out : outputs_) {
      CompatibleDenseTensorUtils::ClearStorage(
          static_cast<DenseTensor*>(out.get()));
    }
    attrs_.clear();
  }

  size_t InputsSize() const { return inputs_.size(); }
  size_t OutputsSize() const { return outputs_.size(); }
  size_t AttrsSize() const { return attrs_.size(); }

 private:
  bool IsDuplicable() const { return input_range_.size() != inputs_.size(); }

 private:
  // DeviceContext base class
  DeviceContext* dev_ctx_;

  // TODO(chenweihang): Tensor -> Tensor*, Tensor should by managed `scope`
  // Note: can't use API Tensor here, the inference don't use this API Tensor
  paddle::SmallVector<std::shared_ptr<TensorBase>> inputs_;
  paddle::SmallVector<std::shared_ptr<TensorBase>> outputs_;
  paddle::SmallVector<paddle::any> attrs_;

  // Only contains input like list[Tensor] need `range`
  paddle::SmallVector<std::pair<int, int>> input_range_;
  paddle::SmallVector<std::pair<int, int>> output_range_;
};

}  // namespace pten

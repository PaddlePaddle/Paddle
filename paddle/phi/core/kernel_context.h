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

#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/optional.h"
#include "paddle/utils/small_vector.h"

namespace phi {

/**
 * Note: KernelContext doesn't manage the life of DeviceContext and Tensor
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

  void EmplaceBackInput(const TensorBase* input);

  void EmplaceBackInputWithoutSetRange(const TensorBase* input);

  void EmplaceBackInputs(paddle::SmallVector<const TensorBase*> inputs);

  void EmplaceBackInputsWithoutSetRange(
      paddle::SmallVector<const TensorBase*> inputs);

  void EmplaceBackOutput(TensorBase* output);

  void EmplaceBackOutputWithoutSetRange(TensorBase* output);

  void EmplaceBackOutputs(paddle::SmallVector<TensorBase*> outputs);

  void EmplaceBackOutputsWithoutSetRange(
      paddle::SmallVector<TensorBase*> outputs);

  void EmplaceBackAttr(Attribute attr);

  const std::pair<int, int>& InputRangeAt(size_t idx) const;

  const std::pair<int, int>& OutputRangeAt(size_t idx) const;

  void AssignInputRange(std::pair<int, int>&& range, size_t idx);

  void AssignOutputRange(std::pair<int, int>&& range, size_t idx);

  template <typename TensorType>
  const TensorType& InputAt(size_t idx) const {
    return static_cast<const TensorType&>(*(inputs_.at(idx)));
  }

  template <typename TensorType>
  paddle::optional<const TensorType&> OptionalInputAt(size_t idx) const {
    const auto& input = inputs_.at(idx);
    return input ? paddle::optional<const TensorType&>{static_cast<
                       const TensorType&>(*input)}
                 : paddle::optional<const TensorType&>{paddle::none};
  }

  template <typename TensorType>
  std::vector<const TensorType*> InputsBetween(size_t start, size_t end) {
    std::vector<const TensorType*> v;
    for (size_t i = start; i < end; ++i) {
      auto* t = static_cast<const TensorType*>(inputs_.at(i));
      v.emplace_back(t);
    }
    return v;
  }

  template <typename TensorType>
  paddle::optional<const std::vector<const TensorType*>> OptionalInputsBetween(
      size_t start, size_t end) {
    const auto& first = inputs_.at(start);

    if (first) {
      std::vector<const TensorType*> v;
      for (size_t i = start; i < end; ++i) {
        auto* t = static_cast<const TensorType*>(inputs_.at(i));
        v.emplace_back(t);
      }
      return paddle::optional<const std::vector<const TensorType*>>(v);
    }
    return paddle::optional<const std::vector<const TensorType*>>(paddle::none);
  }

  template <typename TensorType>
  TensorType* MutableOutputAt(size_t idx) {
    return static_cast<TensorType*>(outputs_.at(idx));
  }

  template <typename TensorType>
  std::vector<TensorType*> MutableOutputBetween(size_t start, size_t end) {
    std::vector<TensorType*> v;
    for (size_t i = start; i < end; ++i) {
      v.emplace_back(static_cast<TensorType*>(outputs_.at(i)));
    }
    return v;
  }

  template <typename AttrType>
  const AttrType& AttrAt(size_t idx) const;

  size_t InputsSize() const { return inputs_.size(); }
  size_t OutputsSize() const { return outputs_.size(); }
  size_t AttrsSize() const { return attrs_.size(); }

 private:
  DeviceContext* dev_ctx_;

  paddle::SmallVector<const TensorBase*> inputs_;
  paddle::SmallVector<TensorBase*> outputs_;
  paddle::SmallVector<Attribute, kAttrSmallVectorSize> attrs_;

  paddle::SmallVector<std::pair<int, int>, kInputSmallVectorSize> input_range_;
  paddle::SmallVector<std::pair<int, int>, kOutputSmallVectorSize>
      output_range_;
};

}  // namespace phi

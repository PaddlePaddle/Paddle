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

  void EmplaceBackInput(const TensorBase* input);

  void EmplaceBackInputWithoutSetRange(const TensorBase* input);

  void EmplaceBackInputs(paddle::SmallVector<const TensorBase*> inputs);

  void EmplaceBackOutput(TensorBase* output);

  void EmplaceBackOutputWithoutSetRange(TensorBase* output);

  void EmplaceBackOutputs(paddle::SmallVector<TensorBase*> outputs);

  void SetOutputWithoutSetRange(int index, TensorBase* output);

  void EmplaceBackAttr(paddle::any attr);

  const std::pair<int, int>& InputRangeAt(size_t idx) const;

  const std::pair<int, int>& OutputRangeAt(size_t idx) const;

  std::pair<int, int>& MutableInputRangeAt(size_t idx);

  std::pair<int, int>& MutableOutputRangeAt(size_t idx);

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
  std::vector<TensorType> MoveInputsBetween(size_t start, size_t end) {
    std::vector<TensorType> v;
    for (size_t i = start; i < end; ++i) {
      auto t = static_cast<const TensorType*>(inputs_.at(i));
      v.emplace_back(*t);
      inputs_.at(i) = nullptr;
    }
    return v;
  }

  void AssignInputRange(std::pair<int, int>&& range, size_t idx);

  void AssignOutputRange(std::pair<int, int>&& range, size_t idx);

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
  void ClearData();

  size_t InputsSize() const { return inputs_.size(); }
  size_t OutputsSize() const { return outputs_.size(); }
  size_t AttrsSize() const { return attrs_.size(); }

 private:
  // DeviceContext base class
  DeviceContext* dev_ctx_;

  // TODO(chenweihang): Tensor -> Tensor*, Tensor should by managed `scope`
  // Note: can't use API Tensor here, the inference don't use this API Tensor
  paddle::SmallVector<const TensorBase*> inputs_;
  paddle::SmallVector<TensorBase*> outputs_;
  paddle::SmallVector<paddle::any> attrs_;

  // Only contains input like list[Tensor] need `range`
  paddle::SmallVector<std::pair<int, int>> input_range_;
  paddle::SmallVector<std::pair<int, int>> output_range_;
};

}  // namespace pten

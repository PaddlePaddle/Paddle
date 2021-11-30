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

#include "paddle/pten/core/kernel_context.h"

namespace pten {
template <typename CtxType>
const CtxType& KernelContext::GetDeviceContext() const {
  return static_cast<const CtxType&>(*dev_ctx_);
}
void KernelContext::EmplaceBackInput(std::shared_ptr<TensorBase> input) {
  int index = inputs_.size();
  inputs_.emplace_back(std::move(input));
  // Record the start and end index of the input
  input_range_.emplace_back(std::pair<int, int>(index, index + 1));
}

void KernelContext::EmplaceBackInputWithoutSetRange(
    std::shared_ptr<TensorBase> input) {
  inputs_.emplace_back(std::move(input));
}

void KernelContext::EmplaceBackInputs(
    paddle::SmallVector<std::shared_ptr<TensorBase>> inputs) {
  int index = inputs_.size();
  // Record the start and end index of the input
  input_range_.emplace_back(std::pair<int, int>(index, index + inputs.size()));
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
}

void KernelContext::EmplaceBackOutput(std::shared_ptr<TensorBase> output) {
  int index = outputs_.size();
  outputs_.emplace_back(std::move(output));
  // Record the start and end index of the input
  output_range_.emplace_back(std::pair<int, int>(index, index + 1));
}

void KernelContext::EmplaceBackOutputWithoutSetRange(
    std::shared_ptr<TensorBase> output) {
  outputs_.emplace_back(std::move(output));
}

void KernelContext::EmplaceBackOutputs(
    paddle::SmallVector<std::shared_ptr<TensorBase>> outputs) {
  int index = outputs_.size();
  // Record the start and end index of the input
  output_range_.emplace_back(
      std::pair<int, int>(index, index + outputs.size()));
  outputs_.insert(outputs_.end(),
                  std::make_move_iterator(outputs.begin()),
                  std::make_move_iterator(outputs.end()));
}

void KernelContext::EmplaceBackAttr(paddle::any attr) {
  attrs_.emplace_back(std::move(attr));
}

template <typename TensorType>
const TensorType& KernelContext::InputAt(size_t idx) const {
  return static_cast<const TensorType&>(*(inputs_.at(idx)));
}

template <typename TensorType>
std::vector<TensorType> KernelContext::InputBetween(size_t start,
                                                    size_t end) const {
  std::vector<TensorType> v;
  for (size_t i = start; i < end; ++i) {
    auto t = std::dynamic_pointer_cast<TensorType>(inputs_.at(i));
    v.emplace_back(std::move(*t.get()));
  }

  return v;
}

const std::pair<int, int>& KernelContext::InputRangeAt(size_t idx) const {
  return input_range_.at(idx);
}

const std::pair<int, int>& KernelContext::OutputRangeAt(size_t idx) const {
  return output_range_.at(idx);
}

std::pair<int, int>& KernelContext::MutableInputRangeAt(size_t idx) {
  return input_range_[idx];
}

std::pair<int, int>& KernelContext::MutableOutputRangeAt(size_t idx) {
  return output_range_[idx];
}

template <typename TensorType>
TensorType* KernelContext::MutableInputAt(size_t idx) {
  return static_cast<TensorType*>(inputs_.at(idx).get());
}

template <typename TensorType>
TensorType* KernelContext::MutableOutputAt(size_t idx) {
  return static_cast<TensorType*>(outputs_.at(idx).get());
}

template <typename TensorType>
std::vector<TensorType*> KernelContext::MutableOutputBetween(size_t start,
                                                             size_t end) {
  std::vector<TensorType*> v;
  for (size_t i = start; i < end; ++i) {
    v.emplace_back(static_cast<TensorType*>(outputs_.at(i).get()));
  }

  return v;
}

template <typename AttrType>
AttrType KernelContext::AttrAt(size_t idx) const {
  try {
    return paddle::any_cast<AttrType>(attrs_.at(idx));
  } catch (paddle::bad_any_cast&) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Attribute cast error in Op Kernel Context."));
  }
}

// Temporary method: For compatible with fluid Tensor and improve performance
// Only deal with DenseTensor now
void KernelContext::ClearData() {
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
}  // namespace pten

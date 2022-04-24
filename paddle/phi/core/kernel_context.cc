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

#include "paddle/phi/core/kernel_context.h"

namespace phi {

void KernelContext::EmplaceBackInput(const TensorBase* input) {
  int index = inputs_.size();
  inputs_.emplace_back(input);
  // Record the start and end index of the input
  input_range_.emplace_back(std::pair<int, int>(index, index + 1));
}

void KernelContext::EmplaceBackInputWithoutSetRange(const TensorBase* input) {
  inputs_.emplace_back(input);
}

void KernelContext::EmplaceBackInputs(
    paddle::SmallVector<const TensorBase*> inputs) {
  int index = inputs_.size();
  // Record the start and end index of the input
  input_range_.emplace_back(std::pair<int, int>(index, index + inputs.size()));
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
}

void KernelContext::EmplaceBackInputsWithoutSetRange(
    paddle::SmallVector<const TensorBase*> inputs) {
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
}

void KernelContext::EmplaceBackOutput(TensorBase* output) {
  int index = outputs_.size();
  outputs_.emplace_back(output);
  // Record the start and end index of the input
  output_range_.emplace_back(std::pair<int, int>(index, index + 1));
}

void KernelContext::EmplaceBackOutputWithoutSetRange(TensorBase* output) {
  outputs_.emplace_back(output);
}

void KernelContext::EmplaceBackOutputs(
    paddle::SmallVector<TensorBase*> outputs) {
  int index = outputs_.size();
  // Record the start and end index of the input
  output_range_.emplace_back(
      std::pair<int, int>(index, index + outputs.size()));
  outputs_.insert(outputs_.end(),
                  std::make_move_iterator(outputs.begin()),
                  std::make_move_iterator(outputs.end()));
}

void KernelContext::EmplaceBackOutputsWithoutSetRange(
    paddle::SmallVector<TensorBase*> outputs) {
  outputs_.insert(outputs_.end(),
                  std::make_move_iterator(outputs.begin()),
                  std::make_move_iterator(outputs.end()));
}

void KernelContext::EmplaceBackAttr(Attribute attr) {
  attrs_.emplace_back(std::move(attr));
}

void KernelContext::AssignInputRange(std::pair<int, int>&& range, size_t idx) {
  if (idx < input_range_.size()) {
    input_range_[idx] = std::move(range);
  } else if (idx == input_range_.size()) {
    input_range_.emplace_back(range);
  } else {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "Invalid idx when trying to set InputRange, "
        "index is `%d`, it is greater than the size(%d) of InputRange.",
        idx,
        input_range_.size()));
  }
}

void KernelContext::AssignOutputRange(std::pair<int, int>&& range, size_t idx) {
  if (idx < output_range_.size()) {
    output_range_[idx] = std::move(range);
  } else if (idx == output_range_.size()) {
    output_range_.emplace_back(range);
  } else {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "Invalid idx when trying to set InputRange, "
        "index is `%d`, it is greater than the size(%d) of InputRange.",
        idx,
        output_range_.size()));
  }
}

const std::pair<int, int>& KernelContext::InputRangeAt(size_t idx) const {
  return input_range_.at(idx);
}

const std::pair<int, int>& KernelContext::OutputRangeAt(size_t idx) const {
  return output_range_.at(idx);
}

template <typename AttrType>
const AttrType& KernelContext::AttrAt(size_t idx) const {
  try {
    return paddle::get<AttrType>(attrs_.at(idx));
  } catch (paddle::bad_variant_access const& ex) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Attribute cast error in Op Kernel Context."));
  }
}

template const bool& KernelContext::AttrAt(size_t idx) const;
template const int& KernelContext::AttrAt(size_t idx) const;
template const int64_t& KernelContext::AttrAt(size_t idx) const;
template const float& KernelContext::AttrAt(size_t idx) const;
template const double& KernelContext::AttrAt(size_t idx) const;
template const std::string& KernelContext::AttrAt(size_t idx) const;
template const std::vector<bool>& KernelContext::AttrAt(size_t idx) const;
template const std::vector<int>& KernelContext::AttrAt(size_t idx) const;
template const std::vector<int64_t>& KernelContext::AttrAt(size_t idx) const;
template const std::vector<float>& KernelContext::AttrAt(size_t idx) const;
template const std::vector<double>& KernelContext::AttrAt(size_t idx) const;
template const std::vector<std::string>& KernelContext::AttrAt(
    size_t idx) const;
template const Scalar& KernelContext::AttrAt(size_t idx) const;
template const std::vector<Scalar>& KernelContext::AttrAt(size_t idx) const;
template const IntArray& KernelContext::AttrAt(size_t idx) const;
template const DataType& KernelContext::AttrAt(size_t idx) const;
template const DataLayout& KernelContext::AttrAt(size_t idx) const;
template const Place& KernelContext::AttrAt(size_t idx) const;

}  // namespace phi

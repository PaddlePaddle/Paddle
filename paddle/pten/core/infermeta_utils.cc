/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/infermeta_utils.h"

namespace pten {

void InferMetaContext::SetMetaConfig(MetaConfig config) {
  config_ = std::move(config);
}

void InferMetaContext::EmplaceBackInput(
    std::shared_ptr<pten::MetaTensor> input) {
  int index = inputs_.size();
  inputs_.emplace_back(std::move(input));
  input_range_.emplace_back(std::pair<int, int>(index, index + 1));
}
void InferMetaContext::EmplaceBackOutput(
    std::shared_ptr<pten::MetaTensor> output) {
  int index = outputs_.size();
  outputs_.emplace_back(std::move(output));
  output_range_.emplace_back(std::pair<int, int>(index, index + 1));
}
void InferMetaContext::EmplaceBackAttr(paddle::any attr) {
  attrs_.emplace_back(std::move(attr));
}

void InferMetaContext::EmplaceBackInputs(
    paddle::SmallVector<std::shared_ptr<pten::MetaTensor>> inputs) {
  int index = inputs_.size();
  input_range_.emplace_back(std::pair<int, int>(index, index + inputs.size()));
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
}
void InferMetaContext::EmplaceBackOutputs(
    paddle::SmallVector<std::shared_ptr<pten::MetaTensor>> outputs) {
  int index = outputs_.size();
  output_range_.emplace_back(
      std::pair<int, int>(index, index + outputs.size()));
  outputs_.insert(outputs_.end(),
                  std::make_move_iterator(outputs.begin()),
                  std::make_move_iterator(outputs.end()));
}

const std::pair<int, int>& InferMetaContext::InputRangeAt(size_t idx) const {
  return input_range_.at(idx);
}
const std::pair<int, int>& InferMetaContext::OutputRangeAt(size_t idx) const {
  return output_range_.at(idx);
}

const MetaConfig& InferMetaContext::GetMetaConfig() const { return config_; }

const MetaTensor& InferMetaContext::InputAt(size_t idx) const {
  return *inputs_.at(idx);
}
MetaTensor* InferMetaContext::MutableOutputAt(size_t idx) {
  return outputs_.at(idx).get();
}

}  // namespace pten

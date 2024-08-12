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

#include "paddle/phi/core/infermeta_utils.h"

namespace phi {

void InferMetaContext::SetMetaConfig(MetaConfig config) { config_ = config; }

void InferMetaContext::EmplaceBackInput(MetaTensor input) {
  int index = static_cast<int>(inputs_.size());
  inputs_.emplace_back(std::move(input));
  input_range_.emplace_back(std::pair<int, int>(index, index + 1));
}
void InferMetaContext::EmplaceBackOutput(MetaTensor output) {
  int index = static_cast<int>(outputs_.size());
  outputs_.emplace_back(std::move(output));
  output_range_.emplace_back(std::pair<int, int>(index, index + 1));
}
void InferMetaContext::EmplaceBackAttr(Attribute attr) {
  attrs_.emplace_back(std::move(attr));
}

void InferMetaContext::EmplaceBackInputs(
    paddle::small_vector<MetaTensor, phi::kInputSmallVectorSize> inputs) {
  int index = static_cast<int>(inputs_.size());
  input_range_.emplace_back(std::pair<int, int>(index, index + inputs.size()));
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
}
void InferMetaContext::EmplaceBackOutputs(
    paddle::small_vector<MetaTensor, phi::kOutputSmallVectorSize> outputs) {
  int index = static_cast<int>(outputs_.size());
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
  return inputs_.at(idx);
}

std::vector<const MetaTensor*> InferMetaContext::InputsBetween(
    size_t start, size_t end) const {
  std::vector<const MetaTensor*> result;
  // If vector only contains one input that is not initialized,
  // we should return a empty vector
  if (end - start == 1 && !inputs_.at(start).initialized()) {
    return result;
  }

  result.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    auto& in = inputs_.at(i);
    result.emplace_back(in.initialized() ? &in : nullptr);
  }

  return result;
}

paddle::optional<std::vector<const MetaTensor*>>
InferMetaContext::OptionalInputsBetween(size_t start, size_t end) const {
  const auto& first = inputs_.at(start);

  if (first.initialized()) {
    std::vector<const MetaTensor*> result;
    result.reserve(end - start);

    for (size_t i = start; i < end; ++i) {
      auto& in = inputs_.at(i);
      result.emplace_back(in.initialized() ? &in : nullptr);
    }

    return paddle::optional<std::vector<const MetaTensor*>>(result);
  }
  return paddle::none;
}

MetaTensor* InferMetaContext::MutableOutputAt(size_t idx) {
  auto& out = outputs_.at(idx);
  return out.initialized() ? &out : nullptr;
}

std::vector<MetaTensor*> InferMetaContext::MutableOutputBetween(size_t start,
                                                                size_t end) {
  std::vector<MetaTensor*> result;
  // If vector only contains one output that is not initialized,
  // we should return a empty vector
  if (end - start == 1 && !outputs_.at(start).initialized()) {
    return result;
  }

  result.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    auto& out = outputs_.at(i);
    result.emplace_back(out.initialized() ? &out : nullptr);
  }
  return result;
}

template <typename AttrType>
const AttrType& InferMetaContext::AttrAt(size_t idx) const {
  try {
    return paddle::get<AttrType>(attrs_.at(idx));
  } catch (paddle::bad_variant_access const& e) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Attribute cast error in InferMeta Context, the expected attribute "
        "type is `%s`.",
        std::type_index(typeid(AttrType)).name()));
  }
}

const Attribute& InferMetaContext::AttrAt(size_t idx) const {
  return attrs_.at(idx);
}

template const bool& InferMetaContext::AttrAt(size_t idx) const;
template const int& InferMetaContext::AttrAt(size_t idx) const;
template const int64_t& InferMetaContext::AttrAt(size_t idx) const;
template const float& InferMetaContext::AttrAt(size_t idx) const;
template const double& InferMetaContext::AttrAt(size_t idx) const;
template const std::string& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<bool>& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<int>& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<int64_t>& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<float>& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<double>& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<std::string>& InferMetaContext::AttrAt(
    size_t idx) const;
template const Scalar& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<Scalar>& InferMetaContext::AttrAt(size_t idx) const;
template const IntArray& InferMetaContext::AttrAt(size_t idx) const;
template TEST_API const DataType& InferMetaContext::AttrAt(size_t idx) const;
template const DataLayout& InferMetaContext::AttrAt(size_t idx) const;
template const Place& InferMetaContext::AttrAt(size_t idx) const;
template const TensorRef& InferMetaContext::AttrAt(size_t idx) const;
template const std::vector<TensorRef>& InferMetaContext::AttrAt(
    size_t idx) const;

MetaFnFactory& MetaFnFactory::Instance() {
  static MetaFnFactory g_meta_fn_map;
  return g_meta_fn_map;
}

}  // namespace phi

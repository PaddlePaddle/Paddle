// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/pattern_rewrite/drr/api/drr_pass_context.h"

#include <glog/logging.h>
#include "paddle/ir/pattern_rewrite/drr/pattern_graph.h"

namespace ir {
namespace drr {

const Op& DrrPassContext::SourceOpPattern(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes = {}) {
  owned_ops_.push_back(std::make_shared<drr::Op>(
      op_type, attributes, source_pattern_graph_.get()));
  return *owned_ops_.back();
}

const drr::Tensor& DrrPassContext::SourceTensorPattern(
    const std::string& tensor_id) {
  return source_pattern_graph_->AddTensor(
      std::make_shared<drr::Tensor>(tensor_id, source_pattern_graph_.get()));
}

const Op& DrrPassContext::ResultOpPattern(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes = {}) {
  owned_ops_.push_back(std::make_shared<drr::Op>(
      op_type, attributes, result_pattern_graph_.get()));
  return *owned_ops_.back();
}

const drr::Tensor& DrrPassContext::SourceTensorPattern(
    const std::string& tensor_id) {
  return result_pattern_graph_->AddTensor(
      std::make_shared<drr::Tensor>(tensor_id, result_pattern_graph_.get()));
}

void Op::operator()(const Tensor& arg, const Tensor* out) const {
  std::vector<std::weak_ptr<const Tensor>> inputs{arg.shared_from_this()};
  std::vector<std::weak_ptr<const Tensor>> outputs{out->shared_from_this()};
  pattern_graph_->AddOpCall(
      std::make_shared<OpCall>(shared_from_this(), inputs, outputs));
}

Tensor& Op::operator()(const Tensor& arg) const {
  std::vector<std::weak_ptr<const Tensor>> inputs{arg.shared_from_this()};
  auto& out = pattern_graph_->AddTmpTensor(std::make_shared<Tensor>(
      "tmp_" + op_type_name_ + std::to_string(count++), pattern_graph_));
  std::vector<std::weak_ptr<const Tensor>> outputs{out.shared_from_this()};
  pattern_graph_->AddOpCall(
      std::make_shared<OpCall>(shared_from_this(), inputs, outputs));
  return out;
}

Tensor& Op::operator()() const {
  std::vector<std::weak_ptr<const Tensor>> inputs{};
  auto& out = pattern_graph_->AddTmpTensor(std::make_shared<Tensor>(
      "tmp_" + op_type_name_ + std::to_string(count++), pattern_graph_));
  std::vector<std::weak_ptr<const Tensor>> outputs{out.shared_from_this()};
  pattern_graph_->AddOpCall(
      std::make_shared<OpCall>(shared_from_this(), inputs, outputs));
  return out;
}

int64_t Op::count = 0;

void Tensor::operator=(Tensor& other) const {  // NOLINT
  // The two tensor must be in the same pattern graph.
  CHECK(this->pattern_graph_ == other.pattern_graph_);
  if (other.tensor_id_.substr(0, 4) == "tmp_") {
    pattern_graph_->UpdateTmpTensor(other.tensor_id_, this->tensor_id_);
  }
}

}  // namespace drr
}  // namespace ir

// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/ir/pattern_rewrite/drr/source_pattern.h"

namespace ir {
namespace drr {

const Op& DrrPassContext::SourceOpPattern(
    const std::string& op_type,
    std::unordered_map<std::string, const Attribute&> attributes = {}) {
  owned_ops_.push_back(std::make_shared<drr::Op>(
      op_type, attributes, [&](const std::shared_ptr<OpCall>& op_call) {
        source_pattern_graph_->AddOpCall(op_call);
      }));
  return *owned_ops_.back();
}

const drr::Tensor& DrrPassContext::SourceTensorPattern(
    const std::string& tensor_id) {
  return source_pattern_graph_->AddTensor(
      std::make_shared<drr::Tensor>(tensor_id));
}

const Op& DrrPassContext::ResultOpPattern(
    const std::string& op_type,
    std::unordered_map<std::string, const Attribute&> attributes = {}) {
  owned_ops_.push_back(std::make_shared<drr::Op>(
      op_type, attributes, [&](const std::shared_ptr<OpCall>& op_call) {
        result_pattern_graph_->AddOpCall(op_call);
      }));
  return *owned_ops_.back();
}

const drr::Tensor& DrrPassContext::SourceTensorPattern(
    const std::string& tensor_id) {
  return result_pattern_graph_->AddTensor(
      std::make_shared<drr::Tensor>(tensor_id));
}

}  // namespace drr
}  // namespace ir

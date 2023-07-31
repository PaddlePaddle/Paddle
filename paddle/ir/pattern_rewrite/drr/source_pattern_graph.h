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

#pragma once

#include "paddle/ir/pattern_rewrite/drr/api/drr_pass_context.h"

namespace ir {
namespace drr {

class Constrain;
class MatchContext;
class OpCall;
class Tensor;

using id_type = std::string;

class SourcePatternGraph {
 public:
  const drr::OpCall& AddOpCall(const std::shared_ptr<drr::OpCall>& op_call) {
    owned_op_call_.push_back(op_call);
    for (const auto& input : op_call->inputs()) {
      const auto& tensor_id = input.lock()->id();
      id2owned_tensor_[tensor_id]->AddConsumer(op_call);

      if (input.lock()->producer().use_count() == 0) {
        input_tensors.insert(tensor_id);
      }
      if (output_tensors.find(tensor_id) != output_tensors.end()) {
        output_tensors.erase(tensor_id);
      }
    }
    for (auto& output : op_call->outputs()) {
      id2owned_tensor_[output.lock()->id()]->set_producer(op_call);
    }
    return *owned_op_call_.back();
  }

  void MergeTensor(drr::Tensor* value_tensor, drr::Tensor* name_tensor) {}

  const drr::Tensor& AddTensor(const std::shared_ptr<drr::Tensor>& tensor) {
    if (id2owned_tensor_.find(tensor->id()) == id2owned_tensor_.end()) {
      id2owned_tensor_[tensor->id()] = tensor;
    }
    output_tensors.insert(tensor->id());
    return *id2owned_tensor_[tensor->id()];
  }

  std::weak_ptr<OpCall> AnchorNode() const {
    return id2owned_tensor_.at(*output_tensors.begin())->producer();
  }

 private:
  friend class DrrPassContext;

  std::unordered_map<id_type, std::shared_ptr<Tensor>> id2owned_tensor_;
  std::vector<std::shared_ptr<OpCall>> owned_op_call_;
  std::unordered_set<id_type> input_tensors;
  std::unordered_set<id_type> output_tensors;
};

class Constrain {
 public:
  bool operator()(const MatchContext& match_context) const {
    return IsContextMatchConstrain_(match_context);
  }

 private:
  std::function<bool(const MatchContext& match_context)>
      IsContextMatchConstrain_;
};

}  // namespace drr
}  // namespace ir

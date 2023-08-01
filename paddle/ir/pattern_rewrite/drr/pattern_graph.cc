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

#include "paddle/ir/pattern_rewrite/drr/pattern_graph.h"

#include <glog/logging.h>
#include "paddle/ir/pattern_rewrite/drr/api/drr_pass_context.h"

namespace ir {
namespace drr {

const drr::OpCall& PatternGraph::AddOpCall(
    const std::shared_ptr<drr::OpCall>& op_call) {
  owned_op_call_.push_back(op_call);
  for (const auto& input : op_call->inputs()) {
    const auto& tensor_id = input.lock()->id();
    CHECK(id2owned_tensor_.count(tensor_id));
    id2owned_tensor_.at(tensor_id)->AddConsumer(op_call);

    if (input.lock()->producer().use_count() == 0) {
      input_tensors.insert(tensor_id);
    }
    if (output_tensors.find(tensor_id) != output_tensors.end()) {
      output_tensors.erase(tensor_id);
    }
  }
  for (auto& output : op_call->outputs()) {
    const auto& out_tensor_id = output.lock()->id();
    CHECK(id2owned_tensor_.count(out_tensor_id));
    id2owned_tensor_[output.lock()->id()]->set_producer(op_call);
  }
  return *owned_op_call_.back();
}

const drr::Tensor& PatternGraph::AddTensor(
    const std::shared_ptr<drr::Tensor>& tensor) {
  if (id2owned_tensor_.find(tensor->id()) == id2owned_tensor_.end()) {
    id2owned_tensor_[tensor->id()] = tensor;
    output_tensors.insert(tensor->id());
  }
  return *id2owned_tensor_[tensor->id()];
}

drr::Tensor& PatternGraph::AddTmpTensor(
    const std::shared_ptr<drr::Tensor>& tensor) {
  CHECK(id2owned_tensor_.find(tensor->id()) == id2owned_tensor_.end());
  id2owned_tensor_[tensor->id()] = tensor;
  output_tensors.insert(tensor->id());
  return *id2owned_tensor_[tensor->id()];
}

void PatternGraph::UpdateTmpTensor(const id_type& tmp_tensor_id,
                                   const id_type& new_tensor_id) {
  auto tmp_tensor = id2owned_tensor_[tmp_tensor_id];
  tmp_tensor->set_id(new_tensor_id);
  id2owned_tensor_[new_tensor_id] = tmp_tensor;
  id2owned_tensor_.erase(tmp_tensor_id);

  if (input_tensors.find(tmp_tensor_id) != input_tensors.end()) {
    input_tensors.erase(tmp_tensor_id);
    input_tensors.insert(new_tensor_id);
  }
  output_tensors.erase(new_tensor_id);
  if (output_tensors.find(tmp_tensor_id) != output_tensors.end()) {
    output_tensors.erase(tmp_tensor_id);
    output_tensors.insert(new_tensor_id);
  }
}

}  // namespace drr
}  // namespace ir

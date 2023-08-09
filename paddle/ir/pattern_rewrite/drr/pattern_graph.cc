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
#include <iostream>

#include "paddle/ir/pattern_rewrite/drr/api/drr_pattern_context.h"

namespace ir {
namespace drr {

const drr::OpCall &
PatternGraph::AddOpCall(const std::shared_ptr<drr::OpCall> &op_call) {
  owned_op_call_.push_back(op_call);
  for (const auto &input : op_call->inputs()) {
    const auto &tensor_id = input->id();
    CHECK(id2owned_tensor_.count(tensor_id));
    id2owned_tensor_.at(tensor_id)->AddConsumer(op_call.get());

    if (input->producer() == nullptr) {
      input_tensors_.insert(tensor_id);
    }
    if (output_tensors_.find(tensor_id) != output_tensors_.end()) {
      output_tensors_.erase(tensor_id);
    }
  }
  for (auto &output : op_call->outputs()) {
    const auto &out_tensor_id = output->id();
    CHECK(id2owned_tensor_.count(out_tensor_id));
    id2owned_tensor_[output->id()]->set_producer(op_call.get());
  }
  return *owned_op_call_.back();
}

const drr::Tensor &
PatternGraph::AddTensor(const std::shared_ptr<drr::Tensor> &tensor) {
  if (id2owned_tensor_.find(tensor->id()) == id2owned_tensor_.end()) {
    id2owned_tensor_[tensor->id()] = tensor;
    output_tensors_.insert(tensor->id());
  }
  return *id2owned_tensor_[tensor->id()];
}

drr::Tensor &
PatternGraph::AddTmpTensor(const std::shared_ptr<drr::Tensor> &tensor) {
  CHECK(id2owned_tensor_.find(tensor->id()) == id2owned_tensor_.end());
  id2owned_tensor_[tensor->id()] = tensor;
  output_tensors_.insert(tensor->id());
  return *id2owned_tensor_[tensor->id()];
}

void PatternGraph::UpdateTmpTensor(const id_type &tmp_tensor_id,
                                   const id_type &new_tensor_id) {
  if (input_tensors_.count(tmp_tensor_id)) {
    input_tensors_.erase(tmp_tensor_id);
    input_tensors_.insert(new_tensor_id);
  }

  output_tensors_.erase(new_tensor_id);
  if (output_tensors_.count(tmp_tensor_id)) {
    output_tensors_.erase(tmp_tensor_id);
    output_tensors_.insert(new_tensor_id);
  }

  auto tmp_tensor = id2owned_tensor_[tmp_tensor_id];
  id2owned_tensor_.erase(tmp_tensor_id);
  tmp_tensor->set_id(new_tensor_id);
  id2owned_tensor_[new_tensor_id] = tmp_tensor;
}

size_t PatternGraph::CountOfOpCalls() const { return owned_op_call_.size(); }

void PatternGraph::Print() const {
  std::cout << "All Tensors:" << std::endl;
  for (const auto &kv : id2owned_tensor_) {
    std::cout << "  " << kv.first;
  }
  std::cout << "\n" << std::endl;

  std::cout << "Input Tensors:" << std::endl;
  for (const auto &tensor_id : input_tensors_) {
    std::cout << "  " << tensor_id;
  }
  std::cout << "\n" << std::endl;

  std::cout << "Output Tensors:" << std::endl;
  for (const auto &tensor_id : output_tensors_) {
    std::cout << "  " << tensor_id;
  }
  std::cout << "\n" << std::endl;

  std::cout << "OpCalls:" << std::endl;
  for (const auto &op_call : owned_op_call_) {
    std::cout << "  " << op_call->name() << " : ";
    std::cout << "inputs[ ";
    for (const auto &input : op_call->inputs()) {
      std::cout << input->id() << " ";
    }
    std::cout << "], ";

    std::cout << "outputs[ ";
    for (const auto &output : op_call->outputs()) {
      std::cout << output->id() << " ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << std::endl;
}

const OpCall *SourcePatternGraph::AnchorNode() const {
  return id2owned_tensor_.at(*output_tensors_.begin())->producer();
}

} // namespace drr
} // namespace ir

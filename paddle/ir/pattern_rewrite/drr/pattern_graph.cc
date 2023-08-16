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

#include <iostream>
#include <queue>
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/pattern_rewrite/drr/api/drr_pattern_context.h"

namespace ir {
namespace drr {

const drr::OpCall &PatternGraph::AddOpCall(
    const std::shared_ptr<drr::OpCall> &op_call) {
  owned_op_call_.push_back(op_call);
  for (const auto *input : op_call->inputs()) {
    const auto &tensor_id = input->name();
    IR_ENFORCE(id2owned_tensor_.count(tensor_id));
    id2owned_tensor_.at(tensor_id)->AddConsumer(op_call.get());

    if (input->producer() == nullptr) {
      input_tensors_.insert(tensor_id);
    }
    if (output_tensors_.find(tensor_id) != output_tensors_.end()) {
      output_tensors_.erase(tensor_id);
    }
  }
  for (auto &output : op_call->outputs()) {
    const auto &out_tensor_id = output->name();
    IR_ENFORCE(id2owned_tensor_.count(out_tensor_id));
    id2owned_tensor_[output->name()]->set_producer(op_call.get());
  }
  return *owned_op_call_.back();
}

const drr::Tensor &PatternGraph::AddTensor(
    const std::shared_ptr<drr::Tensor> &tensor) {
  if (id2owned_tensor_.find(tensor->name()) == id2owned_tensor_.end()) {
    id2owned_tensor_[tensor->name()] = tensor;
    output_tensors_.insert(tensor->name());
  }
  return *id2owned_tensor_[tensor->name()];
}

drr::Tensor &PatternGraph::AddTmpTensor(
    const std::shared_ptr<drr::Tensor> &tensor) {
  IR_ENFORCE(id2owned_tensor_.count(tensor->name()) == 0);
  id2owned_tensor_[tensor->name()] = tensor;
  output_tensors_.insert(tensor->name());
  return *id2owned_tensor_[tensor->name()];
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
  tmp_tensor->set_name(new_tensor_id);
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

  for (const auto &op_call : owned_op_call_) {
    std::cout << "  " << op_call->name() << " : ";
    std::cout << "inputs[ ";
    for (const auto *input : op_call->inputs()) {
      std::cout << input->name() << " ";
    }
    std::cout << "], ";

    std::cout << "outputs[ ";
    for (const auto &output : op_call->outputs()) {
      std::cout << output->name() << " ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << std::endl;
}

const OpCall *SourcePatternGraph::AnchorNode() const {
  return id2owned_tensor_.at(*output_tensors_.begin())->producer();
}

void GraphTopo::WalkGraphNodesTopoOrder(
    const std::function<void(const OpCall &)> &VisitNode) const {
  // graph data
  const std::unordered_set<ir::drr::PatternGraph::id_type> &inputs_tensor =
      graph_->input_tensors();
  const std::unordered_map<ir::drr::PatternGraph::id_type,
                           std::shared_ptr<ir::drr::Tensor>> &id2owned_tensor =
      graph_->id2owend_tensor();
  const std::vector<std::shared_ptr<OpCall>> &owend_opcall =
      graph_->owned_op_call();

  std::queue<const ir::drr::OpCall *> opcall_queue;
  std::unordered_map<const ir::drr::OpCall *, std::unordered_set<std::string>>
      opcall_dependent;

  // init opcall_dependent;
  for (const std::shared_ptr<OpCall> &opcall_sptr : owend_opcall) {
    if (opcall_sptr.get()->inputs().empty()) {  // opcall inputs is empty
      opcall_queue.push(opcall_sptr.get());
    } else {
      for (const auto &pre_depd_tensor : opcall_sptr.get()->inputs()) {
        opcall_dependent[opcall_sptr.get()].insert(pre_depd_tensor->name());
      }
    }
  }

  // init queue
  for (const auto &tensor_id : inputs_tensor) {
    const std::string &tensor_name =
        id2owned_tensor.at(tensor_id).get()->name();

    for (const auto &tensor_comsumer :
         id2owned_tensor.at(tensor_id).get()->consumers()) {
      opcall_dependent[tensor_comsumer].erase(tensor_name);
      if (opcall_dependent[tensor_comsumer].empty()) {
        opcall_queue.push(tensor_comsumer);
      }
    }
  }

  while (!opcall_queue.empty()) {
    const ir::drr::OpCall *opcall = opcall_queue.front();
    opcall_queue.pop();
    VisitNode(*opcall);

    // update opcall_dependent
    for (const auto &output_tensor : opcall->outputs()) {
      for (const auto &tensor_comsumer : output_tensor->consumers()) {
        opcall_dependent[tensor_comsumer].erase(output_tensor->name());
        if (opcall_dependent[tensor_comsumer].empty()) {
          opcall_queue.push(tensor_comsumer);
        }
      }
    }
  }
}

}  // namespace drr
}  // namespace ir

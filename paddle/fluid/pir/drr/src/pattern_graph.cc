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

#include "paddle/fluid/pir/drr/src/pattern_graph.h"

#include <queue>

#include "paddle/common/errors.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/phi/core/enforce.h"

namespace paddle::drr {

const drr::OpCall &PatternGraph::AddOpCall(
    const std::shared_ptr<drr::OpCall> &op_call) {
  owned_op_call_.push_back(op_call);
  for (const auto *input : op_call->inputs()) {
    const auto &tensor_name = input->name();
    PADDLE_ENFORCE_NE(
        id2owned_tensor_.count(tensor_name),
        0,
        common::errors::NotFound("Not found tensor."
                                 "The input tensor [%s] must exist "
                                 "in pattern graph to be obtained.",
                                 tensor_name));
    id2owned_tensor_.at(tensor_name)->AddConsumer(op_call.get());

    if (input->producer() == nullptr) {
      input_tensors_.insert(tensor_name);
    }
    if (output_tensors_.find(tensor_name) != output_tensors_.end()) {
      output_tensors_.erase(tensor_name);
    }
  }
  for (auto &output : op_call->outputs()) {
    const auto &out_tensor_name = output->name();
    PADDLE_ENFORCE_NE(
        id2owned_tensor_.count(out_tensor_name),
        0,
        common::errors::NotFound("Not found tensor."
                                 "The output tensor [%s] must exist "
                                 "in pattern graph to be obtained.",
                                 out_tensor_name));
    id2owned_tensor_[output->name()]->set_producer(op_call.get());
  }
  return *owned_op_call_.back();
}

drr::Tensor &PatternGraph::AddTensor(
    const std::shared_ptr<drr::Tensor> &tensor) {
  if (id2owned_tensor_.find(tensor->name()) == id2owned_tensor_.end()) {
    id2owned_tensor_[tensor->name()] = tensor;
    output_tensors_.insert(tensor->name());
  }
  return *id2owned_tensor_[tensor->name()];
}

drr::Tensor &PatternGraph::AddTmpTensor(
    const std::shared_ptr<drr::Tensor> &tensor) {
  PADDLE_ENFORCE_EQ(id2owned_tensor_.count(tensor->name()),
                    0,
                    common::errors::AlreadyExists(
                        "Tensor already exists."
                        "The tensor [%s] must not exist in pattern graph.",
                        tensor->name()));
  id2owned_tensor_[tensor->name()] = tensor;
  output_tensors_.insert(tensor->name());
  return *id2owned_tensor_[tensor->name()];
}

void PatternGraph::UpdateTmpTensor(const std::string &tmp_tensor_name,
                                   const std::string &new_tensor_name) {
  if (input_tensors_.count(tmp_tensor_name)) {
    input_tensors_.erase(tmp_tensor_name);
    input_tensors_.insert(new_tensor_name);
  }

  output_tensors_.erase(new_tensor_name);
  if (output_tensors_.count(tmp_tensor_name)) {
    output_tensors_.erase(tmp_tensor_name);
    output_tensors_.insert(new_tensor_name);
  }

  auto tmp_tensor = id2owned_tensor_[tmp_tensor_name];
  id2owned_tensor_.erase(tmp_tensor_name);
  tmp_tensor->set_name(new_tensor_name);
  id2owned_tensor_[new_tensor_name] = tmp_tensor;
}

size_t PatternGraph::CountOfOpCalls() const { return owned_op_call_.size(); }

std::unordered_set<const OpCall *> SourcePatternGraph::OutputNodes() const {
  std::unordered_set<const OpCall *> output_op_set;
  for (const auto &output_tensor : output_tensors_) {
    OpCall *output_op_candidate =
        id2owned_tensor_.at(output_tensor)->producer();
    if (std::all_of(output_op_candidate->outputs().begin(),
                    output_op_candidate->outputs().end(),
                    [this](const Tensor *output) -> bool {
                      return this->output_tensors().count(output->name());
                    }))
      output_op_set.insert(output_op_candidate);
  }
  if (output_op_set.empty()) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unable to find a valid anchor in drr's source result pattern!"));
  }
  return output_op_set;
}

void ResultPatternGraph::AssignTensor(const Tensor &from, const Tensor &to) {
  if (to.producer() == nullptr) {
    input_tensors_.insert(to.name());
  }
  output_tensors_.erase(to.name());
  PADDLE_ENFORCE_EQ(output_tensors_.count(from.name()),
                    1,
                    common::errors::PreconditionNotMet(
                        "The Tensor (%s) which be assigned must "
                        "be the output of result pattern graph.",
                        from.name()));
  tensor_assign_map_[from.name()] = to.name();
}

void GraphTopo::WalkGraphNodesTopoOrder(
    const std::function<void(const OpCall &)> &VisitNode) const {
  // graph data
  const std::unordered_set<std::string> &inputs_tensor =
      graph_->input_tensors();
  const std::unordered_map<std::string, std::shared_ptr<Tensor>>
      &id2owned_tensor = graph_->id2owned_tensor();
  const std::vector<std::shared_ptr<OpCall>> &owned_opcall =
      graph_->owned_op_call();

  std::queue<const OpCall *> opcall_queue;
  std::unordered_map<const OpCall *, std::unordered_set<std::string>>
      opcall_dependent;

  // init opcall_dependent
  for (const std::shared_ptr<OpCall> &opcall_sptr : owned_opcall) {
    if (opcall_sptr->inputs().empty()) {  // opcall inputs is empty
      opcall_queue.push(opcall_sptr.get());
    } else {
      for (const auto &pre_depd_tensor : opcall_sptr->inputs()) {
        opcall_dependent[opcall_sptr.get()].insert(pre_depd_tensor->name());
      }
    }
  }

  // init queue
  for (const auto &tensor_name : inputs_tensor) {
    PADDLE_ENFORCE_NE(
        id2owned_tensor.count(tensor_name),
        0,
        common::errors::NotFound("Not found tensor."
                                 "The input tensor [%s] must exists "
                                 "in pattern graph to be obtained.",
                                 tensor_name));
    for (const auto &tensor_consumer :
         id2owned_tensor.at(tensor_name).get()->consumers()) {
      opcall_dependent[tensor_consumer].erase(tensor_name);
      if (opcall_dependent[tensor_consumer].empty()) {
        opcall_queue.push(tensor_consumer);
      }
    }
  }

  while (!opcall_queue.empty()) {
    const OpCall *opcall = opcall_queue.front();
    opcall_queue.pop();
    VisitNode(*opcall);

    // update opcall_dependent
    for (const auto &output_tensor : opcall->outputs()) {
      for (const auto &tensor_consumer : output_tensor->consumers()) {
        opcall_dependent[tensor_consumer].erase(output_tensor->name());
        if (opcall_dependent[tensor_consumer].empty()) {
          opcall_queue.push(tensor_consumer);
        }
      }
    }
  }
}

std::ostream &operator<<(std::ostream &os, const PatternGraph &pattern_graph) {
  os << "\nAll Tensors:\n";
  for (const auto &kv : pattern_graph.id2owned_tensor()) {
    os << "  " << kv.first;
  }
  os << "\n\n";

  os << "Input Tensors:\n";
  for (const auto &tensor_name : pattern_graph.input_tensors()) {
    os << "  " << tensor_name;
  }
  os << "\n\n";

  os << "Output Tensors:\n";
  for (const auto &tensor_name : pattern_graph.output_tensors()) {
    os << "  " << tensor_name;
  }
  os << "\n\n";

  for (const auto &op_call : pattern_graph.owned_op_call()) {
    os << "  " << op_call->name() << " : ";
    os << "inputs[ ";
    for (const auto *input : op_call->inputs()) {
      os << input->name() << " ";
    }
    os << "], ";

    os << "outputs[ ";
    for (const auto &output : op_call->outputs()) {
      os << output->name() << " ";
    }
    os << "]\n";
  }
  os << "\n";
  return os;
}

}  // namespace paddle::drr

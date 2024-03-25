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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace drr {

class Constraint;
class MatchContext;
class OpCall;
class Tensor;

class PatternGraph {
 public:
  virtual ~PatternGraph() {}

  const drr::OpCall& AddOpCall(const std::shared_ptr<drr::OpCall>& op_call);

  drr::Tensor& AddTensor(const std::shared_ptr<drr::Tensor>& tensor);

  drr::Tensor& AddTmpTensor(const std::shared_ptr<drr::Tensor>& tensor);

  void UpdateTmpTensor(const std::string& tmp_tensor_name,
                       const std::string& new_tensor_name);

  const std::unordered_set<std::string>& input_tensors() const {
    return input_tensors_;
  }

  const std::unordered_set<std::string>& output_tensors() const {
    return output_tensors_;
  }

  size_t CountOfOpCalls() const;

  const std::vector<std::shared_ptr<OpCall>>& owned_op_call() const {
    return owned_op_call_;
  }

  const std::unordered_map<std::string, std::shared_ptr<Tensor>>&
  id2owned_tensor() const {
    return id2owned_tensor_;
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<Tensor>> id2owned_tensor_;
  std::vector<std::shared_ptr<OpCall>> owned_op_call_;
  std::unordered_set<std::string> input_tensors_;
  std::unordered_set<std::string> output_tensors_;
};

std::ostream& operator<<(std::ostream& os, const PatternGraph& pattern_graph);

class SourcePatternGraph : public PatternGraph {
 public:
  std::unordered_set<const OpCall*> OutputNodes() const;

 private:
  friend class DrrPatternContext;
};

class ResultPatternGraph : public PatternGraph {
 public:
  void AssignTensor(const Tensor& from, const Tensor& to);

  const std::unordered_map<std::string, std::string>& tensor_assign_map()
      const {
    return tensor_assign_map_;
  }

 private:
  std::unordered_map<std::string, std::string> tensor_assign_map_;
};

class GraphTopo {
 public:
  explicit GraphTopo(const PatternGraph* graph) : graph_(graph) {}

  void WalkGraphNodesTopoOrder(
      const std::function<void(const OpCall&)>& VisitNode) const;

 private:
  const PatternGraph* graph_;
};

}  // namespace drr
}  // namespace paddle

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

namespace ir {
namespace drr {

class Constrain;
class MatchContext;
class OpCall;
class Tensor;

class PatternGraph {
 public:
  using id_type = std::string;

  const drr::OpCall& AddOpCall(const std::shared_ptr<drr::OpCall>& op_call);

  const drr::Tensor& AddTensor(const std::shared_ptr<drr::Tensor>& tensor);

  drr::Tensor& AddTmpTensor(const std::shared_ptr<drr::Tensor>& tensor);

  void UpdateTmpTensor(const id_type& tmp_tensor_id,
                       const id_type& new_tensor_id);

 protected:
  std::unordered_map<id_type, std::shared_ptr<Tensor>> id2owned_tensor_;
  std::vector<std::shared_ptr<OpCall>> owned_op_call_;
  std::unordered_set<id_type> input_tensors;
  std::unordered_set<id_type> output_tensors;
};

class SourcePatternGraph : public PatternGraph {
 public:
  std::weak_ptr<OpCall> AnchorNode() const {
    return id2owned_tensor_.at(*output_tensors.begin())->producer();
  }

 private:
  friend class DrrPassContext;
};

class ResultPatternGraph : public PatternGraph {};

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

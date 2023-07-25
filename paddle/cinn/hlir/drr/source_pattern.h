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
#include <functional>
#include <memory>

#include "paddle/cinn/hlir/drr/drr_pass_context.h"
#include "visit_status.h"

namespace cinn {
namespace hlir {
namespace drr {

class Constrain;
class SourcePatternGraph;
class MatchContext;
class OpCall;
class Tensor;

using id_type = std::string;

class SourcePatternGraph {
 public:
  void AddOpCall(const std::shared_ptr<drr::OpCall>& op_call) {
    owned_op_call_.insert(op_call);
  }

 private:
  friend class DrrPassContext;

  std::unordered_map<id_type, std::shared_ptr<Tensor>> id2owned_tensor_;
  std::unordered_set<std::shared_ptr<OpCall>> owned_op_call_;
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
}  // namespace hlir
}  // namespace cinn

// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {

static const char kParamScopeAttr[] = "__param_scope__";
static const char kFuseStatisAttr[] = "__fuse_statis__";

class FusePassBase : public Pass {
 public:
  void Init(const std::string& repr, Graph* graph) const {
    repr_ = repr;
    graph_ = graph;
  }

  Scope* param_scope() const {
    PADDLE_ENFORCE(graph_->Has(kParamScopeAttr));
    return graph_->Get<framework::Scope*>(kParamScopeAttr);
  }

  void AddStatis(int count_of_fused) const {
    PADDLE_ENFORCE(graph_);
    PADDLE_ENFORCE(!repr_.empty());
    if (!graph_->Has(kFuseStatisAttr)) {
      graph_->Set(kFuseStatisAttr, new std::unordered_map<std::string, int>);
    }
    auto& info =
        graph_->Get<std::unordered_map<std::string, int>>(kFuseStatisAttr);
    info[repr_] = count_of_fused;
  }

  virtual ~FusePassBase() {}

 protected:
  mutable Graph* graph_;
  mutable std::string repr_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

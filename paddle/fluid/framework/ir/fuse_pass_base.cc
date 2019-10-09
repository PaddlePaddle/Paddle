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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include <unordered_map>

namespace paddle {
namespace framework {
namespace ir {

void FusePassBase::Init(const std::string& repr, Graph* graph) const {
  repr_ = repr;
  graph_ = graph;
}

Scope* FusePassBase::param_scope() const {
  PADDLE_ENFORCE(graph_->Has(kParamScopeAttr));
  auto& scope = graph_->Get<framework::Scope>(kParamScopeAttr);
  return &scope;
}

void FusePassBase::AddStatis(int count_of_fused) const {
  PADDLE_ENFORCE(graph_);
  PADDLE_ENFORCE(!repr_.empty());
  if (!graph_->Has(kFuseStatisAttr)) {
    graph_->Set(kFuseStatisAttr, new std::unordered_map<std::string, int>);
  }
  auto& info =
      graph_->Get<std::unordered_map<std::string, int>>(kFuseStatisAttr);
  info[repr_] = count_of_fused;
}

FuseOptions FusePassBase::FindFuseOption(const Node& node1,
                                         const Node& node2) const {
#ifdef PADDLE_WITH_MKLDNN
  bool node1_mkldnn = node1.Op()->HasAttr("use_mkldnn") &&
                      boost::get<bool>(node1.Op()->GetAttr("use_mkldnn"));
  bool node2_mkldnn = node2.Op()->HasAttr("use_mkldnn") &&
                      boost::get<bool>(node2.Op()->GetAttr("use_mkldnn"));
  if (node1_mkldnn && node2_mkldnn)
    return FUSE_MKLDNN;
  else if (!node1_mkldnn && !node2_mkldnn)
    return FUSE_NATIVE;
  else
    return DO_NOT_FUSE;
#else
  return FUSE_NATIVE;
#endif
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

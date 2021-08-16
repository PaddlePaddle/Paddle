// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/autograd_meta.h"
/**
 * Implementation of GradNodeBase and Edge.
**/
namespace egr {

void GradNodeBase::AddEdge(const std::vector<AutogradMeta*>& metas) {
  VLOG(0) << "Add Edge for tensors";
  for (const auto& meta : metas) {
    adj_edges_.emplace_back(meta->GetMutableGradNode(), meta->OutRank());
  }
}

const std::vector<Edge>& GradNodeBase::GetEdges() const { return adj_edges_; }

void GradNodeBase::RecordStopGradient(
    const std::vector<AutogradMeta*>& ins_autograds) {
  for (size_t i = 0; i < ins_autograds.size(); ++i) {
    bwd_stop_gradients_.emplace_back(ins_autograds->NumericStopGradient());
  }
}

}  // namespace egr

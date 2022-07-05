// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

using GraphWithStats = std::pair<ir::Graph*, int>;

class FCResidualConnectionMKLDNNFusePass : public FusePassBase {
 private:
  GraphWithStats FuseFC(const std::string& name_scope,
                        const GraphWithStats& graph_with_stats,
                        bool fc_as_x) const;

 public:
  FCResidualConnectionMKLDNNFusePass();
  virtual ~FCResidualConnectionMKLDNNFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const;

  static bool HasFusedActivation(Node* fc_node) {
    return !(
        fc_node->Op()->GetAttrIfExists<std::string>("activation_type").empty());
  }

  const std::string name_scope_{"fc_elementwise_add_mkldnn_fuse"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

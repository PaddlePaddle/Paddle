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
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {
/*
* Use the group attribute to fuse conv operators.
*/
class ConvFuseWithGroupPass : public FusePassBase {
 public:
  virtual ~ConvFuseWithGroupPass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;
  std::string name_scope_{"conv_fuse_with_group_pass"};

 private:
  Node* IsSameSingleInput(std::vector<Node*>& nodes,  // NOLINT
                          std::string type) const;
  void GetSpeicalOpNodes(std::vector<Node*>& nodes, std::string type,  // NOLINT
                         std::vector<Node*>* dst_nodes) const;
  Node* GetSpeicalVarNode(std::vector<Node*>& nodes,  // NOLINT
                          std::string name) const;
  void SortNodes(std::vector<Node*>& nodes) const;  // NOLINT
  Node* GetConvWeightBiasNodes(const std::vector<Node*>& nodes,
                               std::vector<Node*>& weights_node,  // NOLINT
                               std::vector<Node*>& biases_node,   // NOLINT
                               int block, int group = 1, int layer = 1,
                               std::string type = "conv") const;
  Node* CreateConvOpWithGroup(const std::unique_ptr<ir::Graph>& graph,
                              Scope* scope, Node* input_mode, Node* conv_node,
                              const std::vector<Node*>& weights_node,
                              const std::vector<Node*>& biases_node,
                              int block = 0, int group = 1, int layer = 1,
                              std::string type = "conv") const;
  Node* CreateConvVarNode(const std::unique_ptr<ir::Graph>& graph, Scope* scope,
                          const std::vector<Node*>& nodes, int block, int group,
                          int layer, bool persistable, std::string type,
                          std::string usage, int64_t axis = 0) const;
  Node* CreateVarNode(const std::unique_ptr<ir::Graph>& graph, Scope* scope,
                      std::string name, DDim dims = make_ddim({1}),
                      bool persistable = false) const;
  Node* CreateResiualNetWithGroup(const std::unique_ptr<ir::Graph>& graph,
                                  Scope* scope,
                                  const std::vector<Node*>& conv_nodes,
                                  const std::vector<Node*>& eltwise_add_nodes,
                                  Node* input_mode, int block, int group,
                                  bool is_project = false) const;
  Node* GetKeyEltwiseAddOpNode(const std::vector<Node*>& nodes, int block,
                               int group) const;
  Node* CreateEltwiseAddOp(const std::unique_ptr<ir::Graph>& graph,
                           Scope* scope, Node* input_x_mode, Node* input_y_mode,
                           Node* eltwise_add_node, int block, int group) const;
  int GetConvOutputChannelsNum(Scope* scope) const;
  void RedirectSplitPoolOpNodes(const std::unique_ptr<ir::Graph>& graph,
                                Scope* scope,
                                std::vector<Node*>& split_nodes,  // NOLINT
                                std::vector<Node*>& pool_nodes,   // NOLINT
                                Node* conv_mode, int conv_channels_num) const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

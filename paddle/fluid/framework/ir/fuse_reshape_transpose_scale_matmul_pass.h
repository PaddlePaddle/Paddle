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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

#define MATMUL_COMPUTE_DIMENSION 2

enum {
  RESHAPE_INPUT = 0,
  RESHAPE_OP,
  TRANSPOSE_INPUT,
  TRANSPOSE_OP,
  SCALE_INPUT,
  SCALE_OP,
  SCALE_OUTPUT,
  MAX_MATMUL_NODES,
  MATMUL_INPUT = SCALE_INPUT,
  TRANSPOSE_REVERSE_INPUT = MATMUL_INPUT,
  TRANSPOSE_REVERSE_OP = TRANSPOSE_OP,
  RESHAPE_REVERSE_INPUT = TRANSPOSE_INPUT,
  RESHAPE_REVERSE_OP = RESHAPE_OP,
  RESHAPE_REVERSE_OUTPUT = RESHAPE_INPUT
};

/*
 * Fuse Matmul+Reshape+Transpose+Scale operators to a Matmul.
 */

class ReshapeTransposeScaleMatmulFusePass : public FusePassBase {
 public:
  virtual ~ReshapeTransposeScaleMatmulFusePass() {}

 protected:
  void GetSpeicalOpNodes(const std::vector<Node*>& nodes,
                         std::string type,  // NOLINT
                         std::vector<Node*>* dst_nodes) const;
  int ReConfigureMatMulOp(ir::Graph* graph,
                          std::multimap<Node*, std::vector<Node*>>&
                              matmul_nodes_map) const;              // NOLINT
  bool IsEnableFuse(std::vector<Node*>& nodes, bool is_out) const;  // NOLINT
  void UpdateFusedNode(ir::Graph* graph, Node* matmul_op,
                       std::vector<Node*>& nodes) const;  // NOLINT
  bool IsEnableReplace(std::vector<Node*>& nodes) const;  // NOLINT
  int DetectFuseNodes(const std::vector<Node*>& nodes,
                      std::multimap<Node*, std::vector<Node*>>&
                          matmul_nodes_map) const;  // NOLINT
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"reshape_transpose_scale_matmul_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

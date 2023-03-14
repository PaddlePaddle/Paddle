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

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct MaxPoolingPadZeroXPUPattern : public PatternBase {
  MaxPoolingPadZeroXPUPattern(PDPattern* pattern,
                              const std::string& name_scope,
                              const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(pre_op);
  PATTERN_DECL_NODE(pool2d);
  // declare variable node's name
  PATTERN_DECL_NODE(pre_out);

 private:
  std::string act_type_;
};

MaxPoolingPadZeroXPUPattern::MaxPoolingPadZeroXPUPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope), act_type_(act_type) {
  auto* pre_op = pattern->NewNode(pre_op_repr())->assert_is_op(act_type_);
  auto* pre_op_out = pattern->NewNode(pre_out_repr())
                         ->assert_is_op_output(act_type_, "Out")
                         ->assert_is_op_input("pool2d", "X");
  auto* max_pool = pattern->NewNode(pool2d_repr())
                       ->assert_op_attr<std::string>("pooling_type", "max");
  pre_op->LinksTo({pre_op_out});
  max_pool->LinksFrom({pre_op_out});
}

}  // namespace patterns

/* Detect Max Pooling which can pad zero instead of pad -inf    */
/* For example:                                                 */
/* graph[1]: sub block                                          */
/*                  relu/sigmoid/relu6...                       */
/*                       |                                      */
/*                       |                                      */
/*                   max_pooling                                */
class MaxPoolingXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph, const std::string& act_type) const;

  const std::string name_scope_{"max_pooling_pad_zero_xpu_fuse_pass"};
};

void MaxPoolingXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto act_type : {
           "relu",
           "sigmoid",
           "hard_sigmoid",
           "relu6",
           "",
       }) {
    found_subgraph_count += ApplyImpl(graph, act_type);
  }

  AddStatis(found_subgraph_count);
}

int MaxPoolingXPUFusePass::ApplyImpl(ir::Graph* graph,
                                     const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::MaxPoolingPadZeroXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle MaxPoolingXPUFusePASS fuse";
    GET_IR_NODE(pool2d);
    GET_IR_NODE(pre_out);
    // pool2d->Op()->SetAttr("pad_zero", true);
    // another way
    auto* block = pool2d->Op()->Block();
    // set pad_zero to true
    framework::OpDesc max_pool_op_desc(block);
    // max_pool_op_desc.SetAttr("pad_zero", true);
    auto* max_pool = graph->CreateOpNode(&max_pool_op_desc);
    IR_NODE_LINK_TO(pre_out, max_pool);
    // delete original pool2d node
    std::unordered_set<const Node*> delete_nodes;
    delete_nodes = {pool2d};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(max_pooling_pad_zero_xpu_fuse_pass,
              paddle::framework::ir::MaxPoolingXPUFusePass);

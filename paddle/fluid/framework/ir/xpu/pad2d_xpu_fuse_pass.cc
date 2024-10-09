// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

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

struct Pad2dXpuFusePattern : public PatternBase {
  Pad2dXpuFusePattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(unsqueeze);
  PATTERN_DECL_NODE(pad3d);
  PATTERN_DECL_NODE(squeeze);
  // declare variable node's name
  PATTERN_DECL_NODE(unsqueeze_input);
  PATTERN_DECL_NODE(unsqueeze_out);
  PATTERN_DECL_NODE(pad3d_out);
  PATTERN_DECL_NODE(squeeze_out);
};

Pad2dXpuFusePattern::Pad2dXpuFusePattern(PDPattern* pattern,
                                         const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  // unsqueeze
  auto unsqueeze =
      pattern->NewNode(unsqueeze_repr())->assert_is_op("unsqueeze2");

  auto unsqueeze_input = pattern->NewNode(unsqueeze_input_repr())
                             ->assert_is_op_input("unsqueeze2", "X")
                             ->AsInput();

  auto unsqueeze_out = pattern->NewNode(unsqueeze_out_repr())
                           ->assert_is_op_output("unsqueeze2", "Out");

  unsqueeze->LinksFrom({unsqueeze_input}).LinksTo({unsqueeze_out});

  // pad3d
  auto pad3d =
      pattern->NewNode(pad3d_repr())
          ->assert_is_op("pad3d")
          ->assert_more([&](Node* node) {
            if (node->Op()->HasAttr("paddings", true) &&
                node->Op()->GetAttrType("paddings", true)) {
              auto paddings =
                  node->Op()->GetAttrIfExists<std::vector<int>>("paddings");
              if (paddings.size() == 6 && paddings[4] == 0 &&
                  paddings[5] == 0) {
                return true;
              } else {
                return false;
              }
            } else {
              return false;
            }
          });

  // pad3d
  auto pad3d_out =
      pattern->NewNode(pad3d_out_repr())->assert_is_op_output("pad3d", "Out");
  pad3d_out->assert_is_op_input("squeeze2", "X");
  unsqueeze_out->assert_is_op_input("pad3d", "X");
  pad3d->LinksFrom({unsqueeze_out}).LinksTo({pad3d_out});

  // squeeze
  auto squeeze = pattern->NewNode(squeeze_repr())->assert_is_op("squeeze2");
  auto squeeze_out = pattern->NewNode(squeeze_out_repr())
                         ->assert_is_op_output("squeeze2", "Out")
                         ->AsOutput();
  squeeze->LinksFrom({pad3d_out}).LinksTo({squeeze_out});
}

}  // namespace patterns

// Delete redundant squeeze/unsqueeze op
/*
For example:
graph:
      Input
        |
        |
    unsqueeze
        |
        |
   unsqueeze out
        |
        |
      pad3d
        |
        |
    pad3d out
        |
        |
      squeeze
        |
        |
      Output
------------------------------------------------------
After the pass is applied:
      Input
        |
        |
    pad2d_xpu
        |
        |
      Output
*/

class Pad2dXpuFusePattern : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int FusedUnsqueezePadOps(ir::Graph* graph) const;

  const std::string name_scope_{"pad2d_xpu_fuse_pass"};
};

void Pad2dXpuFusePattern::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  int found_subgraph_count = 0;
  found_subgraph_count += FusedUnsqueezePadOps(graph);
  AddStatis(found_subgraph_count);
}

int Pad2dXpuFusePattern::FusedUnsqueezePadOps(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Pad2dXpuFusePattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle squeeze activation unsqueeze elimination.";

    GET_IR_NODE_FROM_SUBGRAPH(unsqueeze_input, unsqueeze_input, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(unsqueeze_out, unsqueeze_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pad3d_out, pad3d_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(squeeze_out, squeeze_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(unsqueeze, unsqueeze, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pad3d, pad3d, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(squeeze, squeeze, pattern);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    // pad2d
    auto* block = pad3d->Op()->Block();
    framework::OpDesc pad2d_op_desc(block);
    pad2d_op_desc.SetType("pad2d_xpu");

    // get pad3d's data_format
    std::string data_format =
        pad3d->Op()->GetAttrIfExists<std::string>("data_format");
    // Judge unsqueeze && squeeze,  if axes is same, then eliminate
    std::vector<int> unsqueeze_axes =
        PADDLE_GET_CONST(std::vector<int>, unsqueeze->Op()->GetAttr("axes"));
    std::vector<int> squeeze_axes =
        PADDLE_GET_CONST(std::vector<int>, squeeze->Op()->GetAttr("axes"));
    bool elimination = (unsqueeze_axes.size() == 1 &&
                        ((data_format == "NCDHW" && unsqueeze_axes[0] == 2 &&
                          squeeze_axes[0] == 2) ||
                         (data_format == "NDHWC" && unsqueeze_axes[0] == 1 &&
                          squeeze_axes[0] == 1)));
    if (!elimination) return;

    // set pad3d's data_format to pad2d's data_format
    if (data_format == "NCDHW") {
      data_format = "NCHW";
    } else if (data_format == "NDHWC") {
      data_format = "NHWC";
    }
    auto paddings = pad3d->Op()->GetAttrIfExists<std::vector<int>>("paddings");
    std::vector<int> paddings_new(4);
    for (int index = 0; index < 4; ++index) {
      paddings_new[index] = paddings[index];
    }

    pad2d_op_desc.SetInput("x", {unsqueeze_input->Var()->Name()});
    pad2d_op_desc.SetAttr("data_format", data_format);
    pad2d_op_desc.SetAttr("mode", pad3d->Op()->GetAttr("mode"));
    pad2d_op_desc.SetAttr("pad_value", pad3d->Op()->GetAttr("value"));
    pad2d_op_desc.SetAttr("paddings", paddings_new);
    pad2d_op_desc.SetOutput("out", {squeeze_out->Var()->Name()});

    auto* pad2d_op = graph->CreateOpNode(&pad2d_op_desc);
    pad2d_op_desc.Flush();
    IR_NODE_LINK_TO(unsqueeze_input, pad2d_op);
    IR_NODE_LINK_TO(pad2d_op, squeeze_out);

    std::unordered_set<const Node*> delete_nodes{
        unsqueeze, unsqueeze_out, pad3d, pad3d_out, squeeze};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(pad2d_xpu_fuse_pass, paddle::framework::ir::Pad2dXpuFusePattern);
REGISTER_PASS_CAPABILITY(pad2d_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "pad2d_xpu", 0));

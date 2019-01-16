// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_relu_depthwise_conv_pass.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FuseReluDepthwiseConvPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(4) << "FuseReluDepthwiseConvPass::ApplyImpl";
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("relu_depthwise_conv", graph.get());
  /*
           x ---act--> y ---layer-> z
            +----------+
            ↓          ↓
    x' <--act'--- y' <-layer'--- z'

    fuse to:

           x ---act-layer-> z
           |
           ↓
    x' <--act-layer'--- z'

  */

  GraphPatternDetector gpd;
  auto *pattern = gpd.mutable_pattern();
  std::string act_type = "relu";
  std::string layer_type = "depthwise_conv2d";
  auto *x = pattern->NewNode("x")->AsInput();
  auto *y = pattern->NewNode("y")->AsIntermediate();
  auto *z = pattern->NewNode("z")->AsOutput();
  auto *xg = pattern->NewNode("xg")->AsOutput();
  auto *yg = pattern->NewNode("yg")->AsIntermediate();
  auto *zg = pattern->NewNode("zg")->AsInput();

  auto *act = pattern->NewNode("act")->assert_is_op(act_type);
  auto *layer = pattern->NewNode("layer")->assert_is_op(layer_type);
  auto *act_g = pattern->NewNode("act_g")->assert_is_op(act_type + "_grad");
  auto *layer_g =
      pattern->NewNode("layer_g")->assert_is_op(layer_type + "_grad");

  act->LinksFrom({x}).LinksTo({y});
  layer->LinksFrom({y}).LinksTo({z});
  layer_g->LinksFrom({y, zg}).LinksTo({yg});
  act_g->LinksFrom({y, yg}).LinksTo({xg});

  int count = 0;
  std::unordered_set<const Node *> need_removed_nodes;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseReluDepthwiseConv fuse";
    // 1. turn on fuse option
    auto *layer_op = subgraph.at(layer)->Op();
    layer_op->SetAttr("use_cudnn", false);
    layer_op->SetAttr("fuse_relu_before_depthwise_conv", true);

    auto *layer_g_op = subgraph.at(layer_g)->Op();
    layer_g_op->SetAttr("use_cudnn", false);
    layer_g_op->SetAttr("fuse_relu_before_depthwise_conv", true);
    // 2. connect x to layer and layer_g, layer_g to xg
    auto *y_var = subgraph.at(y)->Var();
    auto *x_var = subgraph.at(x)->Var();
    auto *yg_var = subgraph.at(yg)->Var();
    auto *xg_var = subgraph.at(xg)->Var();

    PADDLE_ENFORCE_EQ(layer_op->Input("Input").size(), 1);
    PADDLE_ENFORCE_EQ(layer_op->Input("Input")[0], y_var->Name());
    layer_op->SetInput("Input", {x_var->Name()});
    subgraph.at(layer)->inputs.push_back(subgraph.at(x));
    subgraph.at(x)->outputs.push_back(subgraph.at(layer));
    VLOG(4) << "replace " << y_var->Name() << " -> " << x_var->Name();

    PADDLE_ENFORCE_EQ(layer_g_op->Input("Input").size(), 1);
    PADDLE_ENFORCE_EQ(layer_g_op->Input("Input")[0], y_var->Name());
    layer_g_op->SetInput("Input", {x_var->Name()});
    subgraph.at(layer_g)->inputs.push_back(subgraph.at(x));
    subgraph.at(x)->outputs.push_back(subgraph.at(layer_g));

    PADDLE_ENFORCE_EQ(layer_g_op->Output(GradVarName("Input")).size(), 1);
    PADDLE_ENFORCE_EQ(layer_g_op->Output(GradVarName("Input"))[0],
                      yg_var->Name());
    layer_g_op->SetOutput(GradVarName("Input"), {xg_var->Name()});
    subgraph.at(layer_g)->outputs.push_back(subgraph.at(xg));
    subgraph.at(xg)->inputs.push_back(subgraph.at(layer_g));
    VLOG(4) << "replace " << yg_var->Name() << " -> " << xg_var->Name();

    // 3. delete y, yg, act, act_g
    need_removed_nodes.insert({subgraph.at(y), subgraph.at(yg),
                               subgraph.at(act), subgraph.at(act_g)});
    count++;
  };
  // for (auto *node : graph->Nodes())
  //  VLOG(4) << node->Name();
  gpd(graph.get(), handler);
  GraphSafeRemoveNodes(graph.get(), need_removed_nodes);
  AddStatis(count);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_relu_depthwise_conv_pass,
              paddle::framework::ir::FuseReluDepthwiseConvPass);

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

#include "paddle/fluid/framework/ir/fuse_reshape_transpose_scale_matmul_pass.h"
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

PDNode* patterns::ReshapeTransposeScaleMatmul::operator()(
    paddle::framework::ir::PDNode* matmul_output, bool scale) {
  // Create Operators
  matmul_output->assert_is_op_output("matmul", "Out");
  auto* reshape_op = pattern->NewNode(reshape_repr())->assert_is_op("reshape2");
  auto* transpose_op =
      pattern->NewNode(transpose_repr())->assert_is_op("transpose2");
  auto* scale_op = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* eltwise_add_op =
      pattern->NewNode(eltwise_add_repr())->assert_is_op("elementwise_add");

  // Create variables
  auto* reshape_input_var = pattern->NewNode(reshape_input_repr())
                                ->AsInput()
                                ->assert_is_op_input("reshape2", "X");
  auto* transpose_input_var = pattern->NewNode(transpose_input_repr())
                                  ->AsInput()
                                  ->AsIntermediate()
                                  ->assert_is_op_output("reshape2")
                                  ->assert_is_op_input("transpose2");
  auto* scale_input_var = pattern->NewNode(scale_input_repr())
                              ->AsInput()
                              ->AsIntermediate()
                              ->assert_is_op_output("transpose2")
                              ->assert_is_op_input("scale");

  auto* matmul_op = pattern->NewNode(matmul_repr())->assert_is_op("matmul");

  auto* matmul_input_var = pattern->NewNode(matmul_input_repr())
                               ->AsInput()
                               ->AsIntermediate()
                               ->assert_is_op_input("matmul");

  // Link operators
  reshape_op->LinksFrom({reshape_input_var}).LinksTo({transpose_input_var});
  if (scale) {
    transpose_op->LinksFrom({transpose_input_var}).LinksTo({scale_input_var});
    scale_op->LinksFrom({scale_input_var}).LinksTo({matmul_input_var});
  } else {
    transpose_op->LinksFrom({transpose_input_var}).LinksTo({matmul_input_var});
  }

  matmul_op->LinksFrom({matmul_input_var}).LinksTo({matmul_output});
  eltwise_add_op->LinksFrom({matmul_output});
  return matmul_output;
}

inline Node* GetInputNode(const Node* node, std::string type = "X") {
  for (auto it = node->inputs.begin(); it != node->inputs.end(); it++) {
    if (0 == node->Op()->Input(type)[0].compare((*it)->Name())) {
      return *it;
    }
  }

  return nullptr;
}

std::unique_ptr<ir::Graph> ReshapeTransposeScaleMatmulFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  auto* matmul_output =
      pattern->NewNode(patterns::PDNodeName(name_scope_, "matmul_out"))
          ->AsInput()
          ->assert_is_op_output("matmul", "Out");

  bool detect_scale = false;

  std::shared_ptr<patterns::ReshapeTransposeScaleMatmul> matmul_pattern(
      new patterns::ReshapeTransposeScaleMatmul(pattern, name_scope_));
  (*(matmul_pattern.get()))(matmul_output, detect_scale);

  int found_fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ReshapeTransposeScaleMatmul fuse";
    GET_IR_NODE_FROM_SUBGRAPH(reshape_op, reshape, (*(matmul_pattern.get())));
    GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose,
                              (*(matmul_pattern.get())));
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul, (*(matmul_pattern.get())));
    GET_IR_NODE_FROM_SUBGRAPH(reshape_input, reshape_input,
                              (*(matmul_pattern.get())));
    GET_IR_NODE_FROM_SUBGRAPH(transpose_input, transpose_input,
                              (*(matmul_pattern.get())));
    GET_IR_NODE_FROM_SUBGRAPH(matmul_input, matmul_input,
                              (*(matmul_pattern.get())));
    Node *scale_op = nullptr, *scale_input = nullptr;
    if (detect_scale) {
      GET_IR_NODE_FROM_SUBGRAPH(scale, scale, (*(matmul_pattern.get())));
      GET_IR_NODE_FROM_SUBGRAPH(scale_input_var, scale_input,
                                (*(matmul_pattern.get())));
      scale_op = scale;
      scale_input = scale_input_var;
    }
    PADDLE_ENFORCE(subgraph.count(matmul_output));

    auto reshape_shape_tz =
        boost::get<std::vector<int>>(reshape_op->Op()->GetAttr("shape"));
    auto transpose_axis_tz =
        boost::get<std::vector<int>>(transpose_op->Op()->GetAttr("axis"));

    bool is_x = matmul_input == GetInputNode(matmul_op, "X");

    if (is_x) {
      matmul_op->Op()->SetAttr("shape_X", reshape_shape_tz);
      matmul_op->Op()->SetAttr("axis_X", transpose_axis_tz);
    } else {
      matmul_op->Op()->SetAttr("shape_Y", reshape_shape_tz);
      matmul_op->Op()->SetAttr("axis_Y", transpose_axis_tz);
    }

    reshape_input->outputs.clear();
    reshape_op->inputs.clear();
    std::unordered_set<const Node*> remove_nodes;
    float bias = 0.0f, alpha = 1.0f;

    if (scale_op != nullptr) {
      bias = boost::get<float>(scale_op->Op()->GetAttr("bias"));
      alpha = boost::get<float>(scale_op->Op()->GetAttr("scale"));
    }

    if (scale_op != nullptr && scale_input != nullptr && bias != 0.0f) {
      scale_input->outputs.clear();
      scale_op->inputs.clear();
      scale_op->Op()->SetInput(
          "X", std::vector<std::string>({reshape_input->Name()}));
      IR_NODE_LINK_TO(reshape_input, scale_op);

      remove_nodes.insert(
          {reshape_op, transpose_op, transpose_input, scale_input});
    } else {
      if (scale_op != nullptr && scale_input != nullptr) {
        matmul_op->Op()->SetAttr("alpha", alpha);
        remove_nodes.insert({scale_op, scale_input});
      }
      matmul_input->outputs.clear();

      for (auto it = matmul_op->inputs.begin();
           it != matmul_op->inputs.end();) {
        if (*it == matmul_input) {
          it = matmul_op->inputs.erase(it);
        } else {
          it++;
        }
      }
      if (is_x) {
        matmul_op->Op()->SetInput(
            "X", std::vector<std::string>({reshape_input->Name()}));
      } else {
        matmul_op->Op()->SetInput(
            "Y", std::vector<std::string>({reshape_input->Name()}));
      }

      IR_NODE_LINK_TO(reshape_input, matmul_op);
      remove_nodes.insert(
          {reshape_op, transpose_op, transpose_input, matmul_input});
    }

    GraphSafeRemoveNodes(graph.get(), remove_nodes);

    found_fuse_count++;
  };

  gpd(graph.get(), handler);

  detect_scale = true;
  GraphPatternDetector gpd_scale;
  pattern = gpd_scale.mutable_pattern();
  matmul_output =
      pattern->NewNode(patterns::PDNodeName(name_scope_, "matmul_out"))
          ->AsInput()
          ->assert_is_op_output("matmul", "Out");
  matmul_pattern.reset(
      new patterns::ReshapeTransposeScaleMatmul(pattern, name_scope_));
  (*(matmul_pattern.get()))(matmul_output, detect_scale);
  gpd_scale(graph.get(), handler);

  AddStatis(found_fuse_count);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_reshape_transpose_scale_matmul_pass,
              paddle::framework::ir::ReshapeTransposeScaleMatmulFusePass);

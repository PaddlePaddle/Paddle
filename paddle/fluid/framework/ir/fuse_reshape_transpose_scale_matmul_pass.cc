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

#define MATMUL_COMPUTE_DIMENSION 2

PDNode* patterns::ReshapeTransposeScaleMatmul::operator()(
    paddle::framework::ir::PDNode* matmul_output) {
  // Create Operators
  matmul_output->assert_is_op_output("matmul", "Out");
  auto* reshape_op = pattern->NewNode(reshape_repr())->assert_is_op("reshape2");
  auto* transpose_op =
      pattern->NewNode(transpose_repr())->assert_is_op("transpose2");

  // Create variables
  auto* reshape_input_var = pattern->NewNode(reshape_input_repr())
                                ->AsInput()
                                ->assert_is_op_input("reshape2", "X");
  auto* transpose_input_var = pattern->NewNode(transpose_input_repr())
                                  ->AsInput()
                                  ->AsIntermediate()
                                  ->assert_is_op_output("reshape2")
                                  ->assert_is_op_input("transpose2");

  auto* matmul_op = pattern->NewNode(matmul_repr())->assert_is_op("matmul");

  auto* matmul_input_var = pattern->NewNode(matmul_input_repr())
                               ->AsInput()
                               ->AsIntermediate()
                               ->assert_is_op_output("transpose2")
                               ->assert_is_op_input("matmul");

  reshape_op->LinksFrom({reshape_input_var}).LinksTo({transpose_input_var});
  transpose_op->LinksFrom({transpose_input_var}).LinksTo({matmul_input_var});

  matmul_op->LinksFrom({matmul_input_var}).LinksTo({matmul_output});
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

  std::cout << "fuse_reshape_transpose_scale_matmul_pass+++" << std::endl;
  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  GraphPatternDetector gpd;
  auto* matmul_output =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "matmul_out"))
          ->AsInput()
          ->assert_is_op_output("matmul", "Out");
  patterns::ReshapeTransposeScaleMatmul pattern(gpd.mutable_pattern(),
                                                name_scope_);
  pattern(matmul_output);
  int found_fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ReshapeTransposeScaleMatmul fuse";
    GET_IR_NODE_FROM_SUBGRAPH(reshape, reshape, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose, transpose, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul, matmul, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_input, reshape_input, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_input, transpose_input, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_input, matmul_input, pattern);

    PADDLE_ENFORCE(subgraph.count(matmul_output));

    auto reshape_shape_tz =
        boost::get<std::vector<int>>(reshape->Op()->GetAttr("shape"));
    auto transpose_axis_tz =
        boost::get<std::vector<int>>(transpose->Op()->GetAttr("axis"));

    int ndmis = reshape_shape_tz.size();

    int stride = 1;
    int ld = 1;

    for (auto i = transpose_axis_tz[ndmis - MATMUL_COMPUTE_DIMENSION - 1] + 1;
         i < ndmis; i++) {
      stride *= reshape_shape_tz[i];
    }

    for (auto i = transpose_axis_tz[ndmis - MATMUL_COMPUTE_DIMENSION] + 1;
         i < ndmis; i++) {
      ld *= reshape_shape_tz[i];
    }

    bool is_x = matmul_input == GetInputNode(matmul, "X");

    for (auto& attr : matmul->Op()->GetAttrMap()) {
      if (attr.first == "LDX" && is_x) {
        matmul->Op()->SetAttr(attr.first, ld);
      }
      if (attr.first == "LDY" && !is_x) {
        matmul->Op()->SetAttr(attr.first, ld);
      }
      if (attr.first == "STRIDEX" && is_x) {
        matmul->Op()->SetAttr(attr.first, stride);
      }
      if (attr.first == "STRIDEY" && !is_x) {
        matmul->Op()->SetAttr(attr.first, stride);
      } else {
        matmul->Op()->SetAttr(attr.first, attr.second);
      }
    }

    auto transpose_shape_tz = reshape_shape_tz;
    for (auto idx = 0; idx < transpose_axis_tz.size(); idx++) {
      if (transpose_axis_tz[idx] > 0 &&
          transpose_axis_tz[idx] < reshape_shape_tz.size()) {
        transpose_shape_tz[idx] = reshape_shape_tz[transpose_axis_tz[idx]];
      }
    }

    auto* tensor = scope->Var(reshape_input->Name())->GetMutable<LoDTensor>();
    tensor->Resize(paddle::framework::make_ddim(transpose_axis_tz));

    reshape_input->outputs.clear();
    reshape->inputs.clear();
    matmul_input->outputs.clear();

    for (auto it = matmul->inputs.begin(); it != matmul->inputs.end();) {
      if (*it == matmul_input) {
        it = matmul->inputs.erase(it);
      } else {
        it++;
      }
    }

    if (is_x) {
      matmul->Op()->SetInput("X",
                             std::vector<std::string>({reshape_input->Name()}));
    } else {
      matmul->Op()->SetInput("Y",
                             std::vector<std::string>({reshape_input->Name()}));
    }

    IR_NODE_LINK_TO(reshape_input, matmul);

    GraphSafeRemoveNodes(graph.get(),
                         {reshape, transpose, transpose_input, matmul_input});

    std::cout << "++++" << found_fuse_count << "+++" << std::endl;

    found_fuse_count++;
  };
  gpd(graph.get(), handler);
  AddStatis(found_fuse_count);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_reshape_transpose_scale_matmul_pass,
              paddle::framework::ir::ReshapeTransposeScaleMatmulFusePass);

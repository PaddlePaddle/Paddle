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

#include "paddle/fluid/framework/ir/trt_map_ops_to_matrix_multiply_pass.h"

#include <cmath>
#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class Node;

TrtMapOpsToMatrixMultiplyPass::TrtMapOpsToMatrixMultiplyPass() {}

void TrtMapOpsToMatrixMultiplyPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "trt_map_ops_to_matrix_multiply_pass";
  FusePassBase::Init(name_scope, graph);

  std::unordered_set<std::string> ops_type = {"mul", "matmul", "matmul_v2"};
  GraphPatternDetector gpd;
  patterns::MulMatmulMatmulV2 mul_matmul_matmul_v2(gpd.mutable_pattern(),
                                                   name_scope);
  mul_matmul_matmul_v2(ops_type);
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    bool with_dynamic_shape = Get<bool>("with_dynamic_shape");
    if (!with_dynamic_shape) {
      VLOG(3)
          << "TrtMapOpsToMatrixMultiplyPass need with_dynamic_shape, stop this "
             "pass."
             "Please reconfig 'SetTRTDynamicShapeInfo'. You can refer to the "
             "https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/"
             "master/c%2B%2B/gpu/resnet50/resnet50_test.cc";
      return;
    }
    VLOG(4) << "trt map some ops to matrix_multiply";
    GET_IR_NODE_FROM_SUBGRAPH(ops, ops, mul_matmul_matmul_v2);
    GET_IR_NODE_FROM_SUBGRAPH(ops_out, ops_out, mul_matmul_matmul_v2);
    OpDesc desc(ops->Op()->Block());
    desc.SetType("matrix_multiply");
    desc.SetInput("X", {ops->Op()->Input("X").front()});
    desc.SetInput("Y", {ops->Op()->Input("Y").front()});
    desc.SetOutput("Out", {ops_out->Name()});

    if (ops->Op()->HasAttr("transpose_X") || ops->Op()->HasAttr("trans_x")) {
      if (ops->Op()->HasAttr("transpose_X")) {
        desc.SetAttr("transpose_x", ops->Op()->GetAttr("transpose_X"));
      } else {
        desc.SetAttr("transpose_x", ops->Op()->GetAttr("trans_x"));
      }
    } else {
      desc.SetAttr("transpose_x", false);
    }

    if (ops->Op()->HasAttr("transpose_Y") || ops->Op()->HasAttr("trans_y")) {
      if (ops->Op()->HasAttr("transpose_Y")) {
        desc.SetAttr("transpose_y", ops->Op()->GetAttr("transpose_Y"));
      } else {
        desc.SetAttr("transpose_y", ops->Op()->GetAttr("trans_y"));
      }
    } else {
      desc.SetAttr("transpose_y", false);
    }

    if (ops->Op()->HasAttr("out_threshold")) {
      desc.SetAttr("out_threshold", ops->Op()->GetAttr("out_threshold"));
    }

    // Todo: remove attr(x_num_col_dims, y_num_col_dims, alpha)
    if (ops->Op()->HasAttr("x_num_col_dims")) {
      desc.SetAttr("x_num_col_dims", ops->Op()->GetAttr("x_num_col_dims"));
    } else {
      int32_t x_num_col_dims = -1;
      desc.SetAttr("x_num_col_dims", x_num_col_dims);
    }

    // op_teller: Only support y_num_col_dims == y.rank - 1;
    int32_t y_num_col_dims = -1;
    desc.SetAttr("y_num_col_dims", y_num_col_dims);

    float alpha = 1;
    if (ops->Op()->HasAttr("alpha")) {
      alpha = PADDLE_GET_CONST(float, ops->Op()->GetAttr("alpha"));
    }
    desc.SetAttr("alpha", alpha);

    auto matrix_multiply_node = g->CreateOpNode(&desc);
    for (auto node : ops->inputs) {
      IR_NODE_LINK_TO(node, matrix_multiply_node);
    }
    IR_NODE_LINK_TO(matrix_multiply_node, ops_out);
    GraphSafeRemoveNodes(graph, {ops});
    ++found_count;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(trt_map_ops_to_matrix_multiply_pass,
              paddle::framework::ir::TrtMapOpsToMatrixMultiplyPass);

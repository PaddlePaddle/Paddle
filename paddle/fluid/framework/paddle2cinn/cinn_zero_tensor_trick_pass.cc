/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/paddle2cinn/cinn_zero_tensor_trick_pass.h"

#include <string>
#include "glog/logging.h"

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

void CinnZeroTensorTrickPass::ApplyImpl(ir::Graph* graph) const {
  // fix shape attr of these ops
  const std::unordered_set<std::string> op_cases_fix_attr{"fill_constant",
                                                          "uniform_random",
                                                          "expand_v2",
                                                          "assign_value",
                                                          "gaussian_random",
                                                          "set_value"};
  // NOTE: Hack squeeze2 0D-Tensor input
  // If squeeze2 inputs 0D-Tensor and axes, The 0D-Tensor's shape will convert
  // to 1D-Tensor, which could lead error. We hack squeeze2's axes attribute to
  // resolve this. Change 0D-Tensor input to 1D-Tensor input and then make
  // axes->axes[: -1]
  for (const ir::Node* n : graph->Nodes()) {
    if (n->IsOp() && n->Op()->Type() == "unsqueeze2") {
      if (n->Op()->HasAttr("axes")) {
        auto axes =
            PADDLE_GET_CONST(std::vector<int32_t>, n->Op()->GetAttr("axes"));
        for (const ir::Node* var : n->inputs) {
          if (var->Var() &&
              var->Var()->GetType() == proto::VarType::LOD_TENSOR) {
            std::vector<int64_t> shape = var->Var()->GetShape();
            if (shape.empty()) {
              axes.pop_back();
              n->Op()->SetAttr("axes", axes);
              VLOG(4) << "unsqueeze2 axes dims is full, fix dim -> dim[:-1] to "
                         "avoid 0D-Tensor input error";
            }
          }
        }
      }
    }
  }

  // CINN ops in this white list support 0D-Tensor
  const std::unordered_set<std::string> white_op_list{"elementwise_add",
                                                      "elementwise_sub",
                                                      "elementwise_mul",
                                                      "elementwise_div"};
  std::unordered_set<std::string> white_tensor_name;
  // enable white_op_list only when graph_node_size = 1, which means single op
  // test
  int graph_node_size = 0;
  for (const ir::Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      graph_node_size++;
      VLOG(6) << "Graph has op node " << n->Op()->Type();
      if (white_op_list.find(n->Op()->Type()) != white_op_list.end()) {
        for (const ir::Node* var : n->inputs) {
          white_tensor_name.insert(var->Var()->Name());

          std::vector<int64_t> shape = var->Var()->GetShape();
          if (shape.empty()) {
            VLOG(6) << "input var " << var->Name()
                    << " dims is empty, keep it's 0D-Tensor status";
          }
        }
        for (const ir::Node* var : n->outputs) {
          white_tensor_name.insert(var->Var()->Name());

          std::vector<int64_t> shape = var->Var()->GetShape();
          if (shape.empty()) {
            VLOG(6) << "output var " << var->Name()
                    << " dims is empty, keep it's 0D-Tensor status";
          }
        }
      }
    }
  }
  VLOG(6) << "Graph has " << graph_node_size << " op node";

  for (const ir::Node* n : graph->Nodes()) {
    if (n->IsOp() && op_cases_fix_attr.count(n->Op()->Type())) {
      if (n->Op()->HasAttr("shape")) {
        auto attr_type = n->Op()->GetAttrType("shape");
        if (attr_type == paddle::framework::proto::INTS) {
          auto shapes =
              PADDLE_GET_CONST(std::vector<int32_t>, n->Op()->GetAttr("shape"));
          if (shapes.empty()) {
            shapes.push_back(1);
            n->Op()->SetAttr("shape", shapes);
            VLOG(4) << "op " << n->Op()->Type()
                    << " shape attribute dims is empty, fix dim -> {1} ";
          }
        } else { /* attr_type == paddle::framework::proto::LONGS */
          auto shapes =
              PADDLE_GET_CONST(std::vector<int64_t>, n->Op()->GetAttr("shape"));
          if (shapes.empty()) {
            shapes.push_back(1);
            n->Op()->SetAttr("shape", shapes);
            VLOG(4) << "op " << n->Op()->Type()
                    << " shape attribute dims is empty, fix dim -> {1} ";
          }
        }
      }
    }
    if (n->IsVar()) {
      if (n->Var() && n->Var()->GetType() == proto::VarType::LOD_TENSOR) {
        if (graph_node_size == 1 && white_tensor_name.find(n->Var()->Name()) !=
                                        white_tensor_name.end()) {
          VLOG(6) << "Keep 0D-Tensor status of var " << n->Var()->Name();
          continue;
        }
        std::vector<int64_t> shape = n->Var()->GetShape();
        if (shape.empty()) {
          shape.push_back(1);
          n->Var()->SetShape(shape);
          VLOG(4) << "var " << n->Name() << " dims is empty, fix dim -> {1} ";
        }
      }
    }
  }
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cinn_zero_tensor_trick_pass,
              paddle::framework::paddle2cinn::CinnZeroTensorTrickPass);

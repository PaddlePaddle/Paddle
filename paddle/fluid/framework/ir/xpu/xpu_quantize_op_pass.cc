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

#include "paddle/fluid/framework/ir/xpu/xpu_quantize_op_pass.h"

#include <sstream>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

static void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

static void MarkAndLogCannotQuantizeOp(Node* op,
                                       const char* details = nullptr) {
  std::stringstream msg_ss;
  msg_ss << "Cannot quantize operator " << op->Name()
         << " (type: " << op->Op()->Type() << ", id: " << op->id() << ").";
  if (details) msg_ss << " " << details;
  VLOG(2) << msg_ss.str().c_str();
}
void XPUQuantizeOpPass::GetQuantInfo(Graph* graph) const {
  var_quant_scales_ =
      GetQuantInfoFromTheGraph(graph, "has_quant_info", "var_quant_scales");
}

void XPUQuantizeOpPass::QuantizeInput(Graph* g,
                                      Node* op,
                                      Node* input,
                                      std::string input_arg_name) const {
  auto inputs = op->Op()->InputNames();
  bool name_found =
      std::find(inputs.begin(), inputs.end(), input_arg_name) != inputs.end();
  PADDLE_ENFORCE_EQ(name_found,
                    true,
                    common::errors::InvalidArgument(
                        "Var(%s) isn't the input of the %s operator.",
                        input_arg_name,
                        op->Op()->Type()));

  // Create quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);
  quantize_out_node->Var()->SetDataType(
      proto::VarType::Type::VarType_Type_INT8);

  // Create a quantize op node
  float scale = GetScaleValueForNode(&var_quant_scales_, input);
  OpDesc q_desc;
  q_desc.SetType("quantize_xpu");
  q_desc.SetInput("x", std::vector<std::string>({input->Name()}));
  q_desc.SetOutput("y", std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("out_dtype",
                 static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
  q_desc.SetAttr("scale", static_cast<float>(scale));
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  // Update op's input
  op->Op()->SetInput(input_arg_name,
                     std::vector<std::string>({quantize_out_node->Name()}));

  // Link quantize op
  UnlinkNodes(input, op);
  IR_NODE_LINK_TO(input, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);
}

void XPUQuantizeOpPass::DequantizeOutput(Graph* g,
                                         Node* op,
                                         Node* output,
                                         std::string output_arg_name) const {
  auto outputs = op->Op()->OutputNames();
  bool name_found =
      std::find(outputs.begin(), outputs.end(), output_arg_name) !=
      outputs.end();
  PADDLE_ENFORCE_EQ(name_found,
                    true,
                    common::errors::InvalidArgument(
                        "Var(%s) isn't the output of the %s operator.",
                        output_arg_name,
                        op->Op()->Type()));

  // Create dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);
  dequantize_in_node->Var()->SetDataType(
      proto::VarType::Type::VarType_Type_INT8);

  float scale = GetScaleValueForNode(&var_quant_scales_, output);
  // Create a quantize op node
  OpDesc deq_desc;
  deq_desc.SetType("dequantize_xpu");
  deq_desc.SetInput("x",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetOutput("y", std::vector<std::string>({output->Name()}));
  deq_desc.SetAttr("out_dtype", static_cast<int>(output->Var()->GetDataType()));
  deq_desc.SetAttr("scale", static_cast<float>(scale));
  auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.

  // Update op's input
  op->Op()->SetOutput(output_arg_name,
                      std::vector<std::string>({dequantize_in_node->Name()}));

  // Link dequantize op
  UnlinkNodes(op, output);
  IR_NODE_LINK_TO(op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(dequantize_op, output);
}

void XPUQuantizeOpPass::QuantizeConv(ir::Graph* graph) const {
  for (auto* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->Type() != "conv2d_xpu") {
        continue;
      }
      Node* w_var_node = nullptr;
      Node* x_var_node = nullptr;
      Node* out_var_node = nullptr;
      Node* branch_var_node = nullptr;

      for (auto* input_node : n->inputs) {
        if (!input_node->IsVar()) {
          continue;
        }
        if (input_node->Var()->Name() == op->Input("x")[0]) {
          x_var_node = input_node;
        } else if (input_node->Var()->Name() == op->Input("filter")[0]) {
          w_var_node = input_node;
        } else if (op->HasInput("branch") &&
                   input_node->Var()->Name() == op->Input("branch")[0]) {
          branch_var_node = input_node;
        }
      }

      for (auto* output_node : n->outputs) {
        if (!output_node->IsVar()) {
          continue;
        }
        if (output_node->Var()->Name() == op->Output("out")[0]) {
          out_var_node = output_node;
        }
      }
      if (!AreScalesPresentForNodes(&var_quant_scales_, {x_var_node}) ||
          w_var_node->Var()->GetDataType() !=
              proto::VarType::Type::VarType_Type_INT8) {
        VLOG(4) << "Skip quantize op: " << n->Name()
                << "x_var_node_name:" << x_var_node->Name()
                << " w_var_node_name:" << w_var_node->Name();
        MarkAndLogCannotQuantizeOp(n, "No scale available for the operator");
        continue;
      }

      QuantizeInput(graph, n, x_var_node, "x");
      auto has_output_scale =
          AreScalesPresentForNodes(&var_quant_scales_, {out_var_node});
      bool has_branch = branch_var_node != nullptr;

      // Note: Conv2d fusion requires branch datatype is same as output
      // datatype, so we should consider branch/output together.
      if (has_branch) {
        bool has_branch_scale =
            AreScalesPresentForNodes(&var_quant_scales_, {branch_var_node});
        if (has_output_scale && has_branch_scale) {
          QuantizeInput(graph, n, branch_var_node, "branch");
          DequantizeOutput(graph, n, out_var_node, "out");
          // Note: out_dtype attr must be set, because if dequantize_output, we
          // consider the kernel out_dtype as int8.
          n->Op()->SetAttr(
              "out_dtype",
              static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
        } else {
          n->Op()->SetAttr("out_dtype", x_var_node->Var()->GetDataType());
        }
      } else {
        if (has_output_scale) {
          DequantizeOutput(graph, n, out_var_node, "out");
          // Note: out_dtype attr must be set, because if dequantize_output, we
          // consider the kernel out_dtype as int8.
          n->Op()->SetAttr(
              "out_dtype",
              static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
        } else {
          n->Op()->SetAttr("out_dtype", x_var_node->Var()->GetDataType());
        }
      }
    }
  }
}

void XPUQuantizeOpPass::QuantizeFC(ir::Graph* graph) const {
  for (auto* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->Type() != "fc_xpu") {
        continue;
      }
      Node* w_var_node = nullptr;
      Node* x_var_node = nullptr;
      Node* out_var_node = nullptr;

      for (auto* input_node : n->inputs) {
        if (!input_node->IsVar()) {
          continue;
        }
        if (input_node->Var()->Name() == op->Input("x")[0]) {
          x_var_node = input_node;
        } else if (input_node->Var()->Name() == op->Input("w")[0]) {
          w_var_node = input_node;
        }
      }

      for (auto* output_node : n->outputs) {
        if (!output_node->IsVar()) {
          continue;
        }
        if (output_node->Var()->Name() == op->Output("out")[0]) {
          out_var_node = output_node;
        }
      }
      if (!AreScalesPresentForNodes(&var_quant_scales_, {x_var_node}) ||
          w_var_node->Var()->GetDataType() !=
              proto::VarType::Type::VarType_Type_INT8) {
        MarkAndLogCannotQuantizeOp(n, "No scale available for the operator");
        continue;
      }

      QuantizeInput(graph, n, x_var_node, "x");

      auto has_output_scale =
          AreScalesPresentForNodes(&var_quant_scales_, {out_var_node});
      if (has_output_scale) {
        DequantizeOutput(graph, n, out_var_node, "out");
        n->Op()->SetAttr(
            "out_dtype",
            static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
      } else {
        n->Op()->SetAttr("out_dtype", x_var_node->Var()->GetDataType());
      }
    }
  }
}

void XPUQuantizeOpPass::QuantizeQkvAttention(ir::Graph* graph) const {
  for (auto* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->Type() != "qkv_attention_xpu") {
        continue;
      }
      std::vector<std::string> max_node_names = {
          "q_max", "k_max", "v_max", "qk_max"};
      std::unordered_map<std::string, Node*> input_node_map;
      for (auto* input_node : n->inputs) {
        if (!input_node->IsVar()) {
          continue;
        }
        for (auto input_name : op->InputNames()) {
          if (op->Input(input_name)[0] == input_node->Var()->Name()) {
            input_node_map[input_name] = input_node;
          }
        }
      }
      bool continue_flag = false;
      for (auto max_name : max_node_names) {
        if (input_node_map.find(max_name) == input_node_map.end()) {
          continue_flag = true;
          break;
        }
      }
      if (continue_flag) {
        continue;
      }
      Node* out_var_node = nullptr;
      for (auto* output_node : n->outputs) {
        if (!output_node->IsVar()) {
          continue;
        }
        if (output_node->Var()->Name() == op->Output("qkv")[0]) {
          out_var_node = output_node;
        }
      }
      if (input_node_map["q"]->Name() == input_node_map["k"]->Name() &&
          input_node_map["q"]->Name() == input_node_map["v"]->Name()) {
        QuantizeInput(graph, n, input_node_map["q"], "q");
        op->SetInput("k", op->Input("q"));
        op->SetInput("v", op->Input("q"));
        UnlinkNodes(input_node_map["k"], n);
        UnlinkNodes(input_node_map["v"], n);
      } else {
        QuantizeInput(graph, n, input_node_map["q"], "q");
        QuantizeInput(graph, n, input_node_map["k"], "k");
        QuantizeInput(graph, n, input_node_map["v"], "v");
      }
      auto has_output_scale =
          AreScalesPresentForNodes(&var_quant_scales_, {out_var_node});
      if (has_output_scale) {
        DequantizeOutput(graph, n, out_var_node, "qkv");
        n->Op()->SetAttr(
            "out_dtype",
            static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
      } else {
        n->Op()->SetAttr("out_dtype",
                         input_node_map["q"]->Var()->GetDataType());
      }
    }
  }
}
void XPUQuantizeOpPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Insert quantize/dequantize op to the graph.";
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);
  PADDLE_ENFORCE_NOT_NULL(
      param_scope(),
      common::errors::InvalidArgument("Scope cannot be nullptr."));

  GetQuantInfo(graph);
  QuantizeConv(graph);
  QuantizeFC(graph);
  QuantizeQkvAttention(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(xpu_quantize_op_pass, paddle::framework::ir::XPUQuantizeOpPass);

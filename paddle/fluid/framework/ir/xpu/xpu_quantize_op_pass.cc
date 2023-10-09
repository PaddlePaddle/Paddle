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

#include "paddle/fluid/framework/ir/quantize_related_pass_utils.h"
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

void XPUQuantizeOpPass::GetQuantInfo(Graph* graph) const {
  GetInfoFromTheTmpOp(
      graph,
      "has_quant_info",
      "var_quant_scales",
      const_cast<std::unordered_map<std::string, std::vector<float>>*>(
          &var_quant_scales_));
}

bool XPUQuantizeOpPass::AreScalesPresentForNodes(
    std::initializer_list<Node*> nodes) const {
  bool present = true;
  for (auto node : nodes) {
    if (var_quant_scales_.count(node->Name()) == 0) {
      present = false;
    }
  }
  return present;
}

float XPUQuantizeOpPass::GetScaleValueForNode(Node* node) const {
  return var_quant_scales_.at(node->Name())[0];
}

void XPUQuantizeOpPass::QuantizeInput(Graph* g,
                                      Node* op,
                                      Node* input,
                                      std::string input_arg_name) const {
  auto* scope = param_scope();
  auto inputs = op->Op()->InputNames();
  bool name_found =
      std::find(inputs.begin(), inputs.end(), input_arg_name) != inputs.end();
  PADDLE_ENFORCE_EQ(name_found,
                    true,
                    platform::errors::InvalidArgument(
                        "Var(%s) isn't the input of the %s operator.",
                        input_arg_name,
                        op->Op()->Type()));

  // Create quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);
  quantize_out_node->Var()->SetDataType(
      proto::VarType::Type::VarType_Type_INT8);
  // Create quantize max_ptr node

  float scale = GetScaleValueForNode(input);
  int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
  std::string input_max_name = input->Name() + "_quantize_max";
  VarDesc input_max_desc(input_max_name);
  input_max_desc.SetPersistable(
      true);  // Need depends on ir_params_sync_among_devices_pass copy to xpu
              // device
  input_max_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
  input_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
  Node* input_max_node = g->CreateVarNode(&input_max_desc);
  auto input_max_tensor =
      scope->Var(input_max_name)->GetMutable<phi::DenseTensor>();
  input_max_tensor->set_type(phi::DataType::FLOAT32);
  input_max_tensor->Resize({max_ptr_size});
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  std::vector<float> input_scales(max_ptr_size, scale);
  memcpy(cpu_ctx->Alloc<float>(input_max_tensor),
         input_scales.data(),
         max_ptr_size * sizeof(float));

  // create a quantize op node
  OpDesc q_desc;
  q_desc.SetType("quantize_xpu");
  q_desc.SetInput("x", std::vector<std::string>({input->Name()}));
  q_desc.SetInput("max", std::vector<std::string>({input_max_name}));
  q_desc.SetOutput("y", std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("out_dtype",
                 static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
  q_desc.SetAttr("scale", static_cast<float>(scale));

  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.
  // update op's input
  op->Op()->SetInput(input_arg_name,
                     std::vector<std::string>({quantize_out_node->Name()}));
  // link quantize op
  UnlinkNodes(input, op);
  IR_NODE_LINK_TO(input, quantize_op);
  IR_NODE_LINK_TO(input_max_node, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);
}

void XPUQuantizeOpPass::DequantizeOutput(Graph* g,
                                         Node* op,
                                         Node* output,
                                         std::string output_arg_name) const {
  auto* scope = param_scope();
  auto outputs = op->Op()->OutputNames();
  bool name_found =
      std::find(outputs.begin(), outputs.end(), output_arg_name) !=
      outputs.end();
  PADDLE_ENFORCE_EQ(name_found,
                    true,
                    platform::errors::InvalidArgument(
                        "Var(%s) isn't the output of the %s operator.",
                        output_arg_name,
                        op->Op()->Type()));

  // Create dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);
  dequantize_in_node->Var()->SetDataType(
      proto::VarType::Type::VarType_Type_INT8);

  // Create dequantize max_ptr node
  float scale = GetScaleValueForNode(output);
  int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
  std::string input_max_name = output->Name() + "_dequantize_max";
  VarDesc input_max_desc(input_max_name);
  input_max_desc.SetPersistable(
      true);  // Need depends on ir_params_sync_among_devices_pass copy to xpu
              // device
  input_max_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
  input_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
  Node* input_max_node = g->CreateVarNode(&input_max_desc);
  auto input_max_tensor =
      scope->Var(input_max_name)->GetMutable<phi::DenseTensor>();
  input_max_tensor->set_type(phi::DataType::FLOAT32);
  input_max_tensor->Resize({max_ptr_size});
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  std::vector<float> input_scales(max_ptr_size, scale);
  memcpy(cpu_ctx->Alloc<float>(input_max_tensor),
         input_scales.data(),
         max_ptr_size * sizeof(float));

  // create a quantize op node
  OpDesc deq_desc;
  deq_desc.SetType("dequantize_xpu");
  deq_desc.SetInput("x",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetInput("max", std::vector<std::string>({input_max_name}));
  deq_desc.SetOutput("y", std::vector<std::string>({output->Name()}));
  deq_desc.SetAttr("out_dtype", static_cast<int>(output->Var()->GetDataType()));
  deq_desc.SetAttr("scale", static_cast<float>(scale));

  auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.
  // update op's input
  op->Op()->SetOutput(output_arg_name,
                      std::vector<std::string>({dequantize_in_node->Name()}));
  // link dequantize op
  UnlinkNodes(op, output);
  IR_NODE_LINK_TO(op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(input_max_node, dequantize_op);
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
      if (!AreScalesPresentForNodes({x_var_node})) {
        // MarkAndLogCannotQuantizeOp(conv_op,
        //                        "No scale available for the operator");
        return;
      }

      QuantizeInput(graph, n, x_var_node, "x");
      // Branch input
      if (branch_var_node != nullptr) {
        if (AreScalesPresentForNodes({branch_var_node})) {
          QuantizeInput(graph, n, branch_var_node, "branch");
        } else {
          n->Op()->SetAttr("xpu_op_force_output_precision",
                           branch_var_node->Var()->GetDataType());
        }
      }

      auto has_output_scale = AreScalesPresentForNodes({out_var_node});
      if (has_output_scale) {
        DequantizeOutput(graph, n, out_var_node, "out");
        n->Op()->SetAttr(
            "out_dtype",
            static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
      } else {
        n->Op()->SetAttr("xpu_op_force_output_precision",
                         x_var_node->Var()->GetDataType());
        n->Op()->SetAttr("out_dtype", x_var_node->Var()->GetDataType());
      }
    }
  }
}

void XPUQuantizeOpPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Insert quantize/dequantize op to the graph.";
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);
  PADDLE_ENFORCE_NOT_NULL(
      param_scope(),
      platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GetQuantInfo(graph);
  VLOG(1) << "Get quant info from graph success.";
  QuantizeConv(graph);
  VLOG(1) << "Quantize conv of the graph success.";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(xpu_quantize_op_pass, paddle::framework::ir::XPUQuantizeOpPass);

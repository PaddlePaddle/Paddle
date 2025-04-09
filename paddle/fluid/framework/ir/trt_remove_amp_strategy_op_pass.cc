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

#include "paddle/fluid/framework/ir/trt_remove_amp_strategy_op_pass.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/phi/common/data_type.h"

namespace paddle::framework::ir {

namespace {
template <typename InType, typename OutType>
void CastDataTypeInplace(phi::DenseTensor *tensor) {
  phi::DenseTensor tmp_tensor;
  tmp_tensor.set_type(phi::CppTypeToDataType<OutType>::Type());
  tmp_tensor.Resize(tensor->dims());
  auto *cpu_ctx = static_cast<phi::CPUContext *>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  auto *tmp_data = cpu_ctx->Alloc<OutType>(&tmp_tensor);
  auto *data = tensor->data<InType>();
  for (int i = 0; i < tensor->numel(); i++) {
    tmp_data[i] = static_cast<OutType>(data[i]);
  }
  tensor->clear();
  paddle::framework::TensorCopySync(tmp_tensor, phi::CPUPlace(), tensor);
}
}  // namespace

// This pass removes cast OPs that inserted by AMP strategy.
// Also, this pass sets the QAT (+ AMP) scale to be fp32.
void TrtRemoveAMPStrategyOpPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      common::errors::PreconditionNotMet(
          "During the trt_remove_strategy_op_pass, the graph "
          "should not be null."));
  FusePassBase::Init("trt_remove_strategy_op_pass", graph);
  auto *scope = param_scope();
  auto op_nodes = TopologySortOperations(*graph);

  // Find all fp16 op nodes and variables
  std::unordered_set<ir::Node *> fp16_ops;
  std::unordered_set<ir::Node *> fp16_vars;
  std::unordered_set<ir::Node *> cast_ops;
  for (auto *op_node : op_nodes) {
    PADDLE_ENFORCE_EQ(
        op_node->IsOp(),
        true,
        common::errors::InvalidArgument(
            "The 'op_node' must be an operator node, but it is not."));

    auto *op_desc = op_node->Op();
    if (op_desc->Type() == "cast") {
      auto input_dtype = op_node->inputs[0]->Var()->GetDataType();
      auto output_dtype = op_node->outputs[0]->Var()->GetDataType();
      if (input_dtype == proto::VarType::FP32 &&
          output_dtype == proto::VarType::FP16) {
        auto op_outputs = op_node->outputs;
        for (auto *out_var_node : op_outputs) {
          fp16_vars.insert(out_var_node);
        }
        cast_ops.insert(op_node);
      } else if (input_dtype == proto::VarType::FP16 &&
                 output_dtype == proto::VarType::FP32) {
        cast_ops.insert(op_node);
      }
    } else {
      auto op_inputs = op_node->inputs;
      for (auto *in_var_node : op_inputs) {
        if (fp16_vars.count(in_var_node)) {
          fp16_ops.insert(op_node);
          auto op_outputs = op_node->outputs;
          for (auto *out_var_node : op_outputs) {
            fp16_vars.insert(out_var_node);
          }
          break;
        }
      }
    }
  }

  // Set fp16 variables to be fp32
  for (auto *var : fp16_vars) {
    if (var->Var()->GetDataType() == proto::VarType::FP16) {
      var->Var()->SetDataType(proto::VarType::FP32);
    }
  }

  // Convert QDQ scale to be fp32
  for (auto *op : fp16_ops) {
    if (op->Op()->Type() == "quantize_linear" ||
        op->Op()->Type() == "dequantize_linear") {
      auto *scale_tensor = scope->FindVar(op->Op()->Input("Scale").front())
                               ->GetMutable<phi::DenseTensor>();
      if (scale_tensor->dtype() == phi::DataType::FLOAT16) {
        CastDataTypeInplace<float16, float>(scale_tensor);
      }
    }
  }

  // Remove cast OPs
  std::unordered_set<const ir::Node *> marked_nodes;
  for (auto *op_node : cast_ops) {
    auto *op_desc = op_node->Op();
    if (op_desc->Type() == "cast") {
      auto *in_var = op_node->inputs[0];
      auto *out_var = op_node->outputs[0];
      auto post_op = out_var->outputs;
      IR_NODE_UNLINK(in_var, op_node);
      IR_NODE_UNLINK(op_node, out_var);
      for (size_t i = 0; i < post_op.size(); ++i) {
        IR_NODE_UNLINK(out_var, post_op[i]);
        IR_NODE_LINK_TO(in_var, post_op[i]);
        post_op[i]->Op()->RenameInput(out_var->Var()->Name(),
                                      in_var->Var()->Name());
      }
      marked_nodes.insert(op_node);
      marked_nodes.insert(out_var);
    }
  }
  GraphSafeRemoveNodes(graph, marked_nodes);

  // Valid all cast OP is removed by this IR pass
  using DataType = proto::VarType;
  auto updated_op_nodes = TopologySortOperations(*graph);
  for (auto *op_node : updated_op_nodes) {
    if (op_node->Op()->Type() == "cast") {
      auto input_dtype = op_node->inputs[0]->Var()->GetDataType();
      auto output_dtype = op_node->outputs[0]->Var()->GetDataType();
      if ((input_dtype == DataType::FP32 && output_dtype == DataType::FP16) ||
          (input_dtype == DataType::FP16 && output_dtype == DataType::FP32)) {
        PADDLE_THROW(common::errors::Fatal(
            "There are cast OPs remaining in the graph."));
      }
    }
  }
}
}  // namespace paddle::framework::ir

REGISTER_PASS(trt_remove_amp_strategy_op_pass,
              paddle::framework::ir::TrtRemoveAMPStrategyOpPass);

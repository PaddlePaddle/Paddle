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

#include "paddle/fluid/framework/ir/trt_add_weight_quantize_linear_op_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void TrtAddWeightQuantizeLinearOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "add_weight_quantize_linear_op_pattern";

  std::cout << "add_weight_quantize_linear_op_pattern" << std::endl;

  FusePassBase::Init(pattern_name, graph);
  GraphPatternDetector detector;
  auto dequantize_linear_op_input_x =
      detector.mutable_pattern()
          ->NewNode("dequantize_linear_op_input_x")
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("dequantize_linear", "X");

  auto dequantize_linear_op_input_scale =
      detector.mutable_pattern()
          ->NewNode("dequantize_linear_op_input_scale")
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("dequantize_linear", "Scale");

  auto dequantize_linear_op_input_zeropoint =
      detector.mutable_pattern()
          ->NewNode("dequantize_linear_op_input_zeropoint")
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("dequantize_linear", "ZeroPoint");

  auto dequantize_linear_op =
      detector.mutable_pattern()
          ->NewNode("dequantize_linear_op")
          ->assert_is_op("dequantize_linear")
          ->assert_more([&](Node* node) {
            return PADDLE_GET_CONST(int, node->Op()->GetAttr("quant_axis")) ==
                   0;
          });

  auto dequantize_linear_op_out =
      detector.mutable_pattern()
          ->NewNode("dequantize_linear_op_out")
          ->assert_is_op_output("dequantize_linear")
          ->assert_more([](Node* x) { return x->outputs.size() == 1UL; });

  dequantize_linear_op
      ->LinksFrom({dequantize_linear_op_input_x,
                   dequantize_linear_op_input_scale,
                   dequantize_linear_op_input_zeropoint})
      .LinksTo({dequantize_linear_op_out});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    Node* dq_node = subgraph.at(dequantize_linear_op);
    Node* dq_input_node = subgraph.at(dequantize_linear_op_input_x);

    auto dq_desc = dq_node->Op();

    int quant_axis = PADDLE_GET_CONST(int, dq_desc->GetAttr("quant_axis"));

    if (quant_axis) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "weight quant_axis should be 0 , please check your model "
          "OP'attribute "));
    }

    auto* scope = param_scope();

    framework::OpDesc quantize_op_desc;
    const std::string& quantize_x_name =
        "add_quantize_" + dq_desc->Input("X")[0];
    const std::string& quantize_scale_name =
        "add_quantize_" + dq_desc->Input("Scale")[0];
    const std::string& quantize_zeropoint_name =
        "add_quantize_" + dq_desc->Input("ZeroPoint")[0];

    auto set_quantize_op_desc = [&](framework::OpDesc& desc,
                                    const std::string& x_name,
                                    const std::string& scale_name,
                                    const std::string& zeropoint_name,
                                    const std::string& out_name) {
      desc.SetType("quantize_linear");
      desc.SetInput("X", {x_name});
      desc.SetInput("Scale", {scale_name});
      desc.SetInput("ZeroPoint", {zeropoint_name});
      desc.SetOutput("Y", {out_name});
      desc.SetAttr("bit_length", 8);
      desc.SetAttr("quant_axis", 0);
      desc.SetAttr("op_role", 0);
      desc.SetAttr("support_int8", false);
      desc.Flush();
    };

    auto set_shape = [&](phi::DenseTensor* node_tensor) {
      std::vector<int64_t> out_shape;
      for (int64_t i = 0; i < node_tensor->dims().size(); i++) {
        out_shape.push_back(node_tensor->dims()[i]);
      }
      return out_shape;
    };

    set_quantize_op_desc(quantize_op_desc,
                         quantize_x_name,
                         quantize_scale_name,
                         quantize_zeropoint_name,
                         dq_desc->Input("X")[0]);
    auto* op_node = graph->CreateOpNode(&quantize_op_desc);

    using VarType = framework::proto::VarType;
    auto CreatePersistableVarNode =
        [&](Graph* graph,
            const std::string& name,
            framework::proto::VarType::Type datatype,
            const std::vector<int64_t>& shape) -> Node* {
      auto var_desc = VarDesc(name);
      var_desc.SetDataType(datatype);
      var_desc.SetShape(shape);
      var_desc.SetPersistable(true);
      auto node = graph->CreateVarNode(&var_desc);
      return node;
    };

    auto* dequantize_weight_tensor =
        scope->GetVar(dq_desc->Input("X")[0])->GetMutable<phi::DenseTensor>();
    auto* dequantize_scale_tensor = scope->GetVar(dq_desc->Input("Scale")[0])
                                        ->GetMutable<phi::DenseTensor>();
    auto* dequantize_zeropoint_tensor =
        scope->GetVar(dq_desc->Input("ZeroPoint")[0])
            ->GetMutable<phi::DenseTensor>();

    auto* quantize_x_tensor =
        scope->Var(quantize_x_name)->GetMutable<phi::DenseTensor>();
    auto* quantize_scale_tensor =
        scope->Var(quantize_scale_name)->GetMutable<phi::DenseTensor>();
    auto* quantize_zeropoint_tensor =
        scope->Var(quantize_zeropoint_name)->GetMutable<phi::DenseTensor>();

    auto quantize_x =
        CreatePersistableVarNode(graph,
                                 quantize_x_name,
                                 VarType::FP32,
                                 set_shape(dequantize_weight_tensor));

    auto quantize_scale =
        CreatePersistableVarNode(graph,
                                 quantize_scale_name,
                                 VarType::FP32,
                                 set_shape(dequantize_scale_tensor));

    auto quantize_zeroPoint =
        CreatePersistableVarNode(graph,
                                 quantize_zeropoint_name,
                                 VarType::INT32,
                                 set_shape(dequantize_zeropoint_tensor));

    quantize_x_tensor->Resize(dequantize_weight_tensor->dims());
    auto* quantize_x_tensor_data =
        quantize_x_tensor->mutable_data<float>(platform::CPUPlace());
    auto* dequantize_wight_data =
        dequantize_weight_tensor->mutable_data<int8_t>(platform::CPUPlace());

    for (int64_t i = 0; i < quantize_x_tensor->numel(); ++i) {
      quantize_x_tensor_data[i] = static_cast<float>(dequantize_wight_data[i]);
    }

    quantize_scale_tensor->Resize(dequantize_scale_tensor->dims());

    auto* quantize_scale_data =
        quantize_scale_tensor->mutable_data<float>(platform::CPUPlace());

    for (int64_t i = 0; i < quantize_scale_tensor->numel(); ++i) {
      quantize_scale_data[i] = 127.;
    }

    quantize_zeropoint_tensor->Resize(dequantize_zeropoint_tensor->dims());
    auto* quantize_zeropoint_tensor_data =
        quantize_zeropoint_tensor->mutable_data<float>(platform::CPUPlace());
    for (int64_t i = 0; i < dequantize_zeropoint_tensor->numel(); ++i) {
      quantize_zeropoint_tensor_data[i] = 0.;
    }

    dq_input_node->Var()->SetPersistable(false);

    IR_NODE_LINK_TO(quantize_x, op_node);
    IR_NODE_LINK_TO(quantize_scale, op_node);
    IR_NODE_LINK_TO(quantize_zeroPoint, op_node);
    IR_NODE_LINK_TO(op_node, dq_input_node);
  };
  detector(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(trt_add_weight_quantize_linear_op_pass,
              paddle::framework::ir::TrtAddWeightQuantizeLinearOpPass);

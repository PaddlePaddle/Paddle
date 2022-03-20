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

#include "paddle/fluid/framework/ir/mkldnn/quant_dequant_mkldnn_v2_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/ir/mkldnn/mkldnn_pass_util.h"

namespace paddle {
namespace framework {
namespace ir {

void QuantDequantMkldnnV2FusePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(0) << "Convert dynamic graph to INT8 mkldnn model";

  FusePassBase::Init("quant_dequant_mkldnn_v2_fuse_pass", graph);
  GraphPatternDetector gpd;

  std::unordered_set<std::string> quantize_linear_types = {"quantize_linear"};
  std::unordered_set<std::string> dequantize_linear_types = {
      "dequantize_linear"};

  std::unordered_map<std::string, std::vector<float>> weights_thresholds{};
  std::unordered_map<std::string, std::vector<float>> var_quant_scales{};

  auto* scope = param_scope();

  GatherInputWeightsScalesFromFake(graph, scope, quantize_linear_types,
                                   dequantize_linear_types, &weights_thresholds,
                                   &var_quant_scales);
  RemoveQuantDequantLinearOps(graph, scope);
  
  RemoveDequantLinearOps(graph, scope);
  VLOG(0) << "Finish dequantize_linear removing";
  // No need to dequantize, because the weights and inputs have been in float32 already.
  // DequantizeWeights(graph, scope, var_quant_scales);
  // UpdateActivations(graph);
  // RemoveCtrlVars(graph);
  
  // save var_quant_scales in the first op's attr
  // for compute_propagate_scales_mkldnn_pass
  SaveInfoInTheFirstOp(graph, "has_quant_info", "var_quant_scales",
                       var_quant_scales);
}

void QuantDequantMkldnnV2FusePass::GatherInputWeightsScalesFromFake(
    ir::Graph* graph, Scope* scope,
    std::unordered_set<std::string> quantize_linear_types,
    std::unordered_set<std::string> dequantize_linear_types,
    std::unordered_map<std::string, std::vector<float>>* weights_thresholds,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  VLOG(0) << "Gather input and weight scales from dequantize_linear";

  // auto* bn_bias_var = scope->FindVar("conv1_bn_offset");
  // PADDLE_ENFORCE_NOT_NULL(
  //         bn_bias_var, "The bn_bias_var is not found.");
  // VLOG(0)<<"bn_bias_var "<<bn_bias_var; // passed
  // auto* bn_bias_tensor = bn_bias_var->GetMutable<LoDTensor>(); //failed
  // // assert if this bn_bias_var is persistable.
  // VLOG(0) << "Passed getting bn_bias_tensor"<<(*bn_bias_tensor);
    
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    
    // find the op with Name dequantize_linear, both weights and scales
    // var_quant_scales
    if (dequantize_linear_types.find(op_node->Name()) !=
        dequantize_linear_types.end()) {
      auto* op_desc = op_node->Op();
      const int bit_length =
          BOOST_GET_CONST(int, op_desc->GetAttr("bit_length"));
      const int quant_axis =
          BOOST_GET_CONST(int, op_desc->GetAttr("quant_axis"));
      auto& y_var_name = op_desc->Output("Y")[0];

      PADDLE_ENFORCE_EQ(bit_length, 8, platform::errors::InvalidArgument(
                                           "Unsupported number quantization "
                                           "bits: %d, only 8 is supported now.",
                                           bit_length));

      if (var_quant_scales->find(y_var_name) == var_quant_scales->end()) {
        
        VLOG(1)<<"y_var_name: "<< y_var_name;
        
        auto& zeropoint_var_name = op_desc->Input("ZeroPoint")[0];
        auto* zeropoint_var = scope->FindVar(zeropoint_var_name);
        PADDLE_ENFORCE_NOT_NULL(
            zeropoint_var, "The zeropoint_var is not found.");	
        auto* zeropoint_tensor = zeropoint_var->GetMutable<LoDTensor>();

        auto zeropoint_data =
            zeropoint_tensor->mutable_data<int>(platform::CPUPlace());

        auto& scale_var_name = op_desc->Input("Scale")[0];
        auto* scale_var = scope->FindVar(scale_var_name);
        PADDLE_ENFORCE_NOT_NULL(
            scale_var, "The scale_var is not found.");	
        auto* scale_tensor = scale_var->GetMutable<LoDTensor>();
        auto scale_data =
            scale_tensor->mutable_data<float>(platform::CPUPlace());
        PADDLE_ENFORCE_EQ(scale_tensor->numel(), zeropoint_tensor->numel(), platform::errors::InvalidArgument(
                                                "scale vec should be same size as zeropoint vec"
                                                "but now scale size is: %d, zeropoint size is: %d.",
                                                scale_tensor->numel(), zeropoint_tensor->numel()));
              
        if (quant_axis < 0 && scale_tensor->numel() != 1 ){
          std::cout<<"This is big error"<<std::endl;
        } 
        size_t scale_zeropoint_size = scale_tensor->numel();
        // std::vector<float> scale_zero_vec(scale_zeropoint_size*2, 0.0f);
        std::vector<float> scale_zero_vec(scale_zeropoint_size, 0.0f);
        
        VLOG(1)<<"zeropoint_data: ";
        
        for (size_t i = 0; i < scale_zeropoint_size; i++){
          scale_zero_vec[i] = static_cast<float>(zeropoint_data[i]);
          VLOG(1)<<scale_zero_vec[i]<<" ";
        }  
        
        VLOG(1)<<"scale_data: ";
        
        // for (size_t i = 0; i < scale_zeropoint_size; i++){
        //   scale_zero_vec[i + scale_zeropoint_size] = scale_data[i];
        //   VLOG(0)<<scale_zero_vec[i + scale_zeropoint_size] <<" ";
        // }       

        for (size_t i = 0; i < scale_zeropoint_size; i++){
          scale_zero_vec[i] = 1.0 / scale_data[i];
          VLOG(1)<<scale_zero_vec[i] <<" ";
        }
        // var_quant_scales->insert(std::make_pair(y_var_name, scale_zero_vec));
        var_quant_scales->insert(std::make_pair(y_var_name, scale_zero_vec));
      }
    }
  }

//   SaveInfoInTheFirstOp(graph, "has_quant_info", "var_quant_scales",
//                        var_quant_scales);
}

void QuantDequantMkldnnV2FusePass::RemoveQuantDequantLinearOps(
    ir::Graph* graph, Scope* scope) const {
  VLOG(0) << "Fuse quantize_linear->dequantize_linear ops";
  GraphPatternDetector gpd;
  patterns::QuantizeDequantizeLinearPattern qdq_pattern(gpd.mutable_pattern(),
                                                        qdq_name_scope_);
  qdq_pattern();
  int found_quantize_dequantize_linear_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(1) << "handle quantize dequantize linear fuse pass";
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_in_x, quantize_linear_in_x,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_in_scale,
                              quantize_linear_in_scale, qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_in_zeropoint,
                              quantize_linear_in_zeropoint, qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_op, quantize_linear_op,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_out, quantize_linear_out,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_op, dequantize_linear_op,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_out, dequantize_linear_out,
                              qdq_pattern);

    prev_op->Op()->RenameOutput(quantize_linear_in_x->Name(), dequantize_linear_out->Name());
    
    VLOG(0) << "pre op output is reset to dequantize_linear output";
    IR_NODE_LINK_TO(prev_op, dequantize_linear_out);
    GraphSafeRemoveNodes(graph,
                         {quantize_linear_in_x, quantize_linear_in_scale,
                          quantize_linear_in_zeropoint, quantize_linear_op,
                          quantize_linear_out, dequantize_linear_op});

    found_quantize_dequantize_linear_count++;
  };

  gpd(graph, handler);
  AddStatis(found_quantize_dequantize_linear_count);
}

void QuantDequantMkldnnV2FusePass::RemoveDequantLinearOps(
    ir::Graph* graph, Scope* scope) const {
  GraphPatternDetector gpd;
  patterns::DequantizeLinearPattern dq_pattern(gpd.mutable_pattern(),
                                                        dq_name_scope_);
  dq_pattern();                                                          
  int found_dequantize_linear_count = 0;
  VLOG(0) << "handle removing dequantize_linear pass";
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(1) << "handle dequantize_linear removing pass";
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_in_x, dequantize_linear_in_x,
                              dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_in_scale,
                              dequantize_linear_in_scale, dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_in_zeropoint,
                              dequantize_linear_in_zeropoint, dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_op, dequantize_linear_op,
                              dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_out, dequantize_linear_out,
                              dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op,
                              dq_pattern);
    next_op->Op()->RenameInput(dequantize_linear_out->Name(), dequantize_linear_in_x->Name());
    
    IR_NODE_LINK_TO(dequantize_linear_in_x, next_op);

    GraphSafeRemoveNodes(graph,
                         {dequantize_linear_in_scale,
                          dequantize_linear_in_zeropoint, dequantize_linear_op, dequantize_linear_out});
 
    found_dequantize_linear_count++;

    //     // Create eltwise_y (conv bias) variable
    // VarDesc eltwise_y_in_desc(
    //     patterns::PDNodeName("fuse_conv_bn", conv_type() + "_eltwise_y_in"));
    // eltwise_y_in_desc.SetShape(phi::vectorize(bn_bias_tensor->dims()));
    // eltwise_y_in_desc.SetDataType(
    //     framework::TransToProtoVarType(bn_bias_tensor->dtype()));
    // eltwise_y_in_desc.SetLoDLevel(bn_bias->Var()->GetLoDLevel());
    // eltwise_y_in_desc.SetPersistable(true);
    // auto* eltwise_y_in_node = g->CreateVarNode(&eltwise_y_in_desc);
    // auto* eltwise_y_in_tensor =
    //     scope->Var(eltwise_y_in_node->Name())->GetMutable<LoDTensor>();

    // // Initialize eltwise_y
    // eltwise_y_in_tensor->Resize(bn_bias_tensor->dims());
    // std::fill_n(eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
    //             eltwise_y_in_tensor->numel(), 0.0f);
  };

  gpd(graph, handler);
  AddStatis(found_dequantize_linear_count);
}

// I just found I could not literally reuse achun's dequantization, cause I don't have weight threshold
//void QuantDequantMkldnnV2FusePass::DequantizeOpWeights(
//    Node* op_node, Scope* scope, const std::string& weight_name,
//    const std::string& output_name,
//    const std::unordered_map<std::string, std::vector<float>>&
//        weight_thresholds) const {
//  auto* op_desc = op_node->Op();
//  std::string weight_var_name = op_desc->Input(weight_name)[0];
//  std::string output_var_name = op_desc->Output(output_name)[0];
//
//  std::vector<float> scales;
//  auto iter = weight_thresholds.find(output_var_name);
//  if (iter != weight_thresholds.end()) {
//    scales = iter->second;
//  } else {
//    PADDLE_THROW(paddle::platform::errors::Fatal(
//        "Could not find threshold information for [%s] var, please check if "
//        "the model is correct.",
//        output_var_name));
//  }
//
//  auto* var = scope->FindVar(weight_var_name);
//  PADDLE_ENFORCE_NOT_NULL(
//      var, "The input persistable var of %s op is not found.", op_desc->Type());
//  auto* weight_tensor = var->GetMutable<LoDTensor>();
//  const auto weight_dims = weight_tensor->dims();
//
//  const int size = scales.size();
//  if (size == 1 || size == weight_dims[0]) {
//    auto* weight_data =
//        weight_tensor->mutable_data<float>(platform::CPUPlace());
//    for (int i = 0; i < weight_tensor->numel(); i++) {
//      weight_data[i] /= 127;
//    }
//
//    TransposeWeight(weight_tensor);
//
//    if (size == 1) {
//      for (int i = 0; i < weight_tensor->numel(); i++) {
//        weight_data[i] *= scales[0];
//      }
//    } else {
//      for (int i = 0; i < weight_tensor->numel(); i++) {
//        weight_data[i] *= scales[i % size];
//      }
//    }
//
//    TransposeWeight(weight_tensor);
//  } else if (weight_dims.size() > 1 && size == weight_dims[1]) {
//    auto* weight_data =
//        weight_tensor->mutable_data<float>(platform::CPUPlace());
//    for (int i = 0; i < weight_tensor->numel(); i++) {
//      weight_data[i] /= 127;
//    }
//
//    int step_n = 1;
//    for (int i = 1; i < weight_dims.size(); i++) {
//      step_n *= weight_dims[i];
//    }
//    int step_c = step_n / size;
//    for (int i = 0; i < weight_dims[0]; i++) {
//      int begin_n = i * step_n;
//      for (int j = begin_n; j < begin_n + step_n; j++) {
//        for (int k = 0; k < size; k++) {
//          int begin_c = k * step_c;
//          for (int m = begin_c; m < begin_c + step_c; m++) {
//            weight_data[m] *= scales[k];
//          }
//        }
//      }
//    }
//  } else {
//    PADDLE_THROW(platform::errors::InvalidArgument(
//        "The size of weight scales vector (%d) does not "
//        "match the dimensions (%d) of the weights tensor %s.",
//        size, weight_tensor->dims().size(), weight_var_name));
//  }
//
//  weight_tensor->Resize(weight_dims);
//}
//
//void QuantDequantMkldnnV2FusePass::DequantizeWeights(
//    ir::Graph* graph, Scope* scope,
//    const std::unordered_map<std::string, std::vector<float>>&
//        weight_thresholds) const {
//  VLOG(0) << "dequantize weight for ops which has weight";
//
//  if (weight_thresholds.empty()) {
//    VLOG(0)
//        << "No need to dequantize weights because weight_thresholds is empty.";
//    return;
//  }
//
//  for (auto* op_node :
//       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
//    if (!op_node->IsOp()) continue;
//    if (op_node->Name() == "conv2d" || op_node->Name() == "depthwise_conv2d") {
//      if (IsInt8Weight(op_node, scope, "Filter")) {
//        DequantizeOpWeights(op_node, scope, "Filter", "Output",
//                            weight_thresholds);
//      }
//    } else if (op_node->Name() == "mul" || op_node->Name() == "matmul" ||
//               op_node->Name() == "matmul_v2") {
//      if (IsInt8Weight(op_node, scope, "Y")) {
//        DequantizeOpWeights(op_node, scope, "Y", "Out", weight_thresholds);
//      }
//    }
//  }
//}
//
//
//void QuantDequantMkldnnV2FusePass::UpdateActivations(ir::Graph* graph) const {
//  VLOG(0) << "update conv2d or depthwise_conv2d fused activation";
//  for (auto* op_node :
//       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
//    if (!op_node->IsOp()) continue;
//
//    if (op_node->Name() == "conv2d" || op_node->Name() == "depthwise_conv2d") {
//      auto* op_desc = op_node->Op();
//      if (!op_desc->HasAttr("fuse_activation")) {
//        std::string activation;
//        if (op_desc->GetAttrIfExists<bool>("fuse_relu")) {
//          activation = "relu";
//        } else if (op_desc->GetAttrIfExists<bool>("fuse_bcprelu")) {
//          activation = "relu6";
//          float alpha = 6.0;
//          if (op_desc->HasAttr("fuse_brelu_threshold")) {
//            alpha = BOOST_GET_CONST(float,
//                                    op_desc->GetAttr("fuse_brelu_threshold"));
//          }
//          op_node->Op()->SetAttr("fuse_alpha", alpha);
//        }
//        op_node->Op()->SetAttr("fuse_activation", "no_relu_relu6");
//      }
//    }
//  }
//}
//
//void QuantDequantMkldnnV2FusePass::RemoveCtrlVars(ir::Graph* graph) const {
//  VLOG(3) << "remove control flow variable";
//  std::unordered_set<const Node*> nodes2rm = {};
//  for (auto* op_node :
//       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
//    if (op_node->IsCtrlVar()) {
//      nodes2rm.insert(op_node);
//    }
//  }
//
//  GraphSafeRemoveNodes(graph, nodes2rm);
//}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_dequant_mkldnn_v2_fuse_pass,
              paddle::framework::ir::QuantDequantMkldnnV2FusePass);

REGISTER_PASS_CAPABILITY(quant_dequant_mkldnn_v2_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("quantize_linear", 0)
            .EQ("dequantize_linear", 0));

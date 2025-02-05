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

#include "paddle/fluid/framework/ir/xpu/decoder_attention_xpu_fuse_pass.h"

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

struct DecoderAttentionFusePattern : public PatternBase {
  DecoderAttentionFusePattern(PDPattern* pattern,
                              const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(reshape2_1);
  PATTERN_DECL_NODE(reshape2_2);
  PATTERN_DECL_NODE(reshape2_3);
  PATTERN_DECL_NODE(transpose2_1);
  PATTERN_DECL_NODE(transpose2_2);
  PATTERN_DECL_NODE(transpose2_3);
  PATTERN_DECL_NODE(qk_matmul);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(qk_softmax);
  PATTERN_DECL_NODE(qkv_matmul);
  PATTERN_DECL_NODE(transpose2_4);
  PATTERN_DECL_NODE(reshape2_4);

  // declare variable node's name
  PATTERN_DECL_NODE(input_q);
  PATTERN_DECL_NODE(input_k);
  PATTERN_DECL_NODE(input_v);
  PATTERN_DECL_NODE(reshape2_1_out);
  PATTERN_DECL_NODE(reshape2_2_out);
  PATTERN_DECL_NODE(reshape2_3_out);
  PATTERN_DECL_NODE(transpose2_1_out);
  PATTERN_DECL_NODE(transpose2_2_out);
  PATTERN_DECL_NODE(transpose2_3_out);
  PATTERN_DECL_NODE(qk_matmul_out);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(qk_softmax_out);
  PATTERN_DECL_NODE(qkv_matmul_out);
  PATTERN_DECL_NODE(transpose2_4_out);
  PATTERN_DECL_NODE(output);
};

DecoderAttentionFusePattern::DecoderAttentionFusePattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* input_q = pattern->NewNode(input_q_repr())
                      ->assert_is_op_input("reshape2", "X")
                      ->AsInput();
  auto* input_k = pattern->NewNode(input_k_repr())
                      ->assert_is_op_input("reshape2", "X")
                      ->AsInput();
  auto* input_v = pattern->NewNode(input_v_repr())
                      ->assert_is_op_input("reshape2", "X")
                      ->AsInput();
  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");
  auto* reshape2_1_out = pattern->NewNode(reshape2_1_out_repr())
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("transpose2", "X");
  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");
  auto* reshape2_2_out = pattern->NewNode(reshape2_2_out_repr())
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("transpose2", "X");
  auto* reshape2_3 =
      pattern->NewNode(reshape2_3_repr())->assert_is_op("reshape2");
  auto* reshape2_3_out = pattern->NewNode(reshape2_3_out_repr())
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("transpose2", "X");
  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis = op_desc->GetAttrIfExists<std::vector<int>>("axis");
            size_t axis_rank = axis.size();
            return axis_rank == 4 && axis[0] == 0 && axis[1] == 2 &&
                   axis[2] == 1 && axis[3] == 3;
          });

  auto* transpose2_1_out = pattern->NewNode(transpose2_1_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("matmul_v2", "X");
  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis = op_desc->GetAttrIfExists<std::vector<int>>("axis");
            size_t axis_rank = axis.size();
            return axis_rank == 4 && axis[0] == 0 && axis[1] == 2 &&
                   axis[2] == 1 && axis[3] == 3;
          });
  auto* transpose2_2_out = pattern->NewNode(transpose2_2_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("matmul_v2", "Y");
  auto* transpose2_3 =
      pattern->NewNode(transpose2_3_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis = op_desc->GetAttrIfExists<std::vector<int>>("axis");
            size_t axis_rank = axis.size();
            return axis_rank == 4 && axis[0] == 0 && axis[1] == 2 &&
                   axis[2] == 1 && axis[3] == 3;
          });
  auto* transpose2_3_out = pattern->NewNode(transpose2_3_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("matmul_v2", "Y");
  auto* qk_matmul =
      pattern->NewNode(qk_matmul_repr())->assert_is_op("matmul_v2");
  auto* qk_matmul_out = pattern->NewNode(qk_matmul_out_repr())
                            ->assert_is_op_output("matmul_v2", "Out")
                            ->assert_is_op_input("scale", "X");
  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out = pattern->NewNode(scale_out_repr())
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input("softmax", "X");
  auto* qk_softmax =
      pattern->NewNode(qk_softmax_repr())->assert_is_op("softmax");
  auto* qk_softmax_out = pattern->NewNode(qk_softmax_out_repr())
                             ->assert_is_op_output("softmax", "Out")
                             ->assert_is_op_input("matmul_v2", "X");
  auto* qkv_matmul =
      pattern->NewNode(qkv_matmul_repr())->assert_is_op("matmul_v2");
  auto* qkv_matmul_out = pattern->NewNode(qkv_matmul_out_repr())
                             ->assert_is_op_output("matmul_v2", "Out")
                             ->assert_is_op_input("transpose2", "X");
  auto* transpose2_4 =
      pattern->NewNode(transpose2_4_repr())->assert_is_op("transpose2");
  auto* transpose2_4_out = pattern->NewNode(transpose2_4_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("reshape2", "X");
  auto* reshape2_4 =
      pattern->NewNode(reshape2_4_repr())->assert_is_op("reshape2");
  auto* output = pattern->NewNode(output_repr())
                     ->AsOutput()
                     ->assert_is_op_output("reshape2", "Out");

  // link nodes
  reshape2_1->LinksFrom({input_q}).LinksTo({reshape2_1_out});
  transpose2_1->LinksFrom({reshape2_1_out}).LinksTo({transpose2_1_out});
  reshape2_2->LinksFrom({input_k}).LinksTo({reshape2_2_out});
  transpose2_2->LinksFrom({reshape2_2_out}).LinksTo({transpose2_2_out});
  qk_matmul->LinksFrom({transpose2_1_out, transpose2_2_out})
      .LinksTo({qk_matmul_out});
  scale->LinksFrom({qk_matmul_out}).LinksTo({scale_out});
  qk_softmax->LinksFrom({scale_out}).LinksTo({qk_softmax_out});
  reshape2_3->LinksFrom({input_v}).LinksTo({reshape2_3_out});
  transpose2_3->LinksFrom({reshape2_3_out}).LinksTo({transpose2_3_out});
  qkv_matmul->LinksFrom({qk_softmax_out, transpose2_3_out})
      .LinksTo({qkv_matmul_out});
  transpose2_4->LinksFrom({qkv_matmul_out}).LinksTo({transpose2_4_out});
  reshape2_4->LinksFrom({transpose2_4_out}).LinksTo({output});
}

}  // namespace patterns

void DecoderAttentionXPUFusePass::ApplyDecoderAttentionXPUFuse(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::DecoderAttentionFusePattern pattern(gpd.mutable_pattern(),
                                                name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DecoderAttentionXPUFusePass";

    // declare operator node's name
    GET_IR_NODE(reshape2_1);
    GET_IR_NODE(reshape2_2);
    GET_IR_NODE(reshape2_3);
    GET_IR_NODE(transpose2_1);
    GET_IR_NODE(transpose2_2);
    GET_IR_NODE(transpose2_3);
    GET_IR_NODE(qk_matmul);
    GET_IR_NODE(scale);
    GET_IR_NODE(qk_softmax);
    GET_IR_NODE(qkv_matmul);
    GET_IR_NODE(transpose2_4);
    GET_IR_NODE(reshape2_4);

    // declare variable node's name
    GET_IR_NODE(input_q);
    GET_IR_NODE(input_k);
    GET_IR_NODE(input_v);
    GET_IR_NODE(reshape2_1_out);
    GET_IR_NODE(reshape2_2_out);
    GET_IR_NODE(reshape2_3_out);
    GET_IR_NODE(transpose2_1_out);
    GET_IR_NODE(transpose2_2_out);
    GET_IR_NODE(transpose2_3_out);
    GET_IR_NODE(qk_matmul_out);
    GET_IR_NODE(scale_out);
    GET_IR_NODE(qk_softmax_out);
    GET_IR_NODE(qkv_matmul_out);
    GET_IR_NODE(transpose2_4_out);
    GET_IR_NODE(output);

    // Generate fuse op
    auto* scope = param_scope();
    auto* block = reshape2_1->Op()->Block();
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("qkv_attention_xpu");

    // set input of fuse_op
    fused_op_desc.SetInput("q", {input_q->Name()});
    fused_op_desc.SetInput("k", {input_k->Name()});
    fused_op_desc.SetInput("v", {input_v->Name()});
    std::unordered_map<std::string, std::vector<float>> var_quant_scales =
        GetQuantInfoFromTheGraph(graph, "has_quant_info", "var_quant_scales");
    // recorded q/k/v max, qk_max, and qkv_max
    std::vector<Node*> input_max_nodes;
    if (var_quant_scales.find(input_q->Name()) != var_quant_scales.end() &&
        var_quant_scales.find(input_k->Name()) != var_quant_scales.end() &&
        var_quant_scales.find(input_v->Name()) != var_quant_scales.end() &&
        var_quant_scales.find(qk_matmul_out->Name()) !=
            var_quant_scales.end() &&
        var_quant_scales.find(qkv_matmul_out->Name()) !=
            var_quant_scales.end()) {
      std::vector<float> input_max_vec;
      input_max_vec.push_back(var_quant_scales.at(input_q->Name())[0]);
      input_max_vec.push_back(var_quant_scales.at(input_k->Name())[0]);
      input_max_vec.push_back(var_quant_scales.at(input_v->Name())[0]);
      input_max_vec.push_back(var_quant_scales.at(qk_matmul_out->Name())[0]);
      input_max_vec.push_back(var_quant_scales.at(qkv_matmul_out->Name())[0]);
      std::vector<std::string> quant_max_names = {
          "q_max", "k_max", "v_max", "qk_max", "qkv_max"};
      for (size_t i = 0; i < input_max_vec.size(); i++) {
        std::string input_max_name =
            input_q->Name() + "_" + std::to_string(i) + "_max_in";
        int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
        VarDesc input_max_desc(input_max_name);
        input_max_desc.SetPersistable(true);
        input_max_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
        input_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
        Node* input_max_in = graph->CreateVarNode(&input_max_desc);
        auto* block_input_max_in_desc = block->Var(input_max_name);
        block_input_max_in_desc->SetPersistable(input_max_desc.Persistable());
        block_input_max_in_desc->SetShape(input_max_desc.GetShape());
        block_input_max_in_desc->SetDataType(input_max_desc.GetDataType());
        phi::DenseTensor input_max_in_cpu_tensor;
        auto* cpu_ctx = static_cast<phi::CPUContext*>(
            phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
        input_max_in_cpu_tensor.set_type(phi::DataType::FLOAT32);
        input_max_in_cpu_tensor.Resize({max_ptr_size});
        std::vector<float> input_max(max_ptr_size, input_max_vec[i]);
        memcpy(cpu_ctx->Alloc<float>(&input_max_in_cpu_tensor),
               input_max.data(),
               max_ptr_size * sizeof(float));
        Assign(input_max_in_cpu_tensor,
               scope->Var(input_max_name)->GetMutable<phi::DenseTensor>());
        fused_op_desc.SetInput(quant_max_names[i], {input_max_name});

        input_max_nodes.push_back(input_max_in);
      }
    }

    // set attributes of fuse_op
    float scale_val = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));
    fused_op_desc.SetAttr("alpha", scale_val);
    fused_op_desc.SetAttr(
        "head_num", static_cast<int>(transpose2_1_out->Var()->GetShape()[1]));
    fused_op_desc.SetAttr(
        "head_dim", static_cast<int>(transpose2_1_out->Var()->GetShape()[3]));
    // In this pattern, there is only one possible situation.
    fused_op_desc.SetAttr("qkv_fc_fusion", false);

    // TODO(tianrui): support more out_dtype
    fused_op_desc.SetAttr("out_dtype", input_q->Var()->GetDataType());

    // set output of fuse_op
    fused_op_desc.SetOutput("qkv", {output->Name()});

    auto* fused_op = graph->CreateOpNode(&fused_op_desc);

    IR_NODE_LINK_TO(input_q, fused_op);
    IR_NODE_LINK_TO(input_k, fused_op);
    IR_NODE_LINK_TO(input_v, fused_op);
    IR_NODE_LINK_TO(fused_op, output);
    for (size_t i = 0; i < input_max_nodes.size(); i++) {
      IR_NODE_LINK_TO(input_max_nodes[i], fused_op);
    }

    // delete useless node
    std::unordered_set<const Node*> del_node_set;
    del_node_set.insert(reshape2_1);
    del_node_set.insert(reshape2_2);
    del_node_set.insert(reshape2_3);
    del_node_set.insert(transpose2_1);
    del_node_set.insert(transpose2_2);
    del_node_set.insert(transpose2_3);
    del_node_set.insert(qk_matmul);
    del_node_set.insert(scale);
    del_node_set.insert(qk_softmax);
    del_node_set.insert(qkv_matmul);
    del_node_set.insert(transpose2_4);
    del_node_set.insert(reshape2_4);
    del_node_set.insert(reshape2_1_out);
    del_node_set.insert(reshape2_2_out);
    del_node_set.insert(reshape2_3_out);
    del_node_set.insert(transpose2_1_out);
    del_node_set.insert(transpose2_2_out);
    del_node_set.insert(transpose2_3_out);
    del_node_set.insert(qk_matmul_out);
    del_node_set.insert(scale_out);
    del_node_set.insert(qk_softmax_out);
    del_node_set.insert(qkv_matmul_out);
    del_node_set.insert(transpose2_4_out);

    GraphSafeRemoveNodes(graph, del_node_set);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void DecoderAttentionXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  ApplyDecoderAttentionXPUFuse(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(decoder_attention_xpu_fuse_pass,
              paddle::framework::ir::DecoderAttentionXPUFusePass);

REGISTER_PASS_CAPABILITY(decoder_attention_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "qkv_attention_xpu", 0));

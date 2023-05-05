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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FusedMultiTransformerPattern : public PatternBase {
  FusedMultiTransformerPattern(PDPattern* pattern,
                               const std::string& name_scope,
                               bool with_pre_caches,
                               bool with_rotary_pos_emb,
                               bool with_time_step,
                               bool with_seq_lengths,
                               bool with_src_mask);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_mt);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(ln_scale);
  PATTERN_DECL_NODE(ln_bias);
  PATTERN_DECL_NODE(qkv_w);
  PATTERN_DECL_NODE(qkv_bias);
  PATTERN_DECL_NODE(pre_caches);
  PATTERN_DECL_NODE(rotary_pos_emb);
  PATTERN_DECL_NODE(time_step);
  PATTERN_DECL_NODE(seq_lengths);
  PATTERN_DECL_NODE(src_mask);
  PATTERN_DECL_NODE(out_linear_w);
  PATTERN_DECL_NODE(out_linear_bias);
  PATTERN_DECL_NODE(ffn_ln_scale);
  PATTERN_DECL_NODE(ffn_ln_bias);
  PATTERN_DECL_NODE(ffn1_w);
  PATTERN_DECL_NODE(ffn1_bias);
  PATTERN_DECL_NODE(ffn2_w);
  PATTERN_DECL_NODE(ffn2_bias);
  PATTERN_DECL_NODE(out);

 private:
  bool with_pre_caches_{false};
  bool with_rotary_pos_emb_{false};
  bool with_time_step_{false};
  bool with_seq_lengths_{false};
  bool with_src_mask_{false};
};

FusedMultiTransformerPattern::FusedMultiTransformerPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    bool with_pre_caches,
    bool with_rotary_pos_emb,
    bool with_time_step,
    bool with_seq_lengths,
    bool with_src_mask)
    : PatternBase(pattern, name_scope, name_scope),
      with_pre_caches_(with_pre_caches),
      with_rotary_pos_emb_(with_rotary_pos_emb),
      with_time_step_(with_time_step),
      with_seq_lengths_(with_seq_lengths),
      with_src_mask_(with_src_mask) {
  std::string op_type = "fused_multi_transformer";
  auto* fused_mt = pattern->NewNode(fused_mt_repr())->assert_is_op(op_type);
  // inputs and outputs
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input(op_type, "X")
                ->assert_var_not_persistable();
  auto* out = pattern->NewNode(out_repr())
                  ->assert_is_op_output(op_type, "Out")
                  ->assert_var_not_persistable();
  // weights and biases
  auto* ln_scale = pattern->NewNode(ln_scale_repr())
                       ->assert_is_op_input(op_type, "LnScale")
                       ->assert_is_persistable_var()
                       ->assert_more([](Node* node) {
                         return node->Var()->GetShape().size() == 1;
                       });
  auto* ln_bias = pattern->NewNode(ln_bias_repr())
                      ->assert_is_op_input(op_type, "LnBias")
                      ->assert_is_persistable_var()
                      ->assert_more([](Node* node) {
                        return node->Var()->GetShape().size() == 1;
                      });
  auto* qkv_w = pattern->NewNode(qkv_w_repr())
                    ->assert_is_op_input(op_type, "QKVW")
                    ->assert_is_persistable_var()
                    ->assert_more([](Node* node) {
                      return node->Var()->GetShape().size() == 4;
                    });
  auto* qkv_bias = pattern->NewNode(qkv_bias_repr())
                       ->assert_is_op_input(op_type, "QKVBias")
                       ->assert_is_persistable_var()
                       ->assert_more([](Node* node) {
                         return node->Var()->GetShape().size() == 3;
                       });
  auto* out_linear_w = pattern->NewNode(out_linear_w_repr())
                           ->assert_is_op_input(op_type, "OutLinearW")
                           ->assert_is_persistable_var()
                           ->assert_more([](Node* node) {
                             return node->Var()->GetShape().size() == 2;
                           });
  auto* out_linear_bias = pattern->NewNode(out_linear_bias_repr())
                              ->assert_is_op_input(op_type, "OutLinearBias")
                              ->assert_is_persistable_var()
                              ->assert_more([](Node* node) {
                                return node->Var()->GetShape().size() == 1;
                              });
  auto* ffn_ln_scale = pattern->NewNode(ffn_ln_scale_repr())
                           ->assert_is_op_input(op_type, "FFNLnScale")
                           ->assert_is_persistable_var()
                           ->assert_more([](Node* node) {
                             return node->Var()->GetShape().size() == 1;
                           });
  auto* ffn_ln_bias = pattern->NewNode(ffn_ln_bias_repr())
                          ->assert_is_op_input(op_type, "FFNLnBias")
                          ->assert_is_persistable_var()
                          ->assert_more([](Node* node) {
                            return node->Var()->GetShape().size() == 1;
                          });
  auto* ffn1_w = pattern->NewNode(ffn1_w_repr())
                     ->assert_is_op_input(op_type, "FFN1Weight")
                     ->assert_is_persistable_var()
                     ->assert_more([](Node* node) {
                       return node->Var()->GetShape().size() == 2;
                     });
  auto* ffn1_bias = pattern->NewNode(ffn1_bias_repr())
                        ->assert_is_op_input(op_type, "FFN1Bias")
                        ->assert_is_persistable_var()
                        ->assert_more([](Node* node) {
                          return node->Var()->GetShape().size() == 1;
                        });
  auto* ffn2_w = pattern->NewNode(ffn2_w_repr())
                     ->assert_is_op_input(op_type, "FFN2Weight")
                     ->assert_is_persistable_var()
                     ->assert_more([](Node* node) {
                       return node->Var()->GetShape().size() == 2;
                     });
  auto* ffn2_bias = pattern->NewNode(ffn2_bias_repr())
                        ->assert_is_op_input(op_type, "FFN2Bias")
                        ->assert_is_persistable_var()
                        ->assert_more([](Node* node) {
                          return node->Var()->GetShape().size() == 1;
                        });

  std::vector<PDNode*> input_vars{x,
                                  ln_scale,
                                  ln_bias,
                                  qkv_w,
                                  qkv_bias,
                                  out_linear_w,
                                  out_linear_bias,
                                  ffn_ln_scale,
                                  ffn_ln_bias,
                                  ffn1_w,
                                  ffn1_bias,
                                  ffn2_w,
                                  ffn2_bias};
  std::vector<PDNode*> output_vars{out};

  // optional node
  PDNode* pre_caches = nullptr;
  PDNode* rotary_pos_emb = nullptr;
  PDNode* time_step = nullptr;
  PDNode* seq_lengths = nullptr;
  PDNode* src_mask = nullptr;
  if (with_pre_caches_) {
    pre_caches = pattern->NewNode(pre_caches_repr())
                     ->assert_is_op_input(op_type, "PreCaches")
                     ->assert_var_not_persistable();
    input_vars.push_back(pre_caches);
  }
  if (with_rotary_pos_emb_) {
    rotary_pos_emb = pattern->NewNode(rotary_pos_emb_repr())
                         ->assert_is_op_input(op_type, "RotaryPosEmb")
                         ->assert_var_not_persistable();
    input_vars.push_back(rotary_pos_emb);
  }
  if (with_time_step_) {
    time_step = pattern->NewNode(time_step_repr())
                    ->assert_is_op_input(op_type, "TimeStep")
                    ->assert_var_not_persistable();
    input_vars.push_back(time_step);
  }
  if (with_seq_lengths_) {
    seq_lengths = pattern->NewNode(seq_lengths_repr())
                      ->assert_is_op_input(op_type, "SeqLengths")
                      ->assert_var_not_persistable();
    input_vars.push_back(seq_lengths);
  }
  if (with_src_mask_) {
    src_mask = pattern->NewNode(src_mask_repr())
                   ->assert_is_op_input(op_type, "SrcMask")
                   ->assert_var_not_persistable();
    input_vars.push_back(src_mask);
  }

  fused_mt->LinksFrom(input_vars).LinksTo(output_vars);
}

}  // namespace patterns

/*
1. transpose and quantify the weights of fused_multi_transformer op from fp32 to
int16
*/
class FusedMultiTransformerXPUQuantPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                bool with_pre_caches,
                bool with_rotary_pos_emb,
                bool with_time_step,
                bool with_seq_lengths,
                bool with_src_mask) const;

  const std::string name_scope_{"fused_multi_transformer_xpu_quant_pass"};
};

void FusedMultiTransformerXPUQuantPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  VLOG(3) << "in FusedMultiTransformerXPUQuantPass::ApplyImpl";

  int found_subgraph_count = 0;
  for (bool with_time_step : {true, false}) {
    found_subgraph_count +=
        ApplyImpl(graph, false, false, with_time_step, false, true);
  }
  AddStatis(found_subgraph_count);
}

int FusedMultiTransformerXPUQuantPass::ApplyImpl(ir::Graph* graph,
                                                 bool with_pre_caches,
                                                 bool with_rotary_pos_emb,
                                                 bool with_time_step,
                                                 bool with_seq_lengths,
                                                 bool with_src_mask) const {
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerPattern pattern(gpd.mutable_pattern(),
                                                 name_scope_,
                                                 with_pre_caches,
                                                 with_rotary_pos_emb,
                                                 with_time_step,
                                                 with_seq_lengths,
                                                 with_src_mask);
  int quant_weight_bits =
      Has("quant_weight_bits") ? Get<int>("quant_weight_bits") : -1;

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FusedMultiTransformerXPUQuantPass fuse";

    GET_IR_NODE(x);
    GET_IR_NODE(ln_scale);
    GET_IR_NODE(ln_bias);
    GET_IR_NODE(qkv_w);
    GET_IR_NODE(qkv_bias);
    GET_IR_NODE(pre_caches);
    GET_IR_NODE(rotary_pos_emb);
    GET_IR_NODE(time_step);
    GET_IR_NODE(seq_lengths);
    GET_IR_NODE(src_mask);
    GET_IR_NODE(out_linear_w);
    GET_IR_NODE(out_linear_bias);
    GET_IR_NODE(ffn_ln_scale);
    GET_IR_NODE(ffn_ln_bias);
    GET_IR_NODE(ffn1_w);
    GET_IR_NODE(ffn1_bias);
    GET_IR_NODE(ffn2_w);
    GET_IR_NODE(ffn2_bias);
    GET_IR_NODE(out);
    GET_IR_NODE(fused_mt);
    auto* block = fused_mt->Op()->Block();
    auto* scope = param_scope();

    // quant weight nodes
    // w_nodes_vec: [QKVW, OutLinearW, FFN1Weight, FFN2Weight]
    std::vector<std::vector<Node*>> w_nodes_vec(4);
    std::vector<std::vector<Node*>> w_intx_nodes_vec(4);
    std::vector<std::vector<Node*>> w_max_nodes_vec(4);
    std::vector<std::vector<std::string>> w_intx_names_vec(4);
    std::vector<std::vector<std::string>> w_max_names_vec(4);
    auto quant_func = [&](const std::string& input_name,
                          std::vector<Node*>* w_nodes,
                          std::vector<Node*>* w_intx_nodes,
                          std::vector<Node*>* w_max_nodes,
                          std::vector<std::string>* w_intx_names,
                          std::vector<std::string>* w_max_names,
                          bool need_transpose) {
      auto w_names = fused_mt->Op()->Input(input_name);
      for (auto w_name : w_names) {
        Node* w_node = FindNodeWithName(graph, w_name);
        Node* w_intx = nullptr;
        Node* w_max = nullptr;
        PADDLE_ENFORCE_NE(
            w_node,
            nullptr,
            platform::errors::Fatal("w node should not be nullptr"));
        if (quant_weight_bits == 8) {
          PrepareWeight<int8_t>(
              graph, scope, block, w_node, &w_intx, &w_max, need_transpose);
        } else {
          PrepareWeight<int16_t>(
              graph, scope, block, w_node, &w_intx, &w_max, need_transpose);
        }
        w_nodes->push_back(w_node);
        w_intx_nodes->push_back(w_intx);
        w_max_nodes->push_back(w_max);
      }
      for (size_t i = 0; i < w_names.size(); ++i) {
        w_intx_names->push_back(w_intx_nodes->at(i)->Name());
        w_max_names->push_back(w_max_nodes->at(i)->Name());
      }
      PADDLE_ENFORCE_EQ(
          w_names.size(),
          w_nodes->size(),
          platform::errors::Fatal(
              "The size of w_names(%d) should be equal to w_nodes(%d)",
              static_cast<int>(w_names.size()),
              static_cast<int>(w_nodes->size())));
      PADDLE_ENFORCE_EQ(
          w_names.size(),
          w_intx_nodes->size(),
          platform::errors::Fatal(
              "The size of w_names(%d) should be equal to w_intx_nodes(%d)",
              static_cast<int>(w_names.size()),
              static_cast<int>(w_intx_nodes->size())));
      PADDLE_ENFORCE_EQ(
          w_names.size(),
          w_max_nodes->size(),
          platform::errors::Fatal(
              "The size of w_names(%d) should be equal to w_max_nodes(%d)",
              static_cast<int>(w_names.size()),
              static_cast<int>(w_max_nodes->size())));
      PADDLE_ENFORCE_EQ(
          w_names.size(),
          w_intx_names->size(),
          platform::errors::Fatal(
              "The size of w_names(%d) should be equal to w_intx_names(%d)",
              static_cast<int>(w_names.size()),
              static_cast<int>(w_intx_names->size())));
      PADDLE_ENFORCE_EQ(
          w_names.size(),
          w_max_names->size(),
          platform::errors::Fatal(
              "The size of w_names(%d) should be equal to w_max_names(%d)",
              static_cast<int>(w_names.size()),
              static_cast<int>(w_max_names->size())));
    };
    quant_func("QKVW",
               &(w_nodes_vec[0]),
               &(w_intx_nodes_vec[0]),
               &(w_max_nodes_vec[0]),
               &(w_intx_names_vec[0]),
               &(w_max_names_vec[0]),
               false);
    quant_func("OutLinearW",
               &(w_nodes_vec[1]),
               &(w_intx_nodes_vec[1]),
               &(w_max_nodes_vec[1]),
               &(w_intx_names_vec[1]),
               &(w_max_names_vec[1]),
               true);
    quant_func("FFN1Weight",
               &(w_nodes_vec[2]),
               &(w_intx_nodes_vec[2]),
               &(w_max_nodes_vec[2]),
               &(w_intx_names_vec[2]),
               &(w_max_names_vec[2]),
               true);
    quant_func("FFN2Weight",
               &(w_nodes_vec[3]),
               &(w_intx_nodes_vec[3]),
               &(w_max_nodes_vec[3]),
               &(w_intx_names_vec[3]),
               &(w_max_names_vec[3]),
               true);

    // cast some nodes to fp32 nodes
    std::vector<Node*> fp32_nodes;
    auto cast_tofp32_func = [&](const std::string& input_name) {
      auto names = fused_mt->Op()->Input(input_name);
      for (auto name : names) {
        auto* curr_tensor = scope->Var(name)->GetMutable<phi::DenseTensor>();
        PADDLE_ENFORCE_NE(
            curr_tensor,
            nullptr,
            platform::errors::Fatal("tensor node should not be nullptr"));
        CastToFp32(curr_tensor);

        Node* curr_node = FindNodeWithName(graph, name);
        fp32_nodes.push_back(curr_node);
      }
    };
    cast_tofp32_func("LnScale");
    cast_tofp32_func("LnBias");
    cast_tofp32_func("QKVBias");
    cast_tofp32_func("OutLinearBias");
    cast_tofp32_func("FFNLnScale");
    cast_tofp32_func("FFNLnBias");
    cast_tofp32_func("FFN1Bias");
    cast_tofp32_func("FFN2Bias");

    // Generate fused_multi_transformer_xpu op inplace
    fused_mt->RenameOp("fused_multi_transformer_xpu");
    framework::OpDesc* fused_mt_xpu_op_desc = fused_mt->Op();
    fused_mt_xpu_op_desc->SetType("fused_multi_transformer_xpu");
    std::unordered_map<std::string, std::vector<std::string>> name_caches;
    for (auto key : fused_mt_xpu_op_desc->InputNames()) {
      name_caches.insert({key, fused_mt_xpu_op_desc->Input(key)});
    }
    for (auto key : fused_mt_xpu_op_desc->OutputNames()) {
      name_caches.insert({key, fused_mt_xpu_op_desc->Output(key)});
    }
    fused_mt_xpu_op_desc->MutableInputs()->clear();
    fused_mt_xpu_op_desc->MutableOutputs()->clear();
    fused_mt_xpu_op_desc->SetInput("x", name_caches.at("X"));
    fused_mt_xpu_op_desc->SetInput("ln_scale", name_caches.at("LnScale"));
    fused_mt_xpu_op_desc->SetInput("ln_bias", name_caches.at("LnBias"));
    fused_mt_xpu_op_desc->SetInput("qkv_bias", name_caches.at("QKVBias"));
    if (name_caches.count("CacheKV") > 0) {
      fused_mt_xpu_op_desc->SetInput("cache_kv", name_caches.at("CacheKV"));
    }
    if (pre_caches) {
      fused_mt_xpu_op_desc->SetInput("pre_caches", name_caches.at("PreCaches"));
    }
    if (rotary_pos_emb) {
      fused_mt_xpu_op_desc->SetInput("rotary_pos_emb",
                                     name_caches.at("RotaryPosEmb"));
    }
    if (time_step) {
      fused_mt_xpu_op_desc->SetInput("time_step", name_caches.at("TimeStep"));
    }
    if (seq_lengths) {
      fused_mt_xpu_op_desc->SetInput("seq_lengths",
                                     name_caches.at("SeqLengths"));
    }
    if (src_mask) {
      fused_mt_xpu_op_desc->SetInput("src_mask", name_caches.at("SrcMask"));
    }
    fused_mt_xpu_op_desc->SetInput("out_linear_bias",
                                   name_caches.at("OutLinearBias"));
    fused_mt_xpu_op_desc->SetInput("ffn_ln_scale",
                                   name_caches.at("FFNLnScale"));
    fused_mt_xpu_op_desc->SetInput("ffn_ln_bias", name_caches.at("FFNLnBias"));
    fused_mt_xpu_op_desc->SetInput("ffn1_bias", name_caches.at("FFN1Bias"));
    fused_mt_xpu_op_desc->SetInput("ffn2_bias", name_caches.at("FFN2Bias"));
    fused_mt_xpu_op_desc->SetOutput("cache_kv_out",
                                    name_caches.at("CacheKVOut"));
    fused_mt_xpu_op_desc->SetOutput("out", name_caches.at("Out"));

    fused_mt_xpu_op_desc->SetInput("qkvw", w_intx_names_vec[0]);
    fused_mt_xpu_op_desc->SetInput("qkvw_max", w_max_names_vec[0]);
    fused_mt_xpu_op_desc->SetInput("out_linear_w", w_intx_names_vec[1]);
    fused_mt_xpu_op_desc->SetInput("out_linear_wmax", w_max_names_vec[1]);
    fused_mt_xpu_op_desc->SetInput("ffn1_weight", w_intx_names_vec[2]);
    fused_mt_xpu_op_desc->SetInput("ffn1_weight_max", w_max_names_vec[2]);
    fused_mt_xpu_op_desc->SetInput("ffn2_weight", w_intx_names_vec[3]);
    fused_mt_xpu_op_desc->SetInput("ffn2_weight_max", w_max_names_vec[3]);
    if (!fused_mt_xpu_op_desc->HasAttr("rotary_emb_dims")) {
      fused_mt_xpu_op_desc->SetAttr("rotary_emb_dims", 0);
    }
    // unlink QKVW/OutLinearW/FFN1Weight/FFN2Weight from fused_mt_xpu
    for (auto nodes : w_nodes_vec) {
      for (auto node : nodes) {
        IR_NODE_UNLINK(node, fused_mt);
      }
    }
    // link int16 format of QKVW/OutLinearW/FFN1Weight/FFN2Weight to
    // fused_mt_xpu
    for (auto nodes : w_intx_nodes_vec) {
      for (auto node : nodes) {
        IR_NODE_LINK_TO(node, fused_mt);
      }
    }
    // link QKVWMax/OutLinearWMax/FFN1WeightMax/FFN2WeightMax to fused_mt_xpu
    for (auto nodes : w_max_nodes_vec) {
      for (auto node : nodes) {
        IR_NODE_LINK_TO(node, fused_mt);
      }
    }

    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_multi_transformer_xpu_quant_pass,
              paddle::framework::ir::FusedMultiTransformerXPUQuantPass);

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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"

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

struct FusedMultiTransformerInt8AssignPattern : public PatternBase {
  FusedMultiTransformerInt8AssignPattern(PDPattern* pattern,
                                         const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(assign);
  // declare variable node's name
  PATTERN_DECL_NODE(assign_out);
};

FusedMultiTransformerInt8AssignPattern::FusedMultiTransformerInt8AssignPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* assign = pattern->NewNode(assign_repr())
                     ->assert_is_op("assign")
                     ->assert_more([&](Node* node) {
                       auto pre_op_nodes = node->inputs[0]->inputs;
                       return pre_op_nodes.size() == 1 &&
                              pre_op_nodes[0]->Op()->Type() ==
                                  "fused_multi_transformer_int8";
                     });
  auto* assign_out =
      pattern->NewNode(assign_out_repr())->assert_is_op_output("assign", "Out");

  assign->LinksTo({assign_out});
}

struct FusedMultiTransformerInt8Pattern : public PatternBase {
  FusedMultiTransformerInt8Pattern(PDPattern* pattern,
                                   const std::string& name_scope,
                                   bool with_pre_caches,
                                   bool with_rotary_pos_emb,
                                   bool with_time_step,
                                   bool with_seq_lengths,
                                   bool with_src_mask);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_mt_int8);
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

FusedMultiTransformerInt8Pattern::FusedMultiTransformerInt8Pattern(
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
  std::string op_type = "fused_multi_transformer_int8";
  auto* fused_mt_int8 =
      pattern->NewNode(fused_mt_int8_repr())->assert_is_op(op_type);
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

  fused_mt_int8->LinksFrom(input_vars).LinksTo(output_vars);
}

}  // namespace patterns

class FusedMultiTransformerInt8XPUQuantPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  /*
  Origin subgraph:
              fused_multi_transformer_int8
               |        |        |
             assign   assign    ...
               |        |        |
             gather   gather    ...

  Fused subgraph:
              fused_multi_transformer_int8
 */
  void RemoveAssignGather(ir::Graph* graph) const;

  /*
  Origin subgraph:
              fused_multi_transformer_int8

  Fused subgraph:
              fused_multi_transformer_int8_xpu
 */
  int FusedMultiTransformerInt8(ir::Graph* graph,
                                bool with_pre_caches,
                                bool with_rotary_pos_emb,
                                bool with_time_step,
                                bool with_seq_lengths,
                                bool with_src_mask) const;

  const std::string name_scope_{"fused_multi_transformer_int8_xpu_quant_pass"};
};

void FusedMultiTransformerInt8XPUQuantPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  VLOG(3) << "in FusedMultiTransformerInt8XPUQuantPass::ApplyImpl";

  int found_subgraph_count = 0;
  RemoveAssignGather(graph);
  for (bool with_time_step : {true, false}) {
    found_subgraph_count += FusedMultiTransformerInt8(
        graph, false, false, with_time_step, false, true);
  }
  AddStatis(found_subgraph_count);
}

void FusedMultiTransformerInt8XPUQuantPass::RemoveAssignGather(
    ir::Graph* graph) const {
  // detect assign + gather
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerInt8AssignPattern pattern(
      gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(1) << "handle RemoveAssignGather";
    GET_IR_NODE(assign);
    GET_IR_NODE(assign_out);
    // Assign_out may not link to gather, so we find gather by input name.
    auto next_ops = FindOpNodeByInputName(graph, assign_out->Name());
    if (next_ops.size() != 1 || next_ops[0]->Name() != "gather") return;
    auto* gather = next_ops[0];

    // "assign_out" is used in multi blocks. "assign_out" should be reserved.
    auto* gather_index = gather->inputs[0];
    auto* assign_in = assign->inputs[0];
    auto* fused_multi_transformer_int8 = assign_in->inputs[0];
    fused_multi_transformer_int8->Op()->Rename(assign_in->Name(),
                                               assign_out->Name());
    fused_multi_transformer_int8->Op()->SetInput("gather_index",
                                                 gather->Op()->Input("Index"));
    fused_multi_transformer_int8->Op()->SetAttr("gather_axis",
                                                gather->Op()->GetAttr("axis"));
    IR_NODE_LINK_TO(gather_index, fused_multi_transformer_int8);
    IR_NODE_LINK_TO(fused_multi_transformer_int8, assign_out);

    std::unordered_set<const Node*> delete_nodes{assign, assign_in, gather};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

int FusedMultiTransformerInt8XPUQuantPass::FusedMultiTransformerInt8(
    ir::Graph* graph,
    bool with_pre_caches,
    bool with_rotary_pos_emb,
    bool with_time_step,
    bool with_seq_lengths,
    bool with_src_mask) const {
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerInt8Pattern pattern(gpd.mutable_pattern(),
                                                     name_scope_,
                                                     with_pre_caches,
                                                     with_rotary_pos_emb,
                                                     with_time_step,
                                                     with_seq_lengths,
                                                     with_src_mask);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FusedMultiTransformerInt8 fuse";

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
    GET_IR_NODE(fused_mt_int8);
    auto* block = fused_mt_int8->Op()->Block();
    auto* scope = param_scope();

    // input max nodes
    std::vector<std::vector<Node*>> input_max_nodes_vec(4);
    std::vector<std::vector<std::string>> input_max_names_vec(4);
    std::vector<std::vector<Node*>> weight_max_nodes_vec(4);
    std::vector<std::vector<std::string>> weight_max_names_vec(4);
    std::vector<std::vector<Node*>> old_weight_max_nodes_vec(4);
    std::vector<std::vector<std::string>> old_weight_max_names_vec(4);

    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    auto attr2weight = [&](const std::string& src_name,
                           std::vector<Node*>* input_max_node,
                           std::vector<std::string>* input_max_name) {
      auto GetPrefixWithoutHash = [](const std::string& name) -> std::string {
        std::size_t found = name.find("_#");
        return found == std::string::npos ? name : name.substr(0, found);
      };

      std::vector<float> in_scale_data = PADDLE_GET_CONST(
          std::vector<float>, fused_mt_int8->Op()->GetAttr(src_name, false));
      int in_scale_data_size = in_scale_data.size();
      for (int i = 0; i < in_scale_data_size; i++) {
        std::vector<float> tmp;
        for (int j = 0; j < 6; j++) {
          tmp.push_back(1.0f / in_scale_data[i]);
        }
        phi::DenseTensor dst_tensor;
        dst_tensor.set_type(phi::DataType::FLOAT32);
        dst_tensor.Resize({(int64_t)tmp.size()});
        memcpy(cpu_ctx->Alloc<float>(&dst_tensor),
               tmp.data(),
               tmp.size() * sizeof(float));

        size_t dst_hash = HashTensor<float>(dst_tensor);
        std::string pre_name = GetPrefixWithoutHash(src_name);
        std::string dst_name = pre_name + "_#" + std::to_string(dst_hash);
        auto* dst_node = FindNodeWithName(graph, dst_name);
        if (dst_node == nullptr) {
          Assign(dst_tensor,
                 scope->Var(dst_name)->GetMutable<phi::DenseTensor>());
          // Create dst node
          // Update dst var_desc in block
          VarDesc dst_desc(dst_name);
          dst_desc.SetPersistable(true);
          dst_desc.SetShape(vectorize(dst_tensor.dims()));
          dst_desc.SetDataType(
              framework::TransToProtoVarType(dst_tensor.dtype()));
          Node* dst = graph->CreateVarNode(&dst_desc);
          auto* block_dst_desc = block->Var(dst_name);
          block_dst_desc->SetPersistable(dst_desc.Persistable());
          block_dst_desc->SetShape(dst_desc.GetShape());
          block_dst_desc->SetDataType(dst_desc.GetDataType());
          input_max_node->push_back(dst);
          input_max_name->push_back(dst_name);
        }
      }
    };

    auto outscale2maxw = [&](const std::string& input_name,
                             const std::string& src_name,
                             std::vector<Node*>* weight_max_node,
                             std::vector<std::string>* weight_max_name,
                             std::vector<Node*>* old_weight_max_node,
                             std::vector<std::string>* old_weight_max_name) {
      auto GetPrefixWithoutHash = [](const std::string& name) -> std::string {
        std::size_t found = name.find("_#");
        return found == std::string::npos ? name : name.substr(0, found);
      };
      std::vector<float> max_bound_pow{127 * 127};  // int8_t
      phi::DenseTensor max_bound_tensor;
      max_bound_tensor.set_type(phi::DataType::FLOAT32);
      max_bound_tensor.Resize({(int64_t)max_bound_pow.size()});
      memcpy(cpu_ctx->Alloc<float>(&max_bound_tensor),
             max_bound_pow.data(),
             max_bound_pow.size() * sizeof(float));
      std::vector<float> in_scale_data = PADDLE_GET_CONST(
          std::vector<float>, fused_mt_int8->Op()->GetAttr(src_name, false));
      auto names = fused_mt_int8->Op()->Input(input_name);
      int id = 0;
      for (auto name : names) {
        phi::DenseTensor in_scale_tensor;
        in_scale_tensor.set_type(phi::DataType::FLOAT32);
        in_scale_tensor.Resize({1});
        memcpy(cpu_ctx->Alloc<float>(&in_scale_tensor),
               &(in_scale_data[id]),
               1 * sizeof(float));
        size_t dst_hash = HashTensor<float>(in_scale_tensor);
        std::string pre_name = GetPrefixWithoutHash(name);
        std::string dst_name = pre_name + "_#" + std::to_string(dst_hash);
        auto* dst_node = FindNodeWithName(graph, dst_name);
        if (dst_node == nullptr) {
          phi::DenseTensor* curr_tensor =
              scope->Var(name)->GetMutable<phi::DenseTensor>();
          PADDLE_ENFORCE_NE(
              curr_tensor,
              nullptr,
              common::errors::Fatal("tensor node should not be nullptr"));
          // Create dst node
          // Update dst var_desc in block
          VarDesc dst_desc(dst_name);
          dst_desc.SetPersistable(true);
          dst_desc.SetShape(vectorize(curr_tensor->dims()));
          dst_desc.SetDataType(
              framework::TransToProtoVarType(curr_tensor->dtype()));
          Node* dst = graph->CreateVarNode(&dst_desc);
          auto* block_dst_desc = block->Var(dst_name);
          block_dst_desc->SetPersistable(dst_desc.Persistable());
          block_dst_desc->SetShape(dst_desc.GetShape());
          block_dst_desc->SetDataType(dst_desc.GetDataType());
          weight_max_node->push_back(dst);
          weight_max_name->push_back(dst_name);
          auto* src_node = FindNodeWithName(graph, name);
          old_weight_max_node->push_back(src_node);
          old_weight_max_name->push_back(name);
          auto* dst_var = scope->FindVar(dst_name);
          if (dst_var == nullptr) {
            phi::DenseTensor tmp_tensor;
            tmp_tensor.set_type(phi::DataType::FLOAT32);
            tmp_tensor.Resize(curr_tensor->dims());
            memcpy(cpu_ctx->Alloc<float>(&tmp_tensor),
                   curr_tensor,
                   curr_tensor->numel() * sizeof(float));
            phi::MultiplyKernel<float>(
                *cpu_ctx, *curr_tensor, max_bound_tensor, &tmp_tensor);
            phi::MultiplyKernel<float>(
                *cpu_ctx, tmp_tensor, in_scale_tensor, &tmp_tensor);
            Assign(tmp_tensor,
                   scope->Var(dst_name)->GetMutable<phi::DenseTensor>());
          }
        }
        id++;
      }
    };
    // generate input node
    attr2weight(
        "qkv_in_scale", &(input_max_nodes_vec[0]), &(input_max_names_vec[0]));
    attr2weight("out_linear_in_scale",
                &(input_max_nodes_vec[1]),
                &(input_max_names_vec[1]));
    attr2weight(
        "ffn1_in_scale", &(input_max_nodes_vec[2]), &(input_max_names_vec[2]));
    attr2weight(
        "ffn2_in_scale", &(input_max_nodes_vec[3]), &(input_max_names_vec[3]));

    // cast some nodes to fp32 nodes
    std::vector<Node*> fp32_nodes;
    auto cast_tofp32_func = [&](const std::string& input_name) {
      auto names = fused_mt_int8->Op()->Input(input_name);
      for (auto name : names) {
        auto* curr_tensor = scope->Var(name)->GetMutable<phi::DenseTensor>();
        PADDLE_ENFORCE_NE(
            curr_tensor,
            nullptr,
            common::errors::Fatal("tensor node should not be nullptr"));
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
    cast_tofp32_func("QKVOutScale");
    cast_tofp32_func("OutLinearOutScale");
    cast_tofp32_func("FFN1OutScale");
    cast_tofp32_func("FFN2OutScale");

    outscale2maxw("QKVOutScale",
                  "qkv_in_scale",
                  &(weight_max_nodes_vec[0]),
                  &(weight_max_names_vec[0]),
                  &(old_weight_max_nodes_vec[0]),
                  &(old_weight_max_names_vec[0]));
    outscale2maxw("OutLinearOutScale",
                  "out_linear_in_scale",
                  &(weight_max_nodes_vec[1]),
                  &(weight_max_names_vec[1]),
                  &(old_weight_max_nodes_vec[1]),
                  &(old_weight_max_names_vec[1]));
    outscale2maxw("FFN1OutScale",
                  "ffn1_in_scale",
                  &(weight_max_nodes_vec[2]),
                  &(weight_max_names_vec[2]),
                  &(old_weight_max_nodes_vec[2]),
                  &(old_weight_max_names_vec[2]));
    outscale2maxw("FFN2OutScale",
                  "ffn2_in_scale",
                  &(weight_max_nodes_vec[3]),
                  &(weight_max_names_vec[3]),
                  &(old_weight_max_nodes_vec[3]),
                  &(old_weight_max_names_vec[3]));

    // Generate max_buffer: per_tensor_max and per_batch_max for kv_cache
    int layer_num = fused_mt_int8->Op()->Input("QKVW").size();
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    phi::DenseTensor max_buffer_tensor;
    max_buffer_tensor.set_type(phi::DataType::FLOAT32);
    int max_buffer_len = max_ptr_size * layer_num * 2;
    max_buffer_tensor.Resize({max_buffer_len});
    std::vector<float> ones_vec(max_buffer_len, 1.f);
    memcpy(cpu_ctx->Alloc<float>(&max_buffer_tensor),
           ones_vec.data(),
           max_buffer_len * sizeof(float));
    size_t max_buffer_hash = HashTensor<float>(max_buffer_tensor);
    std::string max_buffer_name =
        "max_buffer_#" + std::to_string(max_buffer_hash);
    auto* max_buffer_node = FindNodeWithName(graph, max_buffer_name);
    if (max_buffer_node == nullptr) {
      // Create dst node
      // Update dst var_desc in block
      VarDesc dst_desc(max_buffer_name);
      dst_desc.SetPersistable(true);
      dst_desc.SetShape(vectorize(max_buffer_tensor.dims()));
      dst_desc.SetDataType(
          framework::TransToProtoVarType(max_buffer_tensor.dtype()));
      max_buffer_node = graph->CreateVarNode(&dst_desc);
      auto* block_dst_desc = block->Var(max_buffer_name);
      block_dst_desc->SetPersistable(dst_desc.Persistable());
      block_dst_desc->SetShape(dst_desc.GetShape());
      block_dst_desc->SetDataType(dst_desc.GetDataType());
      auto* max_buffer_var = scope->FindVar(max_buffer_name);
      if (max_buffer_var == nullptr) {
        Assign(max_buffer_tensor,
               scope->Var(max_buffer_name)->GetMutable<phi::DenseTensor>());
      }
    }

    // Generate fused_multi_transformer_int8_xpu op inplace
    fused_mt_int8->RenameOp("fused_multi_transformer_int8_xpu");
    framework::OpDesc* fused_mt_int8_xpu_op_desc = fused_mt_int8->Op();
    fused_mt_int8_xpu_op_desc->SetType("fused_multi_transformer_int8_xpu");
    std::unordered_map<std::string, std::vector<std::string>> name_caches;
    for (auto key : fused_mt_int8_xpu_op_desc->InputNames()) {
      name_caches.insert({key, fused_mt_int8_xpu_op_desc->Input(key)});
    }
    for (auto key : fused_mt_int8_xpu_op_desc->OutputNames()) {
      name_caches.insert({key, fused_mt_int8_xpu_op_desc->Output(key)});
    }
    fused_mt_int8_xpu_op_desc->MutableInputs()->clear();
    fused_mt_int8_xpu_op_desc->MutableOutputs()->clear();
    fused_mt_int8_xpu_op_desc->SetInput("x", name_caches.at("X"));
    fused_mt_int8_xpu_op_desc->SetInput("ln_scale", name_caches.at("LnScale"));
    fused_mt_int8_xpu_op_desc->SetInput("ln_bias", name_caches.at("LnBias"));
    fused_mt_int8_xpu_op_desc->SetInput("qkv_bias", name_caches.at("QKVBias"));
    if (name_caches.count("CacheKV") > 0) {
      fused_mt_int8_xpu_op_desc->SetInput("cache_kv",
                                          name_caches.at("CacheKV"));
    }
    if (name_caches.count("gather_index") > 0) {
      fused_mt_int8_xpu_op_desc->SetInput("gather_index",
                                          name_caches.at("gather_index"));
    }
    if (!fused_mt_int8_xpu_op_desc->HasAttr("gather_axis")) {
      fused_mt_int8_xpu_op_desc->SetAttr("gather_axis", 0);
    }
    if (pre_caches) {
      fused_mt_int8_xpu_op_desc->SetInput("pre_caches",
                                          name_caches.at("PreCaches"));
    }
    if (rotary_pos_emb) {
      fused_mt_int8_xpu_op_desc->SetInput("rotary_pos_emb",
                                          name_caches.at("RotaryPosEmb"));
    }
    if (time_step) {
      fused_mt_int8_xpu_op_desc->SetInput("time_step",
                                          name_caches.at("TimeStep"));
    }
    if (seq_lengths) {
      fused_mt_int8_xpu_op_desc->SetInput("seq_lengths",
                                          name_caches.at("SeqLengths"));
    }
    if (src_mask) {
      fused_mt_int8_xpu_op_desc->SetInput("src_mask",
                                          name_caches.at("SrcMask"));
    }
    fused_mt_int8_xpu_op_desc->SetInput("out_linear_bias",
                                        name_caches.at("OutLinearBias"));
    fused_mt_int8_xpu_op_desc->SetInput("ffn_ln_scale",
                                        name_caches.at("FFNLnScale"));
    fused_mt_int8_xpu_op_desc->SetInput("ffn_ln_bias",
                                        name_caches.at("FFNLnBias"));
    fused_mt_int8_xpu_op_desc->SetInput("ffn1_bias",
                                        name_caches.at("FFN1Bias"));
    fused_mt_int8_xpu_op_desc->SetInput("ffn2_bias",
                                        name_caches.at("FFN2Bias"));
    fused_mt_int8_xpu_op_desc->SetOutput("cache_kv_out",
                                         name_caches.at("CacheKVOut"));
    fused_mt_int8_xpu_op_desc->SetOutput("out", name_caches.at("Out"));
    fused_mt_int8_xpu_op_desc->SetInput("qkvw", name_caches.at("QKVW"));
    fused_mt_int8_xpu_op_desc->SetInput("qkv_scales", weight_max_names_vec[0]);
    fused_mt_int8_xpu_op_desc->SetInput("out_linear_w",
                                        name_caches.at("OutLinearW"));
    fused_mt_int8_xpu_op_desc->SetInput("out_linear_scales",
                                        weight_max_names_vec[1]);
    fused_mt_int8_xpu_op_desc->SetInput("ffn1_weight",
                                        name_caches.at("FFN1Weight"));
    fused_mt_int8_xpu_op_desc->SetInput("ffn1_scales", weight_max_names_vec[2]);
    fused_mt_int8_xpu_op_desc->SetInput("ffn2_weight",
                                        name_caches.at("FFN2Weight"));
    fused_mt_int8_xpu_op_desc->SetInput("ffn2_scales", weight_max_names_vec[3]);

    fused_mt_int8_xpu_op_desc->SetInput("qkv_in_max", input_max_names_vec[0]);
    fused_mt_int8_xpu_op_desc->SetInput("out_linear_in_max",
                                        input_max_names_vec[1]);
    fused_mt_int8_xpu_op_desc->SetInput("ffn1_in_max", input_max_names_vec[2]);
    fused_mt_int8_xpu_op_desc->SetInput("ffn2_in_max", input_max_names_vec[3]);
    fused_mt_int8_xpu_op_desc->SetInput("max_buffer", {max_buffer_name});

    if (!fused_mt_int8_xpu_op_desc->HasAttr("rotary_emb_dims")) {
      fused_mt_int8_xpu_op_desc->SetAttr("rotary_emb_dims", 0);
    }

    for (auto nodes : old_weight_max_nodes_vec) {
      for (auto node : nodes) {
        IR_NODE_UNLINK(node, fused_mt_int8);
      }
    }

    for (auto nodes : weight_max_nodes_vec) {
      for (auto node : nodes) {
        IR_NODE_LINK_TO(node, fused_mt_int8);
      }
    }
    // link QKVWMax/OutLinearWMax/FFN1WeightMax/FFN2WeightMax to
    // fused_mt_int8_xpu
    for (auto nodes : input_max_nodes_vec) {
      for (auto node : nodes) {
        IR_NODE_LINK_TO(node, fused_mt_int8);
      }
    }
    IR_NODE_LINK_TO(max_buffer_node, fused_mt_int8);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_multi_transformer_int8_xpu_quant_pass,
              paddle::framework::ir::FusedMultiTransformerInt8XPUQuantPass);

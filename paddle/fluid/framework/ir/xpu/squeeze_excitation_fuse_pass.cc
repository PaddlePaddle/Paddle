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

#include "paddle/fluid/framework/ir/xpu/squeeze_excitation_fuse_pass.h"
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace math {

template <typename T>
static inline void Transpose(const T* in, T* out, int h, int w) {
  for (int h1 = 0; h1 < w; ++h1) {
    for (int w1 = 0; w1 < h; ++w1) {
      out[h1 * h + w1] = in[w1 * w + h1];
    }
  }
}

}  // namespace math

namespace patterns {

struct SqueezeExcitationFusePattern : public PatternBase {
  SqueezeExcitationFusePattern(PDPattern* pattern,
                               const std::string& name_scope,
                               const std::string& op_type,
                               const std::string& act_type,
                               bool with_branch,
                               bool with_bias);

  // declare operator node`s name
  PATTERN_DECL_NODE(pool2d);
  PATTERN_DECL_NODE(mul_1);
  PATTERN_DECL_NODE(mul_2);
  PATTERN_DECL_NODE(ew_mul);
  PATTERN_DECL_NODE(ew_branch_add);
  PATTERN_DECL_NODE(block_act);

  // declare variable node`s name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(pool2d_out);
  PATTERN_DECL_NODE(mul_1_w);
  PATTERN_DECL_NODE(mul_1_w_max);
  PATTERN_DECL_NODE(mul_1_bias);
  PATTERN_DECL_NODE(mul_1_out);
  PATTERN_DECL_NODE(mul_1_out_max);
  PATTERN_DECL_NODE(mul_2_w);
  PATTERN_DECL_NODE(mul_2_w_max);
  PATTERN_DECL_NODE(mul_2_bias);
  PATTERN_DECL_NODE(mul_2_out);
  PATTERN_DECL_NODE(mul_2_out_max);
  PATTERN_DECL_NODE(ew_mul_out);
  PATTERN_DECL_NODE(ew_branch_add_in);
  PATTERN_DECL_NODE(ew_branch_add_out);
  PATTERN_DECL_NODE(block_act_out);
};

SqueezeExcitationFusePattern::SqueezeExcitationFusePattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& op_type,
    const std::string& act_type,
    bool with_branch,
    bool with_bias)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input("pool2d", "X")
                ->assert_is_op_input("elementwise_mul", "X")
                ->AsInput();

  auto pool2d_teller = [](const Node* x) {
    auto* op_desc = x->Op();
    bool has_adap = op_desc->HasAttr("adaptive");
    if (has_adap) {
      auto ksize =
          PADDLE_GET_CONST(std::vector<int>, op_desc->GetAttr("ksize"));
      if (ksize[0] != 1 || ksize[1] != 1) {
        return false;
      }
    } else if (PADDLE_GET_CONST(bool, op_desc->GetAttr("global_pooling")) ==
               false) {
      return false;
    }
    return true;
  };

  auto* pool2d = pattern->NewNode(pool2d_repr())
                     ->assert_is_op("pool2d")
                     ->assert_op_attr<std::string>("pooling_type", "avg")
                     ->assert_more(pool2d_teller);

  auto* pool2d_out = pattern->NewNode(pool2d_out_repr())
                         ->assert_is_op_output("pool2d", "Out")
                         ->assert_is_op_input(op_type, "x");

  auto mul_w_teller = [](const Node* x) {
    auto* var_desc = x->Var();
    auto filter_dims = var_desc->GetShape();
    auto filter_dtype = var_desc->GetDataType();
    if (filter_dtype == proto::VarType::Type::VarType_Type_INT8) {
      return false;
    }
    auto in_c = filter_dims[0];
    auto out_c = filter_dims[1];
    auto bigger = std::max(in_c, out_c);
    auto smaller = std::min(in_c, out_c);
    if (bigger % smaller != 0) {
      return false;
    }
    return true;
  };

  auto* mul_1 = pattern->NewNode(mul_1_repr())->assert_is_op(op_type);
  auto* mul_1_w = pattern->NewNode(mul_1_w_repr())
                      ->assert_is_op_input(op_type, "filter")
                      ->assert_more(mul_w_teller);
  auto* mul_1_w_max = pattern->NewNode(mul_1_w_max_repr())
                          ->assert_is_op_input(op_type, "filter_max");
  auto* mul_1_out = pattern->NewNode(mul_1_out_repr())
                        ->assert_is_op_output(op_type, "out")
                        ->assert_is_op_input(op_type, "x");
  auto* mul_1_out_max = pattern->NewNode(mul_1_out_max_repr())
                            ->assert_is_op_output(op_type, "out_max");
  auto* mul_2 = pattern->NewNode(mul_2_repr())->assert_is_op(op_type);
  auto* mul_2_w = pattern->NewNode(mul_2_w_repr())
                      ->assert_is_op_input(op_type, "filter")
                      ->assert_more(mul_w_teller);
  auto* mul_2_w_max = pattern->NewNode(mul_2_w_max_repr())
                          ->assert_is_op_input(op_type, "filter_max");
  auto* mul_2_out = pattern->NewNode(mul_2_out_repr())
                        ->assert_is_op_output(op_type, "out")
                        ->assert_is_op_input("elementwise_mul", "Y");
  auto* mul_2_out_max = pattern->NewNode(mul_2_out_max_repr())
                            ->assert_is_op_output(op_type, "out_max");

  PDNode* mul_1_bias = nullptr;
  PDNode* mul_2_bias = nullptr;
  if (with_bias) {
    mul_1_bias = pattern->NewNode(mul_1_bias_repr())
                     ->assert_is_op_input(op_type, "bias");
    mul_2_bias = pattern->NewNode(mul_2_bias_repr())
                     ->assert_is_op_input(op_type, "bias");
  }
  auto* ew_mul =
      pattern->NewNode(ew_mul_repr())->assert_is_op("elementwise_mul");
  auto* ew_mul_out = pattern->NewNode(ew_mul_out_repr())
                         ->assert_is_op_output("elementwise_mul", "Out");

  // branch
  PDNode* ew_branch_add_in = nullptr;
  PDNode* ew_branch_add = nullptr;
  PDNode* ew_branch_add_out = nullptr;
  if (with_branch) {
    ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())
                           ->assert_is_op_input("elementwise_add", "X")
                           ->AsInput();
    ew_branch_add =
        pattern->NewNode(ew_branch_add_repr())->assert_is_op("elementwise_add");
    ew_branch_add_out = pattern->NewNode(ew_branch_add_out_repr())
                            ->assert_is_op_output("elementwise_add", "out");
  }
  // act
  PDNode* block_act = nullptr;
  PDNode* block_act_out = nullptr;
  if (act_type != "linear") {
    block_act = pattern->NewNode(block_act_repr())->assert_is_op(act_type);
    block_act_out = pattern->NewNode(block_act_out_repr())
                        ->assert_is_op_output(act_type, "Out");
  }

  // pass
  pool2d->LinksFrom({x}).LinksTo({pool2d_out});
  mul_1->LinksFrom({mul_1_w, mul_1_w_max, pool2d_out})
      .LinksTo({mul_1_out, mul_1_out_max});
  mul_2->LinksFrom({mul_2_w, mul_2_w_max, mul_1_out})
      .LinksTo({mul_2_out, mul_2_out_max});
  ew_mul->LinksFrom({x, mul_2_out}).LinksTo({ew_mul_out});

  if (with_branch) {
    ew_mul_out->assert_is_op_input("elementwise_add", "Y");
    ew_branch_add->LinksFrom({ew_mul_out, ew_branch_add_in})
        .LinksTo({ew_branch_add_out});
  } else {
    ew_branch_add_out = ew_mul_out;
  }
  if (act_type != "linear") {
    ew_branch_add_out->assert_is_op_input(act_type, "X");
    block_act->LinksFrom({ew_branch_add_out}).LinksTo({block_act_out});
  } else {
    block_act_out = ew_branch_add_out;
  }
  if (with_bias) {
    mul_1->LinksFrom({mul_1_bias});
    mul_2->LinksFrom({mul_2_bias});
  }
}

}  // namespace patterns

void SqueezeExcitationFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null. "));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto with_branch : {true, false}) {
    for (auto with_bias : {true, false}) {
      for (auto op_type : {"conv2d_xpu"}) {
        for (auto act_type : {"relu",
                              "sigmoid",
                              "tanh",
                              "leaky_relu",
                              "hard_swish",
                              "hard_sigmoid",
                              "relu6",
                              "linear"}) {
          found_subgraph_count +=
              ApplyImpl(graph, op_type, act_type, with_branch, with_bias);
        }
      }
    }
  }
  AddStatis(found_subgraph_count);
}

int SqueezeExcitationFusePass::ApplyImpl(ir::Graph* graph,
                                         const std::string& op_type,
                                         const std::string& act_type,
                                         bool with_branch,
                                         bool with_bias) const {
  GraphPatternDetector gpd;
  patterns::SqueezeExcitationFusePattern pattern(gpd.mutable_pattern(),
                                                 name_scope_,
                                                 op_type,
                                                 act_type,
                                                 with_branch,
                                                 with_bias);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle SqueezeExcitationFusePass";
    /* declare operator node's name */
    GET_IR_NODE(pool2d);
    GET_IR_NODE(mul_1);
    GET_IR_NODE(mul_2);
    GET_IR_NODE(ew_mul);
    GET_IR_NODE(ew_branch_add);
    GET_IR_NODE(block_act)
    /* declare variable node's name*/
    GET_IR_NODE(x);
    GET_IR_NODE(mul_1_w);
    GET_IR_NODE(mul_1_w_max);
    GET_IR_NODE(mul_1_bias);
    GET_IR_NODE(mul_1_out);
    GET_IR_NODE(mul_2_w);
    GET_IR_NODE(mul_2_w_max);
    GET_IR_NODE(mul_2_bias);
    GET_IR_NODE(mul_2_out);
    GET_IR_NODE(ew_mul_out);
    GET_IR_NODE(ew_branch_add_in);
    GET_IR_NODE(ew_branch_add_out);
    GET_IR_NODE(block_act_out);

    auto* block = pool2d->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("squeeze_excitation_block");
    fused_op_desc.SetInput("x", {x->Name()});
    if (with_branch) {
      fused_op_desc.SetInput("branch", {ew_branch_add_in->Name()});
    }
    // filter
    auto mul_1_w_name = mul_1_w->Name();
    auto* mul_1_w_t =
        scope->FindVar(mul_1_w_name)->GetMutable<phi::DenseTensor>();
    auto mul_1_w_dims = mul_1_w_t->dims();
    auto mul_1_w_len = mul_1_w_t->numel();
    int16_t* mul_1_w_ptr = mul_1_w_t->data<int16_t>();
    auto* mul_2_w_t =
        scope->FindVar(mul_2_w->Name())->GetMutable<phi::DenseTensor>();
    auto mul_2_w_dims = mul_2_w_t->dims();
    auto mul_2_w_len = mul_2_w_t->numel();
    int16_t* mul_2_w_ptr = mul_2_w_t->data<int16_t>();
    if (mul_1_w_dims[0] != mul_2_w_dims[1] ||
        mul_1_w_dims[1] != mul_2_w_dims[0] ||
        mul_1_w_len != mul_1_w_dims[0] * mul_1_w_dims[1]) {
      std::stringstream ss;
      ss << "Error: Dims of excitation mul1 weight is: " << mul_1_w_dims
         << ", but get dims of excitation mul2 weight is: " << mul_2_w_dims;
      PADDLE_THROW(common::errors::InvalidArgument(ss.str()));
    }
    std::vector<int16_t> encode_filter_int16;
    encode_filter_int16.resize(mul_1_w_len + mul_2_w_len);

    PADDLE_ENFORCE_EQ(mul_1_w_dims[1] % mul_1_w_dims[0] == 0,
                      1,
                      common::errors::InvalidArgument(
                          "Reduction ratio of excitation is not an integer."
                          "Received mul_1_w_dims[1]: %d, mul_1_w_dims[0]: %d",
                          mul_1_w_dims[1],
                          mul_1_w_dims[0]));
    fused_op_desc.SetAttr(
        "filter_dims",
        std::vector<int>{static_cast<int>(mul_1_w_dims[1] / mul_1_w_dims[0]),
                         static_cast<int>(mul_1_w_dims[1])});

    paddle::framework::ir::math::Transpose(mul_1_w_ptr,
                                           encode_filter_int16.data(),
                                           mul_1_w_dims[0],
                                           mul_1_w_dims[1]),
        paddle::framework::ir::math::Transpose(
            mul_2_w_ptr,
            encode_filter_int16.data() + mul_1_w_len,
            mul_2_w_dims[0],
            mul_2_w_dims[1]);

    std::string new_filter_name = "se_" + mul_1_w_name;
    Node* new_filter_node = nullptr;
    VarDesc dst_desc(new_filter_name);
    dst_desc.SetPersistable(true);
    dst_desc.SetShape({mul_1_w_len + mul_2_w_len});
    dst_desc.SetDataType(framework::TransToProtoVarType(mul_1_w_t->dtype()));
    new_filter_node = graph->CreateVarNode(&dst_desc);
    auto* block_dst_desc = block->Var(new_filter_name);
    block_dst_desc->SetPersistable(dst_desc.Persistable());
    block_dst_desc->SetShape(dst_desc.GetShape());
    block_dst_desc->SetDataType(dst_desc.GetDataType());

    phi::DenseTensor new_filter_t;
    new_filter_t.Resize(DDim({mul_1_w_len + mul_2_w_len}));
    new_filter_t.set_type(phi::DataType::INT16);
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    auto* new_filter_data = cpu_ctx->Alloc<int16_t>(&new_filter_t);

    memcpy(new_filter_data,
           encode_filter_int16.data(),
           (mul_1_w_len + mul_2_w_len) * sizeof(int16_t));

    Assign(new_filter_t,
           scope->Var(new_filter_name)->GetMutable<phi::DenseTensor>());
    fused_op_desc.SetInput("filter", {new_filter_name});

    // filter max
    std::vector<float> encode_filter_max;
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    int filter_max_size = max_ptr_size + max_ptr_size;
    encode_filter_max.resize(filter_max_size);

    auto mul_1_w_max_name = mul_1_w_max->Name();
    auto mul_2_w_max_name = mul_2_w_max->Name();
    auto* mul_1_w_max_t =
        scope->FindVar(mul_1_w_max_name)->GetMutable<phi::DenseTensor>();
    auto* mul_2_w_max_t =
        scope->FindVar(mul_2_w_max_name)->GetMutable<phi::DenseTensor>();

    float* mul_1_w_max_ptr = mul_1_w_max_t->data<float>();
    float* mul_2_w_max_ptr = mul_2_w_max_t->data<float>();
    memcpy(encode_filter_max.data(),
           mul_1_w_max_ptr,
           max_ptr_size * sizeof(float));
    memcpy(encode_filter_max.data() + max_ptr_size,
           mul_2_w_max_ptr,
           max_ptr_size * sizeof(float));

    std::string new_filter_max_name = new_filter_name + "_max";
    Node* new_filter_max_node = nullptr;
    VarDesc filter_max_desc(new_filter_max_name);
    filter_max_desc.SetPersistable(true);
    filter_max_desc.SetShape({filter_max_size});
    filter_max_desc.SetDataType(
        framework::TransToProtoVarType(mul_1_w_max_t->dtype()));
    new_filter_max_node = graph->CreateVarNode(&filter_max_desc);
    auto* block_filter_max_desc = block->Var(new_filter_max_name);
    block_filter_max_desc->SetPersistable(filter_max_desc.Persistable());
    block_filter_max_desc->SetShape(filter_max_desc.GetShape());
    block_filter_max_desc->SetDataType(filter_max_desc.GetDataType());

    phi::DenseTensor new_filter_max_t;
    new_filter_max_t.Resize(DDim({filter_max_size}));
    new_filter_max_t.set_type(phi::DataType::FLOAT32);
    auto* new_filter_max_data = cpu_ctx->Alloc<float>(&new_filter_max_t);

    memcpy(new_filter_max_data,
           encode_filter_max.data(),
           (filter_max_size) * sizeof(float));

    Assign(new_filter_max_t,
           scope->Var(new_filter_max_name)->GetMutable<phi::DenseTensor>());

    fused_op_desc.SetInput("filter_max", {new_filter_max_name});

    // bias
    std::string new_bias_name = new_filter_name + "_bias";
    VarDesc new_bias_desc(new_bias_name);
    new_bias_desc.SetPersistable(true);
    new_bias_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    Node* new_bias_node = graph->CreateVarNode(&new_bias_desc);
    if (with_bias) {
      auto mul_1_bias_name = mul_1_bias->Name();
      auto mul_2_bias_name = mul_2_bias->Name();
      auto* mul_1_bias_t =
          scope->FindVar(mul_1_bias_name)->GetMutable<phi::DenseTensor>();
      auto* mul_2_bias_t =
          scope->FindVar(mul_2_bias_name)->GetMutable<phi::DenseTensor>();
      int mul_1_bias_numel = mul_1_bias_t->numel();
      int mul_2_bias_numel = mul_2_bias_t->numel();

      std::vector<float> encode_bias;
      encode_bias.resize(mul_1_bias_numel + mul_2_bias_numel);
      float* mul_1_bias_ptr = mul_1_bias_t->data<float>();
      float* mul_2_bias_ptr = mul_2_bias_t->data<float>();

      memcpy(
          encode_bias.data(), mul_1_bias_ptr, mul_1_bias_numel * sizeof(float));
      memcpy(encode_bias.data() + mul_1_bias_numel,
             mul_2_bias_ptr,
             mul_2_bias_numel * sizeof(float));

      new_bias_desc.SetShape({mul_1_bias_numel + mul_2_bias_numel});
      auto* block_new_bias_dst_desc = block->Var(new_bias_name);
      block_new_bias_dst_desc->SetPersistable(new_bias_desc.Persistable());
      block_new_bias_dst_desc->SetShape(new_bias_desc.GetShape());
      block_new_bias_dst_desc->SetDataType(new_bias_desc.GetDataType());

      phi::DenseTensor new_bias_t;
      new_bias_t.Resize(DDim({mul_1_bias_numel + mul_2_bias_numel}));
      new_bias_t.set_type(phi::DataType::FLOAT32);
      auto* cpu_ctx = static_cast<phi::CPUContext*>(
          phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
      auto* new_bias_data = cpu_ctx->Alloc<float>(&new_bias_t);

      memcpy(new_bias_data,
             encode_bias.data(),
             (mul_1_bias_numel + mul_2_bias_numel) * sizeof(float));
      Assign(new_bias_t,
             scope->Var(new_bias_name)->GetMutable<phi::DenseTensor>());
      fused_op_desc.SetInput("bias", {new_bias_name});
    }
    fused_op_desc.SetAttr("has_bias", with_bias);
    fused_op_desc.SetAttr("has_branch", with_branch);
    std::string output_name;
    if (act_type != "linear") {
      output_name = block_act_out->Name();
    } else if (with_branch) {
      output_name = ew_branch_add_out->Name();
    } else {
      output_name = ew_mul_out->Name();
    }
    fused_op_desc.SetOutput("out", {output_name});
    fused_op_desc.SetAttr("op_type", std::vector<int>{4});
    fused_op_desc.SetAttr("place_x", std::vector<int>{0});
    fused_op_desc.SetAttr("place_y", std::vector<int>{9});
    fused_op_desc.SetAttr("place_z", std::vector<int>{10});
    fused_op_desc.SetAttr("strides", std::vector<int>{});
    fused_op_desc.SetAttr("paddings", std::vector<int>{});
    fused_op_desc.SetAttr("dilations", std::vector<int>{});
    fused_op_desc.SetAttr("groups", std::vector<int>{});
    fused_op_desc.SetAttr("block_lod", std::vector<int>{1});
    fused_op_desc.SetAttr("conv_bias", std::vector<int>{with_bias});

    std::map<std::string, int> act_map{{"linear", 0},
                                       {"relu", 1},
                                       {"sigmoid", 2},
                                       {"tanh", 3},
                                       {"leaky_relu", 5},
                                       {"hard_swish", 14},
                                       {"hard_sigmoid", 15},
                                       {"relu6", 17}};

    float block_act_param_ = 0.f;
    if (act_type == "leak_relu") {
      block_act_param_ =
          PADDLE_GET_CONST(float, block_act->Op()->GetAttr("alpha"));
    } else if (act_type == "hard_sigmoid") {
      block_act_param_ =
          PADDLE_GET_CONST(float, block_act->Op()->GetAttr("slope"));
    }
    fused_op_desc.SetAttr(
        "act_type",
        std::vector<int>{
            PADDLE_GET_CONST(int, mul_1->Op()->GetAttr("act_type")),
            PADDLE_GET_CONST(int, mul_2->Op()->GetAttr("act_type")),
            act_map[act_type]});

    fused_op_desc.SetAttr(
        "act_param",
        std::vector<float>{
            PADDLE_GET_CONST(float, mul_1->Op()->GetAttr("act_param")),
            PADDLE_GET_CONST(float, mul_2->Op()->GetAttr("act_param")),
            block_act_param_});

    auto* new_op_node = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(x, new_op_node);
    if (with_branch) {
      IR_NODE_LINK_TO(ew_branch_add_in, new_op_node);
    }
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_filter_max_node, new_op_node);

    if (with_bias) {
      IR_NODE_LINK_TO(new_bias_node, new_op_node);
    }

    if (act_type != "linear") {
      IR_NODE_LINK_TO(new_op_node, block_act_out);
    } else if (with_branch) {
      IR_NODE_LINK_TO(new_op_node, ew_branch_add_out);
    } else {
      IR_NODE_LINK_TO(new_op_node, ew_mul_out);
    }
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {
        pool2d, mul_1, mul_1_out, mul_2, mul_2_out, ew_mul};
    if (with_bias) {
      delete_nodes.insert(mul_1_bias);
      delete_nodes.insert(mul_2_bias);
    }
    if (with_branch) {
      delete_nodes.insert(ew_branch_add);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(squeeze_excitation_fuse_pass,
              paddle::framework::ir::SqueezeExcitationFusePass);

REGISTER_PASS_CAPABILITY(squeeze_excitation_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "squeeze_excitation_block", 0));

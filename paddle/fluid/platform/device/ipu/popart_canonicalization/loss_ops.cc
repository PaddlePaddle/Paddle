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

#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

bool is_dynamic_graph() {
  auto *ipu_backend = platform::ipu::IpuBackend::GetInstance();
  return ipu_backend->GetIpuStrategy()->is_dynamic;
}

Node *identity_loss_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto reduction = BOOST_GET_CONST(int, op->GetAttr("reduction"));
  return CreateIdentityLossOp(
      graph, node, node->inputs, node->outputs, reduction);
}

Node *cross_entropy_general_handler(Graph *graph,
                                    Node *node,
                                    Node *logits,
                                    Node *label,
                                    Node *output,
                                    bool soft_label,
                                    int ignore_index,
                                    int reduction,
                                    int axis) {
  Node *cast_and_reshape = nullptr;
  Node *final_loss_node = nullptr;
  if (soft_label) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "soft_label is not supported yet in IPU"));
  }
  bool append_identity_loss = is_dynamic_graph();
  bool is_last_var_node = IsLastVarNode(output);
  append_identity_loss = append_identity_loss && is_last_var_node;

  if (label->Var()->GetDataType() == framework::proto::VarType::INT32) {
    cast_and_reshape = label;
  } else {
    cast_and_reshape =
        CreateCast(graph, node, {label}, {}, framework::proto::VarType::INT32)
            ->outputs.front();
  }

  auto label_shape_ = label->Var()->GetShape();
  auto logits_shape_ = logits->Var()->GetShape();

  axis = axis < 0 ? logits_shape_.size() + axis : axis;

  auto label_transposed(label_shape_);

  if (axis != (logits_shape_.size() - 1)) {
    // the softmax axis(a) is not at the last dimension.
    // logit shape: [N1, ..., C, ..., Nk]
    // label shape: [N1, ..., 1, ..., Nk]
    //                   _____^_____
    // dim:           0, ..., a, ..., k-1
    // needs to transpose the softmax axis in logit to last dimension
    // with following transpose perm: [0, ..., a-1, a+1, ..., k-1, a]
    std::vector<int64_t> trans(logits_shape_.size(), 0);
    std::iota(trans.begin(), trans.begin() + axis, 0);
    std::iota(trans.begin() + axis, trans.end() - 1, axis + 1);
    trans.back() = axis;

    // transpose logits
    logits =
        CreateBaseOp(
            graph, node, "popart_transpose", {logits}, {}, {{"perm", trans}})
            ->outputs.front();

    // no need to transpose label, transform the label size and reshape later.
    std::transform(
        trans.cbegin(),
        trans.cend(),
        label_transposed.begin(),
        [&label_shape_](int64_t index) { return label_shape_[index]; });
  }

  if (label_transposed.back() == 1) {
    // input shape: [N1, N2, ... , Nk, C]
    // label shape: [N1, N2, ... , Nk, 1]
    // reshape label shape to [N1, N2, ... , Nk]
    std::vector<int64_t> new_shape_(label_transposed.begin(),
                                    label_transposed.end() - 1);
    auto const_before_loss =
        CreateBaseOp(
            graph,
            node,
            "popart_constant",
            {},
            {},
            {{"value", new_shape_},
             {"dims",
              std::vector<int64_t>{static_cast<int64_t>(new_shape_.size())}},
             {"dtype", ONNXDataType::INT64}})
            ->outputs.front();

    cast_and_reshape = CreateBaseOp(graph,
                                    node,
                                    "popart_reshape",
                                    {cast_and_reshape, const_before_loss},
                                    {},
                                    {})
                           ->outputs.front();
  }

  auto log = CreateBaseOp(graph, node, "popart_log", {logits}, {}, {})
                 ->outputs.front();

  bool reshape_back = reduction == 2 && label_transposed.back() == 1;

  final_loss_node = CreateBaseOp(graph,
                                 node,
                                 "popart_nllloss_v2",
                                 {log, cast_and_reshape},
                                 !(reshape_back || append_identity_loss)
                                     ? std::vector<Node *>{output}
                                     : std::vector<Node *>{},
                                 {
                                     {"reduction", reduction},
                                     {"ignoreIndex", ignore_index},
                                     {"inputIsLogProbability", true},
                                 })
                        ->outputs.front();

  if (reshape_back) {
    // reshape output to the shape of input label.
    auto const_after_loss =
        CreateBaseOp(
            graph,
            node,
            "popart_constant",
            {},
            {},
            {{"value", label_shape_},
             {"dims",
              std::vector<int64_t>{static_cast<int64_t>(label_shape_.size())}},
             {"dtype", ONNXDataType::INT64}})
            ->outputs.front();
    final_loss_node =
        CreateBaseOp(graph,
                     node,
                     "popart_reshape",
                     {final_loss_node, const_after_loss},
                     append_identity_loss ? std::vector<Node *>{}
                                          : std::vector<Node *>{output},
                     {})
            ->outputs.front();
  }

  if (append_identity_loss) {
    final_loss_node =
        CreateIdentityLossOp(graph, node, {final_loss_node}, {output}, 2);
  }

  return final_loss_node;
}

Node *cross_entropy2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int reduction = RemoveTailReduction(graph, node, "Y");
  auto logits = GetInputVarNode("X", node);
  auto label = GetInputVarNode("Label", node);
  auto output = GetOutputVarNode("Y", node);
  auto ignore_index = BOOST_GET_CONST(int, op->GetAttr("ignore_index"));
  return cross_entropy_general_handler(graph,
                                       node,
                                       logits,
                                       label,
                                       output,
                                       false, /*soft_label*/
                                       ignore_index,
                                       reduction,
                                       -1); /*axis*/
}

Node *softmax_with_cross_entropy_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int reduction = RemoveTailReduction(graph, node, "Loss");
  auto logits = GetInputVarNode("Logits", node);
  auto label = GetInputVarNode("Label", node);
  auto output = GetOutputVarNode("Loss", node);
  auto ignore_index = BOOST_GET_CONST(int, op->GetAttr("ignore_index"));
  auto axis = BOOST_GET_CONST(int, op->GetAttr("axis"));
  auto soft_label = BOOST_GET_CONST(bool, op->GetAttr("soft_label"));

  logits = CreateSoftmaxOpset11(
               graph, node, {logits}, {GetOutputVarNode("Softmax", node)}, axis)
               ->outputs.front();
  return cross_entropy_general_handler(graph,
                                       node,
                                       logits,
                                       label,
                                       output,
                                       soft_label,
                                       ignore_index,
                                       reduction,
                                       axis);
}

Node *kldiv_loss_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto reduction = ConvertToPopartReduction(
      BOOST_GET_CONST(std::string, op->GetAttr("reduction")));
  if (reduction == 2) {
    reduction = RemoveTailReduction(graph, node, "Loss");
  }
  bool append_identity_loss = is_dynamic_graph();
  bool is_last_var_node = IsLastVarNode(GetOutputVarNode("Loss", node));
  append_identity_loss = append_identity_loss && is_last_var_node;

  // log(pred)
  auto log =
      CreateBaseOp(
          graph, node, "popart_log", {GetInputVarNode("Target", node)}, {}, {})
          ->outputs.front();

  // log(pred) - label
  auto log_minus =
      CreateBaseOp(
          graph, node, "popart_sub", {log, GetInputVarNode("X", node)}, {}, {})
          ->outputs.front();

  // label * (log(pred) - label)
  auto loss =
      CreateBaseOp(graph,
                   node,
                   "popart_mul",
                   {GetInputVarNode("Target", node), log_minus},
                   append_identity_loss || reduction != 2
                       ? std::vector<Node *>{}
                       : std::vector<Node *>{GetOutputVarNode("Loss", node)},
                   {});

  auto attrs = AttributeMap{{"reduce_all", true}, {"keepdims", 0L}};
  if (append_identity_loss) {
    loss = CreateIdentityLossOp(graph,
                                node,
                                loss->outputs,
                                {GetOutputVarNode("Loss", node)},
                                reduction);
  } else if (reduction == 0) {
    // Sum
    loss = CreateBaseOp(graph,
                        node,
                        "popart_reducesum",
                        loss->outputs,
                        {GetOutputVarNode("Loss", node)},
                        attrs);
  } else if (reduction == 1) {
    // Mean
    loss = CreateBaseOp(graph,
                        node,
                        "popart_reducemean",
                        loss->outputs,
                        {GetOutputVarNode("Loss", node)},
                        attrs);
  }
  return loss;
}

Node *binary_cross_entropy_handler(Graph *graph, Node *node) {
  // Out = -1 * weight * (label * log(x) + (1 - label) * log(1 - x))
  int reduction = 2;
  if (is_dynamic_graph()) {
    reduction = RemoveTailReduction(graph, node, "Out");
  }
  bool append_identity_loss =
      is_dynamic_graph() && IsLastVarNode(GetOutputVarNode("Loss", node));

  auto x = GetInputVarNode("X", node);
  auto label = GetInputVarNode("Label", node);
  // log(x)
  auto log =
      CreateBaseOp(graph, node, "popart_log", {x}, {}, {})->outputs.front();

  // label * log(x)
  auto log_mul = CreateBaseOp(graph, node, "popart_mul", {label, log}, {}, {})
                     ->outputs.front();

  // const one
  auto one =
      CreateConst(graph, node, std::vector<float>{1.0}, {1}, GetVarDType(x))
          ->outputs.front();
  // (1 - x)
  auto minus_input = CreateBaseOp(graph, node, "popart_sub", {one, x}, {}, {})
                         ->outputs.front();

  // log(1 - x)
  auto log_minus_input =
      CreateBaseOp(graph, node, "popart_log", {minus_input}, {}, {})
          ->outputs.front();

  // (1 - label)
  auto minus_label =
      CreateBaseOp(graph, node, "popart_sub", {one, label}, {}, {})
          ->outputs.front();

  // (1 - label) * log(1 - x)
  auto minus_log_mul =
      CreateBaseOp(
          graph, node, "popart_mul", {minus_label, log_minus_input}, {}, {})
          ->outputs.front();

  // (label * log(x) + (1 - label) * log(1 - x))
  auto add =
      CreateBaseOp(graph, node, "popart_add", {log_mul, minus_log_mul}, {}, {})
          ->outputs.front();

  // -1 * (label * log(x) + (1 - label) * log(1 - x))
  auto loss = CreateBaseOp(
      graph,
      node,
      "popart_neg",
      {add},
      append_identity_loss ? std::vector<Node *>{}
                           : std::vector<Node *>{GetOutputVarNode("Out", node)},
      {});
  if (append_identity_loss) {
    loss = CreateIdentityLossOp(
        graph, node, loss->outputs, {GetOutputVarNode("Out", node)}, reduction);
  }
  return loss;
}

Node *huber_loss_handler(Graph *graph, Node *node) {
  // if abs(label - input) < delta
  //   huber_loss = 0.5 * (label - input) * (label - input)
  // else
  //   huber_loss = delta * abs(label - input) - 0.5 * delta * delta
  auto *op = node->Op();
  int reduction = 2;
  if (is_dynamic_graph()) {
    reduction = RemoveTailReduction(graph, node, "Out");
  }
  bool append_identity_loss =
      is_dynamic_graph() && IsLastVarNode(GetOutputVarNode("Out", node));

  auto x = GetInputVarNode("X", node);
  auto label = GetInputVarNode("Y", node);
  // (label - input)
  auto diff = CreateBaseOp(graph, node, "popart_sub", {label, x}, {}, {})
                  ->outputs.front();

  // abs(label - input)
  auto abs_diff =
      CreateBaseOp(graph, node, "popart_abs", {diff}, {}, {})->outputs.front();

  // const 0.5
  auto dot_five =
      CreateConst(graph, node, std::vector<float>{0.5}, {1}, GetVarDType(x))
          ->outputs.front();

  // const delta
  auto delta_value = BOOST_GET_CONST(float, op->GetAttr("delta"));
  auto delta =
      CreateConst(
          graph, node, std::vector<float>{delta_value}, {1}, GetVarDType(x))
          ->outputs.front();
  auto delta_square_coff =
      CreateConst(graph,
                  node,
                  std::vector<float>{0.5f * delta_value * delta_value},
                  {1},
                  GetVarDType(x))
          ->outputs.front();

  // (label - input) * (label - input)
  auto square = CreateBaseOp(graph, node, "popart_mul", {diff, diff}, {}, {})
                    ->outputs.front();

  // 0.5 * (label - input) * (label - input)
  auto dot_five_square =
      CreateBaseOp(graph, node, "popart_mul", {dot_five, square}, {}, {})
          ->outputs.front();

  // delta * abs(label - input)
  auto delta_mul_diff =
      CreateBaseOp(graph, node, "popart_mul", {delta, abs_diff}, {}, {})
          ->outputs.front();

  // delta * abs(label - input) - 0.5 * delta * delta
  auto sub_delta_square = CreateBaseOp(graph,
                                       node,
                                       "popart_sub",
                                       {delta_mul_diff, delta_square_coff},
                                       {},
                                       {})
                              ->outputs.front();

  // abs(label - input) < delta
  auto less_cond =
      CreateBaseOp(graph, node, "popart_less", {abs_diff, delta}, {}, {})
          ->outputs.front();
  auto loss = CreateBaseOp(
      graph,
      node,
      "popart_where",
      {less_cond, dot_five_square, sub_delta_square},
      append_identity_loss ? std::vector<Node *>{}
                           : std::vector<Node *>{GetOutputVarNode("Out", node)},
      {});

  if (append_identity_loss) {
    loss = CreateIdentityLossOp(
        graph, node, loss->outputs, {GetOutputVarNode("Out", node)}, reduction);
  }
  return loss;
}

Node *warpctc_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto logits = GetInputVarNode("Logits", node);
  auto label = GetInputVarNode("Label", node);
  auto logits_length = GetInputVarNode("LogitsLength", node);
  auto label_length = GetInputVarNode("LabelLength", node);
  auto blank = BOOST_GET_CONST(int, op->GetAttr("blank"));
  auto norm_by_times = BOOST_GET_CONST(bool, op->GetAttr("norm_by_times"));
  int reduction = 2;
  if (is_dynamic_graph()) {
    reduction = RemoveTailReduction(graph, node, "Loss");
  }
  bool append_identity_loss =
      is_dynamic_graph() && IsLastVarNode(GetOutputVarNode("Loss", node));
  if (norm_by_times) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "norm_by_times is not supported yet in IPU"));
  }

  int axis = -1;
  auto softmax_logits =
      CreateSoftmaxOpset11(graph, node, {logits}, {}, axis)->outputs.front();
  auto log_softmax_logits =
      CreateBaseOp(graph, node, "popart_log", {softmax_logits}, {}, {})
          ->outputs.front();
  auto cast_label = CreateBaseOp(graph,
                                 node,
                                 "popart_cast",
                                 {label},
                                 {},
                                 {{"to", std::string("UINT32")}})
                        ->outputs.front();
  auto cast_logits_length = CreateBaseOp(graph,
                                         node,
                                         "popart_cast",
                                         {logits_length},
                                         {},
                                         {{"to", std::string("UINT32")}})
                                ->outputs.front();
  auto cast_label_length = CreateBaseOp(graph,
                                        node,
                                        "popart_cast",
                                        {label_length},
                                        {},
                                        {{"to", std::string("UINT32")}})
                               ->outputs.front();
  // TODO(czr): zero_infinity is not supported in current sdk which lead
  // difference with paddle result.
  auto loss = CreateBaseOp(
      graph,
      node,
      "popart_ctcloss",
      {log_softmax_logits, cast_label, cast_logits_length, cast_label_length},
      append_identity_loss
          ? std::vector<Node *>{}
          : std::vector<Node *>{GetOutputVarNode("Loss", node)},
      {{"blank", blank},
       {"reduction", reduction},
       {"outDataType", std::string("UNDEFINED")}});
  if (append_identity_loss) {
    loss = CreateIdentityLossOp(
        graph, node, loss->outputs, {GetOutputVarNode("Loss", node)}, 2);
  }
  return loss;
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(identity_loss, identity_loss_handler);
REGISTER_HANDLER(softmax_with_cross_entropy,
                 softmax_with_cross_entropy_handler);
REGISTER_HANDLER(cross_entropy2, cross_entropy2_handler);
REGISTER_HANDLER(kldiv_loss, kldiv_loss_handler);
REGISTER_HANDLER(bce_loss, binary_cross_entropy_handler);
REGISTER_HANDLER(huber_loss, huber_loss_handler);
REGISTER_HANDLER(warpctc, warpctc_handler);

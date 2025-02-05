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
  auto reduction = PADDLE_GET_CONST(int, op->GetAttr("reduction"));
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
    PADDLE_THROW(common::errors::InvalidArgument(
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
  auto ignore_index = PADDLE_GET_CONST(int, op->GetAttr("ignore_index"));
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
  auto ignore_index = PADDLE_GET_CONST(int, op->GetAttr("ignore_index"));
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto soft_label = PADDLE_GET_CONST(bool, op->GetAttr("soft_label"));

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
      PADDLE_GET_CONST(std::string, op->GetAttr("reduction")));
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

Node *sigmoid_cross_entropy_with_logits_handler(Graph *graph, Node *node) {
  // Out = max(logits, 0) - logits * label + log(1 + exp(-abs(logits)))
  auto *op = node->Op();
  int reduction = 2;
  if (is_dynamic_graph()) {
    reduction = RemoveTailReduction(graph, node, "Out");
  }
  bool append_identity_loss =
      is_dynamic_graph() && IsLastVarNode(GetOutputVarNode("Out", node));

  auto logits = GetInputVarNode("X", node);
  auto label = GetInputVarNode("Label", node);
  // sigmoid_cross_entropy_with_logits uses float label as input.
  auto ignore_index_value =
      static_cast<float>(PADDLE_GET_CONST(int, op->GetAttr("ignore_index")));
  auto normalize = PADDLE_GET_CONST(bool, op->GetAttr("normalize"));

  // const
  auto one = CreateConst(
                 graph, node, std::vector<float>{1.0}, {1}, GetVarDType(logits))
                 ->outputs.front();
  auto zero =
      CreateConst(
          graph, node, std::vector<float>{0.0}, {1}, GetVarDType(logits))
          ->outputs.front();
  auto ignore_index = CreateConst(graph,
                                  node,
                                  std::vector<float>{ignore_index_value},
                                  {1},
                                  GetVarDType(label))
                          ->outputs.front();
  // max(logits, 0)
  auto max_zero =
      CreateBaseOp(graph, node, "popart_max", {logits, zero}, {}, {})
          ->outputs.front();

  // logits * label
  auto mul = CreateBaseOp(graph, node, "popart_mul", {logits, label}, {}, {})
                 ->outputs.front();

  // abs(logits)
  auto abs = CreateBaseOp(graph, node, "popart_abs", {logits}, {}, {})
                 ->outputs.front();
  // -abs(logits)
  auto neg_abs =
      CreateBaseOp(graph, node, "popart_neg", {abs}, {}, {})->outputs.front();
  // exp(-abs(logits))
  auto exp_neg_abs = CreateBaseOp(graph, node, "popart_exp", {neg_abs}, {}, {})
                         ->outputs.front();
  // 1+exp(-abs(logits))
  auto log_term =
      CreateBaseOp(graph, node, "popart_add", {exp_neg_abs, one}, {}, {})
          ->outputs.front();
  // log(1+exp(-abs(logits)))
  auto log = CreateBaseOp(graph, node, "popart_log", {log_term}, {}, {})
                 ->outputs.front();

  // max(logits, 0) - logits * label
  auto sub = CreateBaseOp(graph, node, "popart_sub", {max_zero, mul}, {}, {})
                 ->outputs.front();
  // max(logits, 0) - logits * label + log(1 + exp(-abs(logits)))
  auto loss = CreateBaseOp(graph, node, "popart_add", {sub, log}, {}, {})
                  ->outputs.front();

  // label == ignore_index ? 0 : loss
  auto equal_cond =
      CreateBaseOp(graph, node, "popart_equal", {label, ignore_index}, {}, {})
          ->outputs.front();
  loss = CreateBaseOp(graph,
                      node,
                      "popart_where",
                      {equal_cond, zero, loss},
                      append_identity_loss || normalize
                          ? std::vector<Node *>{}
                          : std::vector<Node *>{GetOutputVarNode("Out", node)},
                      {});

  if (normalize) {
    // normalize the output as: loss = loss / sum(label != ignore_index)
    auto not_equal =
        CreateBaseOp(graph, node, "popart_logical_not", {equal_cond}, {}, {})
            ->outputs.front();
    auto mask =
        CreateCast(graph, node, {not_equal}, {}, logits->Var()->GetDataType())
            ->outputs.front();
    auto sum = CreateBaseOp(graph,
                            node,
                            "popart_reducesum",
                            {mask},
                            {},
                            {{"keepdims", int64_t{0}}})
                   ->outputs.front();
    auto eps =
        CreateConst(
            graph, node, std::vector<float>{1e-5}, {1}, GetVarDType(logits))
            ->outputs.front();
    // avoid division by zero
    auto add_eps = CreateBaseOp(graph, node, "popart_add", {sum, eps}, {}, {})
                       ->outputs.front();
    loss =
        CreateBaseOp(graph,
                     node,
                     "popart_div",
                     {loss->outputs[0], add_eps},
                     append_identity_loss
                         ? std::vector<Node *>{}
                         : std::vector<Node *>{GetOutputVarNode("Out", node)},
                     {});
  }

  if (append_identity_loss) {
    loss = CreateIdentityLossOp(
        graph, node, loss->outputs, {GetOutputVarNode("Out", node)}, reduction);
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
      is_dynamic_graph() && IsLastVarNode(GetOutputVarNode("Out", node));

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
  auto delta_value = PADDLE_GET_CONST(float, op->GetAttr("delta"));
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
  auto blank = PADDLE_GET_CONST(int, op->GetAttr("blank"));
  auto norm_by_times = PADDLE_GET_CONST(bool, op->GetAttr("norm_by_times"));
  int reduction = 2;
  if (is_dynamic_graph()) {
    reduction = RemoveTailReduction(graph, node, "Loss");
  }
  bool append_identity_loss =
      is_dynamic_graph() && IsLastVarNode(GetOutputVarNode("Loss", node));
  if (norm_by_times) {
    PADDLE_THROW(common::errors::InvalidArgument(
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
      "popart_ctcloss_v2",
      {log_softmax_logits, cast_label, cast_logits_length, cast_label_length},
      append_identity_loss
          ? std::vector<Node *>{}
          : std::vector<Node *>{GetOutputVarNode("Loss", node)},
      {{"blank", int64_t{blank}},
       {"reduction", reduction},
       {"outDataType", std::string("UNDEFINED")}});
  if (append_identity_loss) {
    loss = CreateIdentityLossOp(
        graph, node, loss->outputs, {GetOutputVarNode("Loss", node)}, 2);
  }
  return loss;
}

Node *rank_loss_handler(Graph *graph, Node *node) {
  // (1.0f + (left - right).exp()).log() - label * (left - right)
  auto label = GetInputVarNode("Label", node);
  auto left = GetInputVarNode("Left", node);
  auto right = GetInputVarNode("Right", node);
  auto output = GetOutputVarNode("Out", node);
  int reduction = 2;
  if (is_dynamic_graph()) {
    reduction = RemoveTailReduction(graph, node, "Out");
  }
  bool append_identity_loss = is_dynamic_graph() && IsLastVarNode(output);

  auto sub = CreateBaseOp(graph, node, "popart_sub", {left, right}, {}, {})
                 ->outputs.front();
  auto mul = CreateBaseOp(graph, node, "popart_mul", {label, sub}, {}, {})
                 ->outputs.front();
  // const
  auto one =
      CreateConst(graph, node, std::vector<float>{1.0}, {1}, GetVarDType(label))
          ->outputs.front();
  auto exp =
      CreateBaseOp(graph, node, "popart_exp", {sub}, {}, {})->outputs.front();
  auto add = CreateBaseOp(graph, node, "popart_add", {one, exp}, {}, {})
                 ->outputs.front();
  auto log =
      CreateBaseOp(graph, node, "popart_log", {add}, {}, {})->outputs.front();
  auto loss = CreateBaseOp(graph,
                           node,
                           "popart_sub",
                           {log, mul},
                           append_identity_loss ? std::vector<Node *>{}
                                                : std::vector<Node *>{output},
                           {})
                  ->outputs.front();
  if (append_identity_loss) {
    loss =
        CreateIdentityLossOp(graph, node, loss->outputs, {output}, reduction);
  }
  return loss;
}

Node *margin_rank_loss_handler(Graph *graph, Node *node) {
  // rank_loss = max(0, -label * (left - right) + margin)
  auto *op = node->Op();
  auto label = GetInputVarNode("Label", node);
  auto left = GetInputVarNode("X1", node);
  auto right = GetInputVarNode("X2", node);
  auto output = GetOutputVarNode("Out", node);
  auto margin_value = PADDLE_GET_CONST(float, op->GetAttr("margin"));
  int reduction = 2;
  if (is_dynamic_graph()) {
    reduction = RemoveTailReduction(graph, node, "Out");
  }
  bool append_identity_loss = is_dynamic_graph() && IsLastVarNode(output);

  // -(left - right)
  auto sub = CreateBaseOp(graph, node, "popart_sub", {right, left}, {}, {})
                 ->outputs.front();
  // -label * (left - right)
  auto mul = CreateBaseOp(graph, node, "popart_mul", {label, sub}, {}, {})
                 ->outputs.front();
  // const
  auto zero =
      CreateConst(graph, node, std::vector<float>{0.0}, {1}, GetVarDType(label))
          ->outputs.front();
  auto margin = CreateConst(graph,
                            node,
                            std::vector<float>{margin_value},
                            {1},
                            GetVarDType(label))
                    ->outputs.front();
  auto margin_add =
      CreateBaseOp(graph, node, "popart_add", {mul, margin}, {}, {})
          ->outputs.front();

  // max(0, term)
  auto loss = CreateBaseOp(graph,
                           node,
                           "popart_max",
                           {zero, margin_add},
                           append_identity_loss ? std::vector<Node *>{}
                                                : std::vector<Node *>{output},
                           {})
                  ->outputs.front();
  if (append_identity_loss) {
    loss =
        CreateIdentityLossOp(graph, node, loss->outputs, {output}, reduction);
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
REGISTER_HANDLER(sigmoid_cross_entropy_with_logits,
                 sigmoid_cross_entropy_with_logits_handler);
REGISTER_HANDLER(kldiv_loss, kldiv_loss_handler);
REGISTER_HANDLER(bce_loss, binary_cross_entropy_handler);
REGISTER_HANDLER(huber_loss, huber_loss_handler);
REGISTER_HANDLER(warpctc, warpctc_handler);
REGISTER_HANDLER(rank_loss, rank_loss_handler);

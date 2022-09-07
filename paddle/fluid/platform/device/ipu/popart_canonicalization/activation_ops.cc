// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

Node *activation_op_handler(Graph *graph, Node *node, const std::string &type) {
  auto new_node = CreateBaseOp(
      graph, node, type, {GetInputVarNode("X", node)}, node->outputs);
  return new_node;
}

Node *abs_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_abs");
}

Node *acos_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_acos");
}

Node *asin_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_asin");
}

Node *atan_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_atan");
}

Node *ceil_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_ceil");
}

Node *cos_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_cos");
}

Node *cosh_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_cosh");
}

Node *erf_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_erf");
}

Node *exp_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_exp");
}

Node *floor_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_floor");
}

Node *log_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_log");
}

Node *reciprocal_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_reciprocal");
}

Node *relu_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_relu");
}

Node *round_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_round");
}

Node *sigmoid_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_sigmoid");
}

Node *sign_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_sign");
}

Node *sin_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_sin");
}

Node *sinh_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_sinh");
}

Node *softplus_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_softplus");
}

Node *softsign_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_softsign");
}

Node *sqrt_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_sqrt");
}

Node *tan_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_tan");
}

Node *tanh_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_tanh");
}

Node *brelu_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto t_min_ = PADDLE_GET_CONST(float, op->GetAttr("t_min"));
  auto t_max_ = PADDLE_GET_CONST(float, op->GetAttr("t_max"));
  auto x = GetInputVarNode("X", node);
  auto cli_min =
      CreateConst(
          graph, node, std::vector<float>{t_min_}, {1}, ONNXDataType::FLOAT)
          ->outputs.front();
  auto clip_max =
      CreateConst(
          graph, node, std::vector<float>{t_max_}, {1}, ONNXDataType::FLOAT)
          ->outputs.front();
  return CreateBaseOp(
      graph, node, "popart_clip", {x, cli_min, clip_max}, node->outputs);
}

Node *gelu_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  // In case of the Op has no `approximate` attr.
  if (!op->HasAttr("approximate")) {
    return activation_op_handler(graph, node, "popart_gelu_v2");
  }
  auto approximate_ = PADDLE_GET_CONST(bool, op->GetAttr("approximate"));
  if (approximate_) {
    return activation_op_handler(graph, node, "popart_gelu_v2");
  } else {
    auto sqrt2 = CreateConst(graph,
                             node,
                             {},
                             {},
                             {{"value", std::vector<float>{1.4142135623730951}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", GetOutputVarDType(node)}});
    auto zero_point_five = CreateConst(graph,
                                       node,
                                       {},
                                       {},
                                       {{"value", std::vector<float>{0.5}},
                                        {"dims", std::vector<int64_t>{1}},
                                        {"dtype", GetOutputVarDType(node)}});
    auto one = CreateConst(graph,
                           node,
                           {},
                           {},
                           {{"value", std::vector<float>{1}},
                            {"dims", std::vector<int64_t>{1}},
                            {"dtype", GetOutputVarDType(node)}});
    auto div = CreateBaseOp(graph,
                            node,
                            "popart_div",
                            {GetInputVarNode("X", node), sqrt2->outputs[0]},
                            {},
                            {});
    auto erf =
        CreateBaseOp(graph, node, "popart_erf", {div->outputs[0]}, {}, {});
    auto add = CreateBaseOp(
        graph, node, "popart_add", {erf->outputs[0], one->outputs[0]}, {}, {});
    auto mul1 = CreateBaseOp(graph,
                             node,
                             "popart_mul",
                             {GetInputVarNode("X", node), add->outputs[0]},
                             {},
                             {});
    return CreateBaseOp(graph,
                        node,
                        "popart_mul",
                        {mul1->outputs[0], zero_point_five->outputs[0]},
                        {GetOutputVarNode("Out", node)},
                        {});
  }
}

Node *log_softmax_handler(Graph *graph, Node *node) {
  auto axis = PADDLE_GET_CONST(int, node->Op()->GetAttr("axis"));
  auto new_softmax = CreateSoftmaxOpset11(graph, node, node->inputs, {}, axis);
  return CreateBaseOp(
      graph, node, "popart_log", new_softmax->outputs, node->outputs);
}

Node *elu_handler(Graph *graph, Node *node) {
  auto alpha_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("alpha"));
  return CreateBaseOp(graph,
                      node,
                      "popart_elu",
                      node->inputs,
                      node->outputs,
                      {
                          {"alpha", alpha_},
                      });
}

Node *hard_shrink_handler(Graph *graph, Node *node) {
  auto threshold_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("threshold"));
  return CreateBaseOp(graph,
                      node,
                      "popart_shrink",
                      node->inputs,
                      node->outputs,
                      {
                          {"lambd", threshold_},
                          {"bias", 0.0f},
                      });
}

Node *hard_sigmoid_handler(Graph *graph, Node *node) {
  auto slope_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("slope"));
  auto offset_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("offset"));
  return CreateBaseOp(graph,
                      node,
                      "popart_hardsigmoid",
                      node->inputs,
                      node->outputs,
                      {
                          {"alpha", slope_},
                          {"beta", offset_},
                      });
}

Node *hard_swish_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  auto scale_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("scale"));
  auto offset_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("offset"));
  auto threshold_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("threshold"));
  auto scale_node =
      CreateConst(graph, node, std::vector<float>{scale_}, {1}, GetVarDType(x))
          ->outputs.front();
  auto offset_node =
      CreateConst(graph, node, std::vector<float>{offset_}, {1}, GetVarDType(x))
          ->outputs.front();
  auto add_node = CreateBaseOp(graph, node, "popart_add", {x, offset_node}, {})
                      ->outputs.front();
  auto cli_min =
      CreateConst(
          graph, node, std::vector<float>{0.0}, {1}, ONNXDataType::FLOAT)
          ->outputs.front();
  auto clip_max =
      CreateConst(
          graph, node, std::vector<float>{threshold_}, {1}, ONNXDataType::FLOAT)
          ->outputs.front();
  auto clip_node =
      CreateBaseOp(
          graph, node, "popart_clip", {add_node, cli_min, clip_max}, {})
          ->outputs.front();
  auto mul_node = CreateBaseOp(graph, node, "popart_mul", {x, clip_node}, {})
                      ->outputs.front();
  return CreateBaseOp(graph,
                      node,
                      "popart_div",
                      {mul_node, scale_node},
                      {GetOutputVarNode("Out", node)});
}

Node *leaky_relu_handler(Graph *graph, Node *node) {
  auto alpha_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("alpha"));
  return CreateBaseOp(graph,
                      node,
                      "popart_leakyrelu",
                      node->inputs,
                      node->outputs,
                      {
                          {"alpha", alpha_},
                      });
}

Node *log10_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  float ln10 = 2.30258509299404568401;
  auto ln10_tensor =
      CreateConst(graph, node, std::vector<float>{ln10}, {1}, GetVarDType(x))
          ->outputs.front();
  auto log = CreateBaseOp(graph, node, "popart_log", {x}, {})->outputs.front();
  return CreateBaseOp(
      graph, node, "popart_div", {log, ln10_tensor}, node->outputs);
}

Node *log1p_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  auto one =
      CreateConst(graph, node, std::vector<float>{1.0}, {1}, GetVarDType(x))
          ->outputs.front();
  auto add =
      CreateBaseOp(graph, node, "popart_add", {x, one}, {})->outputs.front();
  return CreateBaseOp(graph, node, "popart_log", {add}, node->outputs);
}

Node *log2_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  float ln2 = 0.693147180559945309;
  auto ln2_tensor =
      CreateConst(graph, node, std::vector<float>{ln2}, {1}, GetVarDType(x))
          ->outputs.front();
  auto log = CreateBaseOp(graph, node, "popart_log", {x}, {})->outputs.front();
  return CreateBaseOp(
      graph, node, "popart_div", {log, ln2_tensor}, node->outputs);
}

Node *logsigmoid_handler(Graph *graph, Node *node) {
  auto sigmoid =
      CreateBaseOp(
          graph, node, "popart_sigmoid", {GetInputVarNode("X", node)}, {})
          ->outputs.front();
  return CreateBaseOp(graph, node, "popart_log", {sigmoid}, node->outputs);
}

Node *mish_handler(Graph *graph, Node *node) {
  auto threshold_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("threshold"));
  if (!is_float_equal(threshold_, 20.0f)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "For mish op, only support threshold = 20.0"));
  }
  auto x = GetInputVarNode("X", node);
  auto softplus =
      CreateBaseOp(graph, node, "popart_softplus", {x}, {})->outputs.front();
  auto tanh =
      CreateBaseOp(graph, node, "popart_tanh", {softplus}, {})->outputs.front();
  return CreateBaseOp(graph, node, "popart_mul", {x, tanh}, node->outputs);
}

Node *prelu_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  auto alpha = GetInputVarNode("Alpha", node);
  auto out = GetOutputVarNode("Out", node);
  auto x_rank = x->Var()->GetShape().size();
  auto alpha_rank = alpha->Var()->GetShape().size();
  if (x_rank != alpha_rank) {
    if (alpha_rank > 1) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "For prelu op, Only support rank of alpha <=1 while Rank(alpha) != "
          "Rank(input)."));
    }
  }

  if (x_rank != alpha_rank) {
    if (alpha_rank > 1) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "For prelu op, Only support rank of alpha <= 1 while rank of alpha "
          "is not equal with rank of input for operator prelu"));
    }
    if (x_rank <= 1) {
      PADDLE_THROW(
          platform::errors::Unimplemented("For prelu op, Rank of input should "
                                          "greater than 2 for operator prelu"));
    }
    auto shape = std::vector<int64_t>(x_rank - 1, 1);
    shape[0] = -1;
    int64_t size = shape.size();
    auto dim = std::vector<int64_t>{size};
    auto reshape_const =
        CreateConst(graph, node, shape, dim, ONNXDataType::INT64)
            ->outputs.front();
    alpha =
        CreateBaseOp(graph, node, "popart_reshape", {alpha, reshape_const}, {})
            ->outputs.front();
  }
  return CreateBaseOp(graph, node, "popart_prelu", {x, alpha}, {out});
}

Node *relu6_handler(Graph *graph, Node *node) {
  auto threshold_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("threshold"));
  auto cli_min =
      CreateConst(
          graph, node, std::vector<float>{0.0}, {1}, ONNXDataType::FLOAT)
          ->outputs.front();
  auto clip_max =
      CreateConst(
          graph, node, std::vector<float>{threshold_}, {1}, ONNXDataType::FLOAT)
          ->outputs.front();
  return CreateBaseOp(graph,
                      node,
                      "popart_clip",
                      {GetInputVarNode("X", node), cli_min, clip_max},
                      node->outputs);
}

Node *rsqrt_handler(Graph *graph, Node *node) {
  auto rsqrt =
      CreateBaseOp(graph, node, "popart_sqrt", {GetInputVarNode("X", node)}, {})
          ->outputs.front();
  return CreateBaseOp(graph, node, "popart_reciprocal", {rsqrt}, node->outputs);
}

Node *selu_handler(Graph *graph, Node *node) {
  auto alpha_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("alpha"));
  auto scale_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("scale"));
  return CreateBaseOp(graph,
                      node,
                      "popart_selu",
                      node->inputs,
                      node->outputs,
                      {
                          {"alpha", alpha_},
                          {"gamma", scale_},
                      });
}

Node *silu_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  auto sigmoid =
      CreateBaseOp(graph, node, "popart_sigmoid", {x}, {})->outputs.front();
  return CreateBaseOp(graph, node, "popart_mul", {x, sigmoid}, node->outputs);
}

Node *softshrink_handler(Graph *graph, Node *node) {
  auto lambda_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("lambda"));
  return CreateBaseOp(graph,
                      node,
                      "popart_shrink",
                      node->inputs,
                      node->outputs,
                      {
                          {"lambd", lambda_},
                          {"bias", lambda_},
                      });
}

Node *square_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  return CreateBaseOp(graph, node, "popart_mul", {x, x}, node->outputs);
}

Node *swish_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  auto out = GetOutputVarNode("Out", node);
  auto beta_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("beta"));
  auto beta_node =
      CreateConst(graph, node, std::vector<float>{beta_}, {1}, GetVarDType(x))
          ->outputs.front();
  auto beta_x_node = CreateBaseOp(graph, node, "popart_mul", {x, beta_node}, {})
                         ->outputs.front();
  auto sigmod_node =
      CreateBaseOp(graph, node, "popart_sigmoid", {beta_x_node}, {})
          ->outputs.front();
  return CreateBaseOp(graph, node, "popart_mul", {x, sigmod_node}, {out});
}

Node *tanh_shrink_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  auto tanh =
      CreateBaseOp(graph, node, "popart_tanh", {x}, {})->outputs.front();
  return CreateBaseOp(graph, node, "popart_sub", {x, tanh}, node->outputs);
}

Node *thresholded_relu_handler(Graph *graph, Node *node) {
  auto threshold_ = PADDLE_GET_CONST(float, node->Op()->GetAttr("threshold"));
  auto x = GetInputVarNode("X", node);
  return CreateBaseOp(graph,
                      node,
                      "popart_thresholdedrelu",
                      {x},
                      node->outputs,
                      {
                          {"alpha", threshold_},
                      });
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(abs, abs_handler);
REGISTER_HANDLER(acos, acos_handler);
REGISTER_HANDLER(asin, asin_handler);
REGISTER_HANDLER(atan, atan_handler);
REGISTER_HANDLER(ceil, ceil_handler);
REGISTER_HANDLER(cos, cos_handler);
REGISTER_HANDLER(cosh, cosh_handler);
REGISTER_HANDLER(erf, erf_handler);
REGISTER_HANDLER(exp, exp_handler);
REGISTER_HANDLER(floor, floor_handler);
REGISTER_HANDLER(log, log_handler);
REGISTER_HANDLER(reciprocal, reciprocal_handler);
REGISTER_HANDLER(relu, relu_handler);
REGISTER_HANDLER(round, round_handler);
REGISTER_HANDLER(sigmoid, sigmoid_handler);
REGISTER_HANDLER(sign, sign_handler);
REGISTER_HANDLER(sin, sin_handler);
REGISTER_HANDLER(sinh, sinh_handler);
REGISTER_HANDLER(softplus, softplus_handler);
REGISTER_HANDLER(softsign, softsign_handler);
REGISTER_HANDLER(sqrt, sqrt_handler);
REGISTER_HANDLER(tan, tan_handler);
REGISTER_HANDLER(tanh, tanh_handler);
REGISTER_HANDLER(brelu, brelu_handler);
REGISTER_HANDLER(gelu, gelu_handler);
REGISTER_HANDLER(log_softmax, log_softmax_handler);
REGISTER_HANDLER(elu, elu_handler);
REGISTER_HANDLER(hard_shrink, hard_shrink_handler);
REGISTER_HANDLER(hard_sigmoid, hard_sigmoid_handler);
REGISTER_HANDLER(hard_swish, hard_swish_handler);
REGISTER_HANDLER(leaky_relu, leaky_relu_handler);
REGISTER_HANDLER(log10, log10_handler);
REGISTER_HANDLER(log1p, log1p_handler);
REGISTER_HANDLER(log2, log2_handler);
REGISTER_HANDLER(logsigmoid, logsigmoid_handler);
REGISTER_HANDLER(mish, mish_handler);
REGISTER_HANDLER(prelu, prelu_handler);
REGISTER_HANDLER(relu6, relu6_handler);
REGISTER_HANDLER(rsqrt, rsqrt_handler);
REGISTER_HANDLER(selu, selu_handler);
REGISTER_HANDLER(silu, silu_handler);
REGISTER_HANDLER(softshrink, softshrink_handler);
REGISTER_HANDLER(square, square_handler);
REGISTER_HANDLER(swish, swish_handler);
REGISTER_HANDLER(tanh_shrink, tanh_shrink_handler);
REGISTER_HANDLER(thresholded_relu, thresholded_relu_handler);

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <memory>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kRelu = "relu";
const char *const kReluGrad = "relu_grad";
const char *const kRelu6 = "relu6";
const char *const kRelu6Grad = "relu6_grad";
const char *const kSigmoid = "sigmoid";
const char *const kSigmoidGrad = "sigmoid_grad";
const char *const kHardSwish = "hard_swish";
const char *const kHardSwishGrad = "hard_swish_grad";
const char *const kHardSigmoid = "hard_sigmoid";
const char *const kHardSigmoidGrad = "hard_sigmoid_grad";
const char *const kSilu = "silu";
const char *const kSiluGrad = "silu_grad";
const char *const kPow = "pow";
const char *const kPowGrad = "pow_grad";
const char *const kSwish = "swish";
const char *const kSwishGrad = "swish_grad";
const char *const kLeakyRelu = "leaky_relu";
const char *const kLeakyReluGrad = "leaky_relu_grad";
const char *const kSquare = "square";
const char *const kSquareGrad = "square_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReluEquivalenceTrans) {
  auto in = map_inputs["X"].at(0);
  auto rank = in->GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Relu(builder::Transpose(*in, {0, 2, 3, 1})), {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Relu(builder::Transpose(*in, {0, 2, 3, 4, 1})),
          {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(builder::Relu(*in));
}

// dx = dy if y > 0 else 0
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReluGradEquivalenceTrans) {
  builder::Op out = *(map_inputs["Out"].at(0));
  builder::Op ddx = *(map_inputs["Out@GRAD"].at(0));
  auto rank = out.GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::ReluGrad(builder::Transpose(ddx, {0, 2, 3, 1}),
                            builder::Transpose(out, {0, 2, 3, 1})),
          {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::ReluGrad(builder::Transpose(ddx, {0, 2, 3, 4, 1}),
                            builder::Transpose(ddx, {0, 2, 3, 4, 1})),
          {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(builder::ReluGrad(ddx, out));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Relu6EquivalenceTrans) {
  auto in = map_inputs["X"].at(0);
  auto rank = in->GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Relu6(builder::Transpose(*in, {0, 2, 3, 1})), {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Relu6(builder::Transpose(*in, {0, 2, 3, 4, 1})),
          {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(builder::Relu6(*in));
}

//           / 0, x <= 0
// dy / dx = - 1, 0 < x < 6
//           \ 0, x >= 6
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Relu6GradEquivalenceTrans) {
  builder::Op out = *(map_inputs["Out"].at(0));
  builder::Op ddx = *(map_inputs["Out@GRAD"].at(0));
  auto rank = out.GetType().GetShape().size();

  auto Relu6Grad = [](const builder::Op &ddx,
                      const builder::Op &out) -> builder::Op {
    auto const_0 = builder::ZerosLike(out);
    auto const_6 = builder::FullLike(out, 6.0f);
    auto lt1_op = const_0 < out;
    auto lt2_op = out < const_6;
    auto cond_op = lt1_op && lt2_op;
    auto ret_op = builder::Select(cond_op, ddx, const_0);
    return ret_op;
  };
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      out = builder::Transpose(out, {0, 2, 3, 1});
      ddx = builder::Transpose(ddx, {0, 2, 3, 1});
      return std::make_shared<GcuOp>(
          builder::Transpose(Relu6Grad(ddx, out), {0, 3, 1, 2}));
    } else if (rank == 5) {
      out = builder::Transpose(out, {0, 2, 3, 4, 1});
      ddx = builder::Transpose(ddx, {0, 2, 3, 4, 1});
      return std::make_shared<GcuOp>(
          builder::Transpose(Relu6Grad(ddx, out), {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(Relu6Grad(ddx, out));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SigmoidEquivalenceTrans) {
  return std::make_shared<GcuOp>(builder::Sigmoid(*(map_inputs["X"].at(0))));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SigmoidGradEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp out = *(map_inputs["Out"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));

  float alpha_v = op->HasAttr("alpha")
                      ? PADDLE_GET_CONST(float, op->GetAttr("alpha"))
                      : 1.0f;
  auto alpha = builder::FullLike(out, alpha_v);
  return std::make_shared<GcuOp>(out_grad * out * (alpha - out));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, HardSwishEquivalenceTrans) {
  builder::Op x = *(map_inputs["X"].at(0));
  auto rank = x.GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::HardSwish(builder::Transpose(x, {0, 2, 3, 1})),
          {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::HardSwish(builder::Transpose(x, {0, 2, 3, 4, 1})),
          {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(builder::HardSwish(x));
}

// Ref: Paddle/paddle/phi/kernels/funcs/activation_functor.h
IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               HardSwishGradEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  float offset = PADDLE_GET_CONST(float, op->GetAttr("offset"));
  float threshold = PADDLE_GET_CONST(float, op->GetAttr("threshold"));
  float scale = PADDLE_GET_CONST(float, op->GetAttr("scale"));

  std::vector<int64_t> input_perm = {0, 2, 3, 1};
  std::vector<int64_t> out_perm = {0, 3, 1, 2};
  auto input_shape = x_op.GetType().GetShape();
  auto dout_shape = dout_op.GetType().GetShape();
  if (input_shape.size() == 5 && dout_shape.size() == 5) {
    input_perm = {0, 2, 3, 4, 1};
    out_perm = {0, 4, 1, 2, 3};
  }

  bool is_need_transpose =
      (input_shape.size() == 4 && dout_shape.size() == 4) ||
      (input_shape.size() == 5 && dout_shape.size() == 5);
  is_need_transpose &= (running_mode == RunningMode::ADAPTIVE);
  if (!is_need_transpose) {
    builder::Op zero_op = builder::ZerosLike(x_op);
    builder::Op one_op = builder::OnesLike(x_op);
    builder::Op two_op = builder::FullLike(x_op, 2.0f);
    builder::Op offset_op = builder::FullLike(x_op, offset);
    builder::Op threshold_op = builder::FullLike(x_op, threshold);
    builder::Op scale_op = builder::FullLike(x_op, scale);

    builder::Op x_add_offset_op = x_op + offset_op;
    builder::Op x_add_offset_less_than_threshold_op =
        builder::Compare(x_add_offset_op, threshold_op, "LT");
    builder::Op x_add_offset_greater_than_zero_op =
        builder::Compare(x_add_offset_op, zero_op, "GT");
    builder::Op two_mul_x_add_offset_op = two_op * x_op + offset_op;

    builder::Op tmp_1_op =
        builder::Select(x_add_offset_less_than_threshold_op, one_op, zero_op);
    builder::Op tmp_2_op =
        builder::Select(x_add_offset_greater_than_zero_op, one_op, zero_op);

    builder::Op out_op =
        dout_op * (tmp_2_op * two_mul_x_add_offset_op / scale_op * tmp_1_op +
                   one_op - tmp_1_op);

    return std::make_shared<GcuOp>(out_op);
  }

  auto t_in = builder::Transpose(x_op, input_perm);
  auto t_dout = builder::Transpose(dout_op, input_perm);

  builder::Op zero_op = builder::ZerosLike(t_in);
  builder::Op one_op = builder::OnesLike(t_in);
  builder::Op two_op = builder::FullLike(t_in, 2.0f);
  builder::Op offset_op = builder::FullLike(t_in, offset);
  builder::Op threshold_op = builder::FullLike(t_in, threshold);
  builder::Op scale_op = builder::FullLike(t_in, scale);

  builder::Op x_add_offset_op = t_in + offset_op;
  builder::Op x_add_offset_less_than_threshold_op =
      builder::Compare(x_add_offset_op, threshold_op, "LT");
  builder::Op x_add_offset_greater_than_zero_op =
      builder::Compare(x_add_offset_op, zero_op, "GT");
  builder::Op two_mul_x_add_offset_op = two_op * t_in + offset_op;
  builder::Op tmp_1_op =
      builder::Select(x_add_offset_less_than_threshold_op, one_op, zero_op);
  builder::Op tmp_2_op =
      builder::Select(x_add_offset_greater_than_zero_op, one_op, zero_op);
  builder::Op out_op =
      t_dout * (tmp_2_op * two_mul_x_add_offset_op / scale_op * tmp_1_op +
                one_op - tmp_1_op);
  auto t_out = builder::Transpose(out_op, out_perm);

  return std::make_shared<GcuOp>(t_out);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, HardSigmoidEquivalenceTrans) {
  auto *op = node->Op();
  auto slope = PADDLE_GET_CONST(float, op->GetAttr("slope"));
  auto offset = PADDLE_GET_CONST(float, op->GetAttr("offset"));
  return std::make_shared<GcuOp>(
      builder::HardSigmoid(*(map_inputs["X"].at(0)), slope, offset));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               HardSigmoidGradEquivalenceTrans) {
  builder::Op out_op = *(map_inputs["Out"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto slope = PADDLE_GET_CONST(float, op->GetAttr("slope"));

  builder::Op zero_op = builder::ZerosLike(out_op);
  builder::Op one_op = builder::OnesLike(out_op);
  builder::Op slope_op = builder::FullLike(out_op, slope);

  builder::Op out_greater_than_zero_op =
      builder::Compare(out_op, zero_op, "GT");
  builder::Op out_less_than_one_op = builder::Compare(out_op, one_op, "LT");
  builder::Op convert_out_greater_than_zero_op = builder::Convert(
      out_greater_than_zero_op,
      builder::Type(out_greater_than_zero_op.GetType().GetShape(),
                    out_op.GetType().GetPrimitiveType()));
  builder::Op convert_out_less_than_one_op =
      builder::Convert(out_less_than_one_op,
                       builder::Type(out_less_than_one_op.GetType().GetShape(),
                                     out_op.GetType().GetPrimitiveType()));
  builder::Op tmp_op =
      convert_out_greater_than_zero_op * convert_out_less_than_one_op;

  builder::Op dx_op = dout_op * tmp_op * slope_op;

  return std::make_shared<GcuOp>(dx_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SiluEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  auto x_sigmoid = builder::Sigmoid(x_op);
  auto res = x_op * x_sigmoid;
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SiluGradEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto ptype = x_op.GetType().GetPrimitiveType();
  auto input_shape = x_op.GetType().GetShape();
  auto scalar_type = builder::Type(ptype);

  auto constant_ones = builder::OnesLike(x_op);
  auto temp1 = constant_ones + builder::Exp(-x_op);
  auto temp2 = x_op * builder::Exp(-x_op);
  auto silu_grad = (constant_ones / temp1) * (constant_ones + (temp2 / temp1));
  return std::make_shared<GcuOp>(dout_op * silu_grad);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, PowEquivalenceTrans) {
  builder::Op inputs = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto factor = PADDLE_GET_CONST(float, op->GetAttr("factor"));
  auto ptype = inputs.GetType().GetPrimitiveType();
  builder::Op factor_op;
  if (map_inputs.count("FactorTensor") != 0 &&
      map_inputs["FactorTensor"].size() != 0) {
    factor_op = *(map_inputs["FactorTensor"].at(0));
    auto factor_shape = factor_op.GetType().GetShape();
    if ((factor_shape[0] != 1) || (factor_shape.size() != 1)) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("Factor_shape(axis)"
                                            "can only be [1]"));
    }
    if (factor_op.GetType().GetPrimitiveType() != ptype)
      factor_op = builder::Convert(factor_op, builder::Type({1}, ptype));
  } else {
    factor_op = builder::FullLike(inputs, factor, ptype, {1});
  }
  auto res = builder::Pow(inputs, factor_op);
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, PowGradEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto factor = PADDLE_GET_CONST(float, op->GetAttr("factor"));
  auto ptype = x_op.GetType().GetPrimitiveType();
  builder::Op factor_op;
  if (map_inputs.count("FactorTensor") != 0 &&
      map_inputs["FactorTensor"].size() != 0) {
    factor_op = *(map_inputs["FactorTensor"].at(0));
    auto factor_shape = factor_op.GetType().GetShape();
    if ((factor_shape[0] != 1) || (factor_shape.size() != 1)) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Factor_shape(axis) can only be [1]"));
    }
    if (factor_op.GetType().GetPrimitiveType() != ptype)
      factor_op = builder::Convert(factor_op, builder::Type({1}, ptype));
  } else {
    factor_op = builder::FullLike(x_op, factor, ptype, {1});
  }

  auto constant_ones = builder::OnesLike(x_op, ptype, {1});
  auto pow_op = builder::Pow(x_op, (factor_op - constant_ones));
  auto pow_grad = factor_op * pow_op;
  return std::make_shared<GcuOp>(dout_op * pow_grad);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SwishEquivalenceTrans) {
  auto x = *(map_inputs["X"].at(0));
  auto rank = x.GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Swish(builder::Transpose(x, {0, 2, 3, 1})), {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Swish(builder::Transpose(x, {0, 2, 3, 4, 1})),
          {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(builder::Swish(x));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SwishGradEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto rank = x_op.GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      x_op = builder::Transpose(x_op, {0, 2, 3, 1});
      dout_op = builder::Transpose(dout_op, {0, 2, 3, 1});
    } else if (rank == 5) {
      x_op = builder::Transpose(x_op, {0, 2, 3, 4, 1});
      dout_op = builder::Transpose(dout_op, {0, 2, 3, 4, 1});
    }
  }

  float beta = PADDLE_GET_CONST(float, node->Op()->GetAttr("beta"));
  auto constant_beta = builder::FullLike(x_op, beta);
  auto constant_ones = builder::OnesLike(x_op);

  auto limit = TransformUtil::GenerateNumericLimit(
      gcu_builder, x_op.GetType().GetPrimitiveType());
  auto exp_out = builder::Exp((-constant_beta) * x_op);
  auto exp_clamp = builder::Clamp(limit.second, exp_out, limit.first);
  auto temp1 = constant_ones / (constant_ones + exp_clamp);
  auto out = x_op * temp1;
  auto temp2 = temp1 * (constant_ones - (constant_beta * out));
  auto res = dout_op * (constant_beta * out + temp2);
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(res, {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(res, {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, LeakyReluEquivalenceTrans) {
  auto x = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto alpha = PADDLE_GET_CONST(float, op->GetAttr("alpha"));
  auto rank = x.GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::LeakyRelu(builder::Transpose(x, {0, 2, 3, 1}),
                             alpha = alpha),
          {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::LeakyRelu(builder::Transpose(x, {0, 2, 3, 4, 1}),
                             alpha = alpha),
          {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(builder::LeakyRelu(x, alpha = alpha));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               LeakyReluGradEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto rank = x_op.GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      x_op = builder::Transpose(x_op, {0, 2, 3, 1});
      dout_op = builder::Transpose(dout_op, {0, 2, 3, 1});
    } else if (rank == 5) {
      x_op = builder::Transpose(x_op, {0, 2, 3, 4, 1});
      dout_op = builder::Transpose(dout_op, {0, 2, 3, 4, 1});
    }
  }
  float alpha = PADDLE_GET_CONST(float, node->Op()->GetAttr("alpha"));
  auto res = builder::LeakyReluGrad(dout_op, x_op, alpha);
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(res, {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(res, {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SquareEquivalenceTrans) {
  auto x = *(map_inputs["X"].at(0));
  return std::make_shared<GcuOp>(builder::Square(x));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SquareGradEquivalenceTrans) {
  auto x = *(map_inputs["X"].at(0));
  auto dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto rank = x.GetType().GetRank();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      x = builder::Transpose(x, {0, 2, 3, 1});
      dout_op = builder::Transpose(dout_op, {0, 2, 3, 1});
    } else if (rank == 5) {
      x = builder::Transpose(x, {0, 2, 3, 4, 1});
      dout_op = builder::Transpose(dout_op, {0, 2, 3, 4, 1});
    }
  }

  auto const_2 = builder::FullLike(x, 2.0f);

  auto res = dout_op * const_2 * x;
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(res, {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(res, {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kRelu, INSENSITIVE, ReluEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReluGrad, INSENSITIVE, ReluGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kRelu6, INSENSITIVE, Relu6EquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kRelu6Grad, INSENSITIVE, Relu6GradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSigmoid, INSENSITIVE, SigmoidEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSigmoidGrad,
                           INSENSITIVE,
                           SigmoidGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kHardSwish, INSENSITIVE, HardSwishEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kHardSwishGrad,
                           INSENSITIVE,
                           HardSwishGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kHardSigmoid,
                           INSENSITIVE,
                           HardSigmoidEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kHardSigmoidGrad,
                           INSENSITIVE,
                           HardSigmoidGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSilu, INSENSITIVE, SiluEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSiluGrad, INSENSITIVE, SiluGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kPow, INSENSITIVE, PowEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kPowGrad, INSENSITIVE, PowGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSwish, INSENSITIVE, SwishEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSwishGrad, INSENSITIVE, SwishGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLeakyRelu, INSENSITIVE, LeakyReluEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLeakyReluGrad,
                           INSENSITIVE,
                           LeakyReluGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSquare, INSENSITIVE, SquareEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSquareGrad,
                           INSENSITIVE,
                           SquareGradEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle

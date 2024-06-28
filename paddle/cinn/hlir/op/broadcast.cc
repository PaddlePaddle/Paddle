// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/broadcast.h"

#include <iostream>

#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/layout.h"
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {
using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

#define StrategyForBinary(op_name__, pe__)                                     \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(                               \
      const framework::NodeAttr &attrs,                                        \
      const std::vector<ir::Tensor> &inputs,                                   \
      const std::vector<Type> &out_type,                                       \
      const std::vector<std::vector<int>> &output_shapes,                      \
      const Target &target) {                                                  \
    return StrategyForBroadcast(                                               \
        attrs, inputs, out_type, output_shapes, target, #op_name__, pe::pe__); \
  }                                                                            \
  std::shared_ptr<OpStrategy> StrategyFor##pe__##Symbolic(                     \
      const framework::NodeAttr &attrs,                                        \
      const std::vector<ir::Tensor> &inputs,                                   \
      const std::vector<Type> &out_type,                                       \
      const std::vector<std::vector<ir::Dim>> &output_shapes,                  \
      const Target &target) {                                                  \
    return StrategyForBroadcastSymbolic(                                       \
        attrs, inputs, out_type, output_shapes, target, #op_name__, pe::pe__); \
  }

std::shared_ptr<OpStrategy> StrategyForBroadcast(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    ir::Tensor (*pe_func)(const ir::Tensor &A,
                          const ir::Tensor &B,
                          const std::string &output_name,
                          const Expr &axis)) {
  framework::CINNCompute binary_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input argument of " << op_name
                             << " compute is empty! Please check.";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 2U)
            << "at least 2 input tensors for " << op_name << " compute";
        CHECK_GE(pack_args.size(), 3U) << op_name << " 's input is not enough!";
        CHECK(pack_args[2].is_string());
        std::string tensor_name = pack_args[2].operator std::string();
        Expr A_expr = pack_args[0];
        Expr B_expr = pack_args[1];
        CHECK(A_expr.as_tensor());
        CHECK(B_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();
        ir::Tensor B = B_expr.as_tensor_ref();
        Expr axis;
        bool trans_a;
        for (auto &iter : attrs.attr_store) {
          if (iter.first == "axis") {
            axis = Expr(absl::get<int>(iter.second));
            break;
          }
        }
        auto out = pe_func(A, B, tensor_name, axis);
        *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(binary_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy." + op_name + ".x86",
                    1);
  return strategy;
}
std::shared_ptr<OpStrategy> StrategyForBroadcastSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    ir::Tensor (*pe_func)(const ir::Tensor &A,
                          const ir::Tensor &B,
                          const std::string &output_name,
                          const Expr &axis)) {
  framework::CINNCompute binary_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input argument of " << op_name
                             << " compute is empty! Please check.";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 2U)
            << "at least 2 input tensors for " << op_name << " compute";
        CHECK_GE(pack_args.size(), 3U) << op_name << " 's input is not enough!";
        CHECK(pack_args[2].is_string());
        std::string tensor_name = pack_args[2].operator std::string();
        Expr A_expr = pack_args[0];
        Expr B_expr = pack_args[1];
        CHECK(A_expr.as_tensor());
        CHECK(B_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();
        ir::Tensor B = B_expr.as_tensor_ref();
        Expr axis;
        bool trans_a;
        for (auto &iter : attrs.attr_store) {
          if (iter.first == "axis") {
            axis = Expr(absl::get<int>(iter.second));
            break;
          }
        }
        auto out = pe_func(A, B, tensor_name, axis);
        *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      binary_compute, lang::PackedFunc(), "strategy." + op_name + ".x86", 1);
  return strategy;
}

std::vector<shape_t> InferShapeForBroadcast(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2UL);
  std::vector<int> out_shape;

  int axis = -1;
  for (auto &iter : attrs) {
    if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
      break;
    }
  }
  VLOG(3) << "broadcast input shapes are : "
          << utils::Join(inputs_shape[0], ", ") << "; "
          << utils::Join(inputs_shape[1], ", ");
  pe::GetBroadcastOutShape(inputs_shape[0], inputs_shape[1], &out_shape, axis);
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");
  return {out_shape};
}

std::vector<Type> InferDtypeForBroadcast(const std::vector<Type> &inputs_type,
                                         const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

void GenerateEquationsForBroadcast(cinn::adt::config::OpEquationContext *ctx) {
  CHECK(ctx->GetInTensorsRanks().size() == 2)
      << "The inputs is " << ctx->GetInTensorsRanks().size()
      << "! Please check again.";
  CHECK(ctx->GetOutTensorsRanks().size() == 1)
      << "The output is " << ctx->GetOutTensorsRanks().size()
      << "! Please check again.";
  std::uint64_t out_tensor_ranks = ctx->GetOutTensorsRanks().at(0);
  std::uint64_t in_tensor0_ranks = ctx->GetInTensorsRanks().at(0);
  std::uint64_t in_tensor1_ranks = ctx->GetInTensorsRanks().at(1);
  int offset0 = out_tensor_ranks - in_tensor0_ranks;
  for (std::size_t i = 0; i < in_tensor0_ranks; ++i) {
    ctx->Equal(ctx->GetInIteratorTuple(0)->at(i),
               ctx->GetBroadcastedInputIterator(
                   ctx->GetOutIteratorTuple(0)->at(i + offset0),
                   ctx->GetInDimTuple(0)->at(i)));
  }
  int offset1 = out_tensor_ranks - in_tensor1_ranks;
  for (std::size_t i = 0; i < in_tensor1_ranks; ++i) {
    ctx->Equal(ctx->GetInIteratorTuple(1)->at(i),
               ctx->GetBroadcastedInputIterator(
                   ctx->GetOutIteratorTuple(0)->at(i + offset1),
                   ctx->GetInDimTuple(1)->at(i)));
  }
}

std::vector<Type> InferDtypeForBroadcastCmp(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  return {Bool()};
}

std::vector<std::vector<std::string>> InferLayoutForBroadcast(
    const std::vector<std::vector<int>> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  int input_size = input_layouts.size();
  CHECK(input_size == 2U || input_size == 3U)
      << "The input's layouts size is not 2 or 3! Please check again.";
  int axis = -1;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  std::vector<std::string> out_layouts = input_layouts;
  if (input_layouts[0].empty() && input_layouts[1].empty()) {
    return {{input_layouts[0]}, input_layouts};
  } else if (input_layouts[0].empty() || input_layouts[1].empty()) {
    int undef_idx = input_layouts[0] == "" ? 0 : 1;
    int def_idx = 1 - undef_idx;
    CHECK_GE(input_shapes[def_idx].size(), input_shapes[undef_idx].size());
    auto ret = out_layouts[def_idx];
    if (input_size == 2) {
      return {{ret}, {ret, ret}};
    } else {
      return {{ret}, {ret, ret, ret}};
    }
  } else {
    // e.g. NCHWxc + NCHW
    ir::Layout layout0(input_layouts[0]);
    ir::Layout layout1(input_layouts[1]);
    int large_idx = layout0.ndims() >= layout1.ndims() ? 0 : 1;
    auto ret = input_layouts[large_idx];
    if (input_size == 2) {
      return {{ret}, {ret, ret}};
    } else {
      return {{ret}, {ret, ret, ret}};
    }
  }
}

std::shared_ptr<OpStrategy> StrategyForBroadcastTo(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::vector<int> out_shape;
  std::vector<int> broadcast_axes;
  CHECK(attrs.attr_store.count("out_shape"));
  out_shape = absl::get<std::vector<int>>(attrs.attr_store.at("out_shape"));
  CHECK(attrs.attr_store.count("broadcast_axes"));
  broadcast_axes =
      absl::get<std::vector<int>>(attrs.attr_store.at("broadcast_axes"));
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");
  VLOG(3) << "broadcast_axes shape: " << utils::Join(broadcast_axes, ", ");

  framework::CINNCompute broadcast_to_compute([=](lang::Args args,
                                                  lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of broadcast_to compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty())
        << "The input tensors of broadcast_to compute is empty! Please check.";
    CHECK_GE(pack_args.size(), 2U);
    CHECK(pack_args[1].is_string());
    std::string tensor_name = pack_args[1].operator std::string();

    Expr A_expr = pack_args[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out = pe::BroadcastTo(A, out_shape, broadcast_axes, tensor_name);
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(broadcast_to_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.broadcast_to.x86",
                    1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForBroadcastToSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  CHECK_EQ(output_shapes.size(), 1);
  std::vector<ir::Expr> out_shape(output_shapes[0].size());
  std::transform(output_shapes[0].begin(),
                 output_shapes[0].end(),
                 out_shape.begin(),
                 [](const ir::Dim &dim) { return dim->dim_expr; });
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");

  framework::CINNCompute broadcast_to_compute([=](lang::Args args,
                                                  lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of broadcast_to compute is empty! Please check.";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty())
        << "The input tensors of broadcast_to compute is empty! Please check.";
    std::string tensor_name = [&] {
      if (pack_args.size() == 2) {
        return pack_args[1].operator std::string();
      } else {
        PADDLE_ENFORCE_EQ(pack_args.size(),
                          3,
                          ::common::errors::InvalidArgument(
                              "The number of input tensors is wrong. "
                              "The expected inputs is 3, but now is %d.",
                              pack_args.size()));
        return pack_args[2].operator std::string();
      }
    }();

    Expr A_expr = pack_args[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out = pe::BroadcastTo(A, out_shape, tensor_name);
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      broadcast_to_compute, lang::PackedFunc(), "strategy.broadcast_to.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForBroadcastTo(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL)
      << "input_shape size should be one. Please Check.";
  std::vector<int> broadcast_axes;
  std::vector<int> out_shape;
  CHECK(attrs.count("broadcast_axes"));
  CHECK(attrs.count("out_shape"));
  out_shape = absl::get<std::vector<int>>(attrs.at("out_shape"));
  broadcast_axes = absl::get<std::vector<int>>(attrs.at("broadcast_axes"));

  VLOG(3) << "broadcast input shape: " << utils::Join(inputs_shape[0], ", ");
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");
  VLOG(3) << "broadcast_axes shape: " << utils::Join(broadcast_axes, ", ");
  if (inputs_shape[0].empty()) {
    CHECK(broadcast_axes.size() == 1 && broadcast_axes[0] == 0)
        << "broadcast_axes's size should be {1} when the input is 0D-Tensor";
  } else {
    CHECK_EQ(inputs_shape[0].size(), broadcast_axes.size())
        << "broadcast_axes's size should be same with the input shape's size";
  }
  CHECK_GE(out_shape.size(), broadcast_axes.size())
      << "broadcast_axes's size should be no more than out_shape's size";

  return {out_shape};
}

void GenerateEquationsForBroadcastTo(
    cinn::adt::config::OpEquationContext *ctx) {
  CHECK(ctx->GetInTensorsRanks().size() == 1)
      << "The inputs is " << ctx->GetInTensorsRanks().size()
      << "! Please check again.";
  CHECK(ctx->GetOutTensorsRanks().size() == 1)
      << "The output is " << ctx->GetOutTensorsRanks().size()
      << "! Please check again.";
  std::size_t out_tensor_rank = ctx->GetOutTensorsRanks().at(0);
  int start_axis = out_tensor_rank - ctx->GetInTensorsRanks().at(0);
  for (std::size_t i = start_axis; i < out_tensor_rank; ++i) {
    ctx->Equal(ctx->GetInIteratorTuple(0)->at(i - start_axis),
               ctx->GetBroadcastedInputIterator(
                   ctx->GetOutIteratorTuple(0)->at(i),
                   ctx->GetInDimTuple(0)->at(i - start_axis)));
  }
}

std::vector<std::vector<std::string>> InferLayoutForBroadcastTo(
    const std::vector<std::vector<int>> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK(input_layouts.size() == 1U)
      << "The input's layouts size is not 1! Please check again.";
  std::vector<std::string> out_layouts = {""};
  if (attrs.attr_store.count("out_layouts")) {
    out_layouts =
        absl::get<std::vector<std::string>>(attrs.attr_store.at("out_layouts"));
  }
  return {out_layouts, input_layouts};
}

std::vector<Type> InferDtypeForBroadcastGrad(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 3UL);
  // Avoid no need buffer var, like elementwise_add_grad's input X and Y is no
  // need buffer var, in this situation, the X and Y's type is default value
  // FP32, not the real type, we should get the real type from dout.
  std::vector<Type> out_type{inputs_type[0], inputs_type[0]};
  return out_type;
}

std::vector<shape_t> InferShapeForBroadcastGrad(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 3UL);
  std::vector<shape_t> out_shape{inputs_shape[1], inputs_shape[2]};

  return out_shape;
}

std::shared_ptr<OpStrategy> StrategyForBroadcastGrad(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  PADDLE_THROW(phi::errors::Fatal(
      "Gradient operator will be decomposed into several primitive "
      "operators. Please Use Decomposer Program Pass."));
}

StrategyForBinary(elementwise_add, Add);
StrategyForBinary(atan2, Atan2);
StrategyForBinary(elementwise_mul, Multiply);

StrategyForBinary(subtract, Subtract);
StrategyForBinary(divide, Divide);
StrategyForBinary(floor_divide, FloorDivide);
StrategyForBinary(mod, Mod);
StrategyForBinary(remainder, Remainder);
StrategyForBinary(max, Maximum);
StrategyForBinary(min, Minimum);
StrategyForBinary(pow, Pow);

StrategyForBinary(logical_and, LogicalAnd);
StrategyForBinary(logical_or, LogicalOr);
StrategyForBinary(logical_xor, LogicalXOr);
StrategyForBinary(greater_than, Greater);
StrategyForBinary(less_than, Less);
StrategyForBinary(equal, Equal);
StrategyForBinary(not_equal, NotEqual);
StrategyForBinary(greater_equal, GreaterEqual);
StrategyForBinary(less_equal, LessEqual);

StrategyForBinary(bitwise_or, BitwiseOr);
StrategyForBinary(bitwise_xor, BitwiseXor);
StrategyForBinary(bitwise_and, BitwiseAnd);
StrategyForBinary(left_shift, LeftShift);
StrategyForBinary(right_shift, RightShift);
StrategyForBinary(logical_right_shift, LogicalRightShift);

#undef StrategyForBinary

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(broadcast_ops) {
#define CINN_REGISTER_BINARY(op__, op_strategy__)                              \
  CINN_REGISTER_OP(op__)                                                       \
      .describe(#op__ " function")                                             \
      .set_num_inputs(1)                                                       \
      .set_num_outputs(1)                                                      \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                      \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)          \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(              \
          "CINNStrategySymbolic",                                              \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)                \
      .set_attr("infershape",                                                  \
                MakeOpFunction(cinn::hlir::op::InferShapeForBroadcast))        \
      .set_attr("inferdtype",                                                  \
                MakeOpFunction(cinn::hlir::op::InferDtypeForBroadcast))        \
      .set_attr("generate_equations",                                          \
                MakeOpFunction(cinn::hlir::op::GenerateEquationsForBroadcast)) \
      .set_attr("inferlayout",                                                 \
                MakeOpFunction(cinn::hlir::op::InferLayoutForBroadcast))       \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                         \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast)       \
      .set_support_level(4);

#define CINN_REGISTER_BINARY_CMP(op__, op_strategy__)                      \
  CINN_REGISTER_OP(op__)                                                   \
      .describe(#op__ " function")                                         \
      .set_num_inputs(1)                                                   \
      .set_num_outputs(1)                                                  \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                  \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)      \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(          \
          "CINNStrategySymbolic",                                          \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)            \
      .set_attr("infershape",                                              \
                MakeOpFunction(cinn::hlir::op::InferShapeForBroadcast))    \
      .set_attr("inferdtype",                                              \
                MakeOpFunction(cinn::hlir::op::InferDtypeForBroadcastCmp)) \
      .set_attr("inferlayout",                                             \
                MakeOpFunction(cinn::hlir::op::InferLayoutForBroadcast))   \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                     \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast)   \
      .set_support_level(4);

  CINN_REGISTER_BINARY(elementwise_add, Add);
  CINN_REGISTER_BINARY(atan2, Atan2);
  CINN_REGISTER_BINARY(elementwise_mul, Multiply);

  CINN_REGISTER_BINARY(subtract, Subtract);
  CINN_REGISTER_BINARY(divide, Divide);
  CINN_REGISTER_BINARY(floor_divide, FloorDivide);
  CINN_REGISTER_BINARY(mod, Mod);
  CINN_REGISTER_BINARY(remainder, Remainder);
  CINN_REGISTER_BINARY(max, Maximum);
  CINN_REGISTER_BINARY(min, Minimum);
  CINN_REGISTER_BINARY(pow, Pow);

  CINN_REGISTER_BINARY_CMP(logical_and, LogicalAnd);
  CINN_REGISTER_BINARY_CMP(logical_or, LogicalOr);
  CINN_REGISTER_BINARY_CMP(logical_xor, LogicalXOr);
  CINN_REGISTER_BINARY_CMP(greater_than, Greater);
  CINN_REGISTER_BINARY_CMP(less_than, Less);
  CINN_REGISTER_BINARY_CMP(equal, Equal);
  CINN_REGISTER_BINARY_CMP(not_equal, NotEqual);
  CINN_REGISTER_BINARY_CMP(greater_equal, GreaterEqual);
  CINN_REGISTER_BINARY_CMP(less_equal, LessEqual);

  CINN_REGISTER_BINARY(bitwise_or, BitwiseOr);
  CINN_REGISTER_BINARY(bitwise_xor, BitwiseXor);
  CINN_REGISTER_BINARY(bitwise_and, BitwiseAnd);
  CINN_REGISTER_BINARY(left_shift, LeftShift);
  CINN_REGISTER_BINARY(right_shift, RightShift);
  CINN_REGISTER_BINARY(logical_right_shift, LogicalRightShift);
#undef CINN_REGISTER_BINARY

  CINN_REGISTER_OP(broadcast_to)
      .describe("broadcast one tensor to the target shape")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForBroadcastTo)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic",
          cinn::hlir::op::StrategyForBroadcastToSymbolic)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForBroadcastTo))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForBroadcast))
      .set_attr("generate_equations",
                MakeOpFunction(cinn::hlir::op::GenerateEquationsForBroadcastTo))
#if !defined(CINN_WITH_CUDA) && !defined(CINN_WITH_HIP)
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForBroadcastTo))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast)
      .set_support_level(4);

  return true;
}

CINN_REGISTER_HELPER(broadcast_grad_ops) {
  CINN_REGISTER_OP(elementwise_add_grad)
      .describe("The gradient of elementwise_add operator.")
      .set_num_inputs(3)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForBroadcastGrad)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForBroadcastGrad))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForBroadcastGrad));

  return true;
}

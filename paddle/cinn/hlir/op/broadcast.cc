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
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/layout.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

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
  framework::CINNCompute binary_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of %s compute is empty! Please check.",
            op_name));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "At least 2 input tensors for %s compute, but got %d",
                          op_name,
                          pack_args.size()));
    PADDLE_ENFORCE_GE(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "At least 3 input tensors for %s compute, but got %d",
                          op_name,
                          pack_args.size()));
    PADDLE_ENFORCE_EQ(
        pack_args[2].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "Required pack_args[2] must be a string. Please check."));
    std::string tensor_name = pack_args[2].operator std::string();
    Expr A_expr = pack_args[0];
    Expr B_expr = pack_args[1];
    PADDLE_ENFORCE_NOT_NULL(
        A_expr.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
    PADDLE_ENFORCE_NOT_NULL(
        B_expr.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
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
  framework::CINNCompute binary_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of %s compute is empty! Please check.",
            op_name));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "At least 2 input tensors for %s compute, but got %d",
                          op_name,
                          pack_args.size()));
    PADDLE_ENFORCE_GE(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "At least 3 input tensors for %s compute, but got %d",
                          op_name,
                          pack_args.size()));
    PADDLE_ENFORCE_EQ(
        pack_args[2].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "Required pack_args[2] must be a string. Please check."));
    std::string tensor_name = pack_args[2].operator std::string();
    Expr A_expr = pack_args[0];
    Expr B_expr = pack_args[1];
    PADDLE_ENFORCE_NOT_NULL(
        A_expr.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
    PADDLE_ENFORCE_NOT_NULL(
        B_expr.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
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

std::shared_ptr<OpStrategy> StrategyForBroadcastTo(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::vector<int> out_shape;
  std::vector<int> broadcast_axes;
  PADDLE_ENFORCE_GE(
      attrs.attr_store.count("out_shape"),
      1,
      ::common::errors::InvalidArgument(
          "The attrs.attr_store doesn't have the attribute of 'out_shape'."));
  out_shape = absl::get<std::vector<int>>(attrs.attr_store.at("out_shape"));
  PADDLE_ENFORCE_GE(
      attrs.attr_store.count("broadcast_axes"),
      1,
      ::common::errors::InvalidArgument("The attrs.attr_store doesn't have the "
                                        "attribute of 'broadcast_axes'."));
  broadcast_axes =
      absl::get<std::vector<int>>(attrs.attr_store.at("broadcast_axes"));
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");
  VLOG(3) << "broadcast_axes shape: " << utils::Join(broadcast_axes, ", ");

  framework::CINNCompute broadcast_to_compute([=](lang::Args args,
                                                  lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument("The input argument of broadcast_to "
                                          "compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_NE(
        pack_args.empty(),
        true,
        ::common::errors::InvalidArgument("The input tensors of broadcast_to "
                                          "compute is empty! Please check."));
    PADDLE_ENFORCE_GE(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument(
            "Required at least 2 input tensors, but got %d", pack_args.size()));
    PADDLE_ENFORCE_EQ(
        pack_args[1].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "Required pack_args[1] must be a string. Please check."));
    std::string tensor_name = pack_args[1].operator std::string();

    Expr A_expr = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        A_expr.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
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
  PADDLE_ENFORCE_EQ(output_shapes.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The size of output_shapes must be 1, but got %d.",
                        output_shapes.size()));
  std::vector<ir::Expr> out_shape(output_shapes[0].size());
  std::transform(output_shapes[0].begin(),
                 output_shapes[0].end(),
                 out_shape.begin(),
                 [](const ir::Dim &dim) { return dim->dim_expr; });
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");

  framework::CINNCompute broadcast_to_compute([=](lang::Args args,
                                                  lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument("The input argument of broadcast_to "
                                          "compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_NE(
        pack_args.empty(),
        true,
        ::common::errors::InvalidArgument("The input tensors of broadcast_to "
                                          "compute is empty! Please check."));
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
    PADDLE_ENFORCE_NOT_NULL(
        A_expr.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out = pe::BroadcastTo(A, out_shape, tensor_name);
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      broadcast_to_compute, lang::PackedFunc(), "strategy.broadcast_to.x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForBroadcastGrad(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  PADDLE_THROW(::common::errors::Fatal(
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
#define CINN_REGISTER_BINARY(op__, op_strategy__)                        \
  CINN_REGISTER_OP(op__)                                                 \
      .describe(#op__ " function")                                       \
      .set_num_inputs(1)                                                 \
      .set_num_outputs(1)                                                \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)    \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(        \
          "CINNStrategySymbolic",                                        \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)          \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                   \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast) \
      .set_support_level(4);

#define CINN_REGISTER_BINARY_CMP(op__, op_strategy__)                    \
  CINN_REGISTER_OP(op__)                                                 \
      .describe(#op__ " function")                                       \
      .set_num_inputs(1)                                                 \
      .set_num_outputs(1)                                                \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)    \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(        \
          "CINNStrategySymbolic",                                        \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)          \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                   \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast) \
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
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast)
      .set_support_level(4);

  return true;
}

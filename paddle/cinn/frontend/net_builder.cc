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

#include "paddle/cinn/frontend/net_builder.h"

#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace frontend {

using common::Context;
using common::Type;
using hlir::framework::Operator;
using utils::AttributeMap;
using utils::ShapeType;

NetBuilder::NetBuilder(const std::string& name) : name_(name) {}

Program NetBuilder::Build(bool in_reverse) {
  utils::RecordEvent("NetBuilder::Build", utils::EventType::kProgram);
  std::vector<Instruction> instrs;
  if (in_reverse) {
    instrs.reserve(instrs_.size());
    for (auto it = instrs_.rbegin(); it != instrs_.rend(); it++) {
      instrs.emplace_back(*it);
    }
  } else {
    instrs = std::move(instrs_);
  }

  Program program{std::move(instrs), std::move(inputs_)};
  program.Validate();
  return program;
}

void NetBuilder::InferShape(Instruction instr) const {
  using ShapeFunc = std::function<std::vector<ShapeType>(
      const std::vector<ShapeType>&, const AttributeMap&)>;
  using TypeFunc = std::function<std::vector<Type>(const std::vector<Type>&,
                                                   const AttributeMap&)>;
  const auto& op_infershape = Operator::GetAttrs<ShapeFunc>("infershape");
  const auto& op_inferdtype = Operator::GetAttrs<TypeFunc>("inferdtype");

  size_t size = instr->inputs.size();
  std::vector<ShapeType> in_shapes(size);
  std::vector<Type> in_types(size);
  std::transform(instr->inputs.begin(),
                 instr->inputs.end(),
                 in_shapes.begin(),
                 [](const Variable& var) { return var->shape; });
  std::transform(instr->inputs.begin(),
                 instr->inputs.end(),
                 in_types.begin(),
                 [](const Variable& var) { return var->type; });
  auto key = Operator::Get(instr->op_type);
  auto out_shapes = op_infershape[key](in_shapes, instr->attrs);
  auto out_types = op_inferdtype[key](in_types, instr->attrs);

  auto& outs = instr->outputs;
  size_t origin_out_num = outs.size();
  outs.resize(out_shapes.size());
  for (size_t i = origin_out_num; i < outs.size(); i++) {
    outs[i] = Variable();
  }
  for (size_t i = 0; i < outs.size(); i++) {
    outs[i]->shape = out_shapes[i];
    outs[i]->type = out_types[i];
  }
}

const std::vector<Variable>& NetBuilder::CustomInstr(
    const std::string& type,
    const std::vector<Variable>& inputs,
    const AttributeMap& attrs) {
  Instruction instr(type, inputs);
  for (auto& kv : attrs) {
    instr.SetAttr(kv.first, kv.second);
  }
  utils::RecordEvent("NetBuilder." + type, utils::EventType::kProgram);
  InferShape(instr);
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable NetBuilder::BinaryOp(const std::string& op_type,
                              const Variable& lhs,
                              const Variable& rhs,
                              int axis) {
  CHECK_EQ(lhs->type, rhs->type)
      << "The inputs type of op " << op_type << " should be equal!";
  return CustomInstr(op_type, {lhs, rhs}, {{"axis", axis}}).front();
}

Variable NetBuilder::UnaryOp(const std::string& op_type,
                             const Variable& operand) {
  return CustomInstr(op_type, {operand}, {}).front();
}

Variable NetBuilder::Reduce(const std::string& op_type,
                            const Variable& x,
                            const std::vector<int>& dim,
                            bool keep_dim) {
  // TODO(thisjiang): move the reduce simplify to frontend pass
  auto product = std::accumulate(
      x->shape.begin(), x->shape.end(), 1, std::multiplies<int>());
  if (product == 1) {
    if (keep_dim) {
      return Identity(x);
    } else {
      CHECK_GE(x->shape.size(), dim.size())
          << "The inputs rank should be greater than or equal to axes.";
      int new_rank =
          x->shape.size() == dim.size() ? 1 : x->shape.size() - dim.size();
      std::vector<int> new_shape(new_rank, 1);
      return Reshape(x, new_shape);
    }
  }
  // Convert the negative dim to a positive number
  std::vector<int> reduce_dim(dim.begin(), dim.end());
  for (int i = 0; i < dim.size(); i++) {
    if (reduce_dim[i] < 0) {
      reduce_dim[i] = x->shape.size() + reduce_dim[i];
    }
  }
  return CustomInstr(
             op_type, {x}, {{"dim", reduce_dim}, {"keep_dim", keep_dim}})
      .front();
}

#define NETBUILDER_UNARY_OP_DEF(func_name__, op_type__)       \
  Variable NetBuilder::func_name__(const Variable& operand) { \
    return UnaryOp(#op_type__, operand);                      \
  }
NETBUILDER_UNARY_OP_DEF(Sqrt, sqrt)
NETBUILDER_UNARY_OP_DEF(Tanh, tanh)
NETBUILDER_UNARY_OP_DEF(Relu, relu)
NETBUILDER_UNARY_OP_DEF(Gelu, gelu)
NETBUILDER_UNARY_OP_DEF(Sigmoid, sigmoid)
NETBUILDER_UNARY_OP_DEF(Identity, identity)
NETBUILDER_UNARY_OP_DEF(Exp, exp)
NETBUILDER_UNARY_OP_DEF(Erf, erf)
NETBUILDER_UNARY_OP_DEF(Rsqrt, rsqrt)
NETBUILDER_UNARY_OP_DEF(Log, log)
NETBUILDER_UNARY_OP_DEF(Log2, log2)
NETBUILDER_UNARY_OP_DEF(Log10, log10)
NETBUILDER_UNARY_OP_DEF(Floor, floor)
NETBUILDER_UNARY_OP_DEF(Ceil, ceil)
NETBUILDER_UNARY_OP_DEF(Round, round)
NETBUILDER_UNARY_OP_DEF(Trunc, trunc)
NETBUILDER_UNARY_OP_DEF(Sin, sin)
NETBUILDER_UNARY_OP_DEF(Cos, cos)
NETBUILDER_UNARY_OP_DEF(Tan, tan)
NETBUILDER_UNARY_OP_DEF(Sinh, sinh)
NETBUILDER_UNARY_OP_DEF(Cosh, cosh)
NETBUILDER_UNARY_OP_DEF(Asin, asin)
NETBUILDER_UNARY_OP_DEF(Acos, acos)
NETBUILDER_UNARY_OP_DEF(Atan, atan)
NETBUILDER_UNARY_OP_DEF(Asinh, asinh)
NETBUILDER_UNARY_OP_DEF(Acosh, acosh)
NETBUILDER_UNARY_OP_DEF(Atanh, atanh)
NETBUILDER_UNARY_OP_DEF(IsNan, isnan)
NETBUILDER_UNARY_OP_DEF(IsFinite, isfinite)
NETBUILDER_UNARY_OP_DEF(IsInf, isinf)
NETBUILDER_UNARY_OP_DEF(LogicalNot, logical_not)
NETBUILDER_UNARY_OP_DEF(BitwiseNot, bitwise_not)
NETBUILDER_UNARY_OP_DEF(Negative, negative)
NETBUILDER_UNARY_OP_DEF(Sign, sign)
NETBUILDER_UNARY_OP_DEF(Abs, abs)
NETBUILDER_UNARY_OP_DEF(Cbrt, cbrt)
NETBUILDER_UNARY_OP_DEF(Clz, clz)
NETBUILDER_UNARY_OP_DEF(Popc, popc)
NETBUILDER_UNARY_OP_DEF(Reciprocal, reciprocal)

#undef NETBUILDER_UNARY_OP_DEF

#define NETBUILDER_BINARY_OP_DEF(func_name__, op_type__)    \
  Variable NetBuilder::func_name__(                         \
      const Variable& lhs, const Variable& rhs, int axis) { \
    return BinaryOp(#op_type__, lhs, rhs, axis);            \
  }
NETBUILDER_BINARY_OP_DEF(Add, elementwise_add)
NETBUILDER_BINARY_OP_DEF(ElementwiseAdd, elementwise_add)
NETBUILDER_BINARY_OP_DEF(Atan2, atan2)
NETBUILDER_BINARY_OP_DEF(Multiply, elementwise_mul)
NETBUILDER_BINARY_OP_DEF(ElementwiseMul, elementwise_mul)
NETBUILDER_BINARY_OP_DEF(Divide, divide)
NETBUILDER_BINARY_OP_DEF(Subtract, subtract)
NETBUILDER_BINARY_OP_DEF(FloorDivide, floor_divide)
NETBUILDER_BINARY_OP_DEF(Mod, mod)
NETBUILDER_BINARY_OP_DEF(Remainder, remainder)
NETBUILDER_BINARY_OP_DEF(Max, max)
NETBUILDER_BINARY_OP_DEF(Min, min)
NETBUILDER_BINARY_OP_DEF(Pow, pow)
NETBUILDER_BINARY_OP_DEF(LogicalAnd, logical_and)
NETBUILDER_BINARY_OP_DEF(LogicalOr, logical_or)
NETBUILDER_BINARY_OP_DEF(LogicalXor, logical_xor)
NETBUILDER_BINARY_OP_DEF(BitwiseAnd, bitwise_and)
NETBUILDER_BINARY_OP_DEF(BitwiseOr, bitwise_or)
NETBUILDER_BINARY_OP_DEF(BitwiseXor, bitwise_xor)
NETBUILDER_BINARY_OP_DEF(LeftShift, left_shift)
NETBUILDER_BINARY_OP_DEF(RightShift, right_shift)
NETBUILDER_BINARY_OP_DEF(GreaterThan, greater_than);
NETBUILDER_BINARY_OP_DEF(LessThan, less_than);
NETBUILDER_BINARY_OP_DEF(Equal, equal);
NETBUILDER_BINARY_OP_DEF(NotEqual, not_equal);
NETBUILDER_BINARY_OP_DEF(GreaterEqual, greater_equal);
NETBUILDER_BINARY_OP_DEF(LessEqual, less_equal);
NETBUILDER_BINARY_OP_DEF(LogicalRightShift, logical_right_shift);

#undef NETBUILDER_BINARY_OP_DEF

#define NETBUILDER_REDUCE_OP_DEF(func_name__, op_type__)               \
  Variable NetBuilder::func_name__(                                    \
      const Variable& x, const std::vector<int>& dim, bool keep_dim) { \
    std::vector<int> axes = dim;                                       \
    if (axes.size() == 0) {                                            \
      for (int idx = 0; idx < x->shape.size(); ++idx) {                \
        axes.push_back(idx);                                           \
      }                                                                \
    }                                                                  \
    return Reduce(#op_type__, x, axes, keep_dim);                      \
  }

NETBUILDER_REDUCE_OP_DEF(ReduceSum, reduce_sum)
NETBUILDER_REDUCE_OP_DEF(ReduceProd, reduce_prod)
NETBUILDER_REDUCE_OP_DEF(ReduceMax, reduce_max)
NETBUILDER_REDUCE_OP_DEF(ReduceMin, reduce_min)
NETBUILDER_REDUCE_OP_DEF(ReduceAll, reduce_all)
NETBUILDER_REDUCE_OP_DEF(ReduceAny, reduce_any)

#undef NETBUILDER_REDUCE_OP_DEF

Placeholder NetBuilder::CreateInput(const Type& type,
                                    const std::vector<int>& shape,
                                    const std::string& id_hint) {
  std::string id = id_hint.empty() ? Context::Global().NewName("placeholder")
                                   : cinn::utils::TransValidVarName(id_hint);
  inputs_.emplace_back(id);
  auto& var = inputs_.back();
  var->type = type;
  var->shape = shape;
  return Placeholder(var);
}

Placeholder NetBuilder::CreateInput(const Variable& var) {
  VLOG_IF(4, var->shape.empty())
      << "The input's shape is empty, Create 0D-Tensor for " << var->id;
  CHECK(!var->type.is_unk()) << "The input's type is not set yet";
  inputs_.push_back(var);
  return Placeholder(var);
}

Variable NetBuilder::FillConstant(const std::vector<int>& shape,
                                  const std::string& str_value,
                                  const std::string& name,
                                  const std::string& dtype,
                                  bool force_cpu) {
  const auto& type = common::Str2Type(dtype);

  utils::Attribute value;
  if (type.is_float()) {
    value = std::stod(str_value);
  } else if (type.is_int() || type.is_uint()) {
    value = static_cast<int64_t>(std::stoll(str_value));
  } else if (type.is_bool()) {
    value = !cinn::runtime::CheckStringFlagFalse(str_value);
  } else {
    LOG(FATAL) << "FillConstant only support int/float/bool, but here "
               << dtype;
  }
  auto out = CustomInstr("fill_constant",
                         {},
                         {{"shape", shape},
                          {"value", value},
                          {"dtype", dtype},
                          {"force_cpu", force_cpu}})
                 .front();
  if (!name.empty()) {
    out.set_id(cinn::utils::TransValidVarName(name));
  }
  return out;
}

std::vector<Variable> NetBuilder::Split(const Variable& operand,
                                        const std::vector<int>& num_or_sections,
                                        int axis) {
  return CustomInstr("split",
                     {operand},
                     {{"num_or_sections", num_or_sections}, {"axis", axis}});
}

Variable NetBuilder::Concat(const std::vector<Variable>& input_vars, int axis) {
  CHECK(!input_vars.empty())
      << "The inputs of concat op should not be empty! Please check.";
  return CustomInstr("concat", input_vars, {{"axis", axis}}).front();
}

Variable NetBuilder::BroadcastTo(const Variable& operand,
                                 const std::vector<int>& out_shape) {
  auto x_shape_size = operand->shape.size();
  if (x_shape_size == 0) {
    VLOG(4) << "0D-Tensor " << operand->id << " broadcast to shape ("
            << cinn::utils::Join(out_shape, ",") << ")";
    return BroadcastTo(operand, out_shape, {0});
  }
  auto y_shape_size = out_shape.size();
  CHECK_LE(x_shape_size, y_shape_size)
      << "The broadcast_p's input shape dimension should less than the "
         "output's, "
      << "but here (" << x_shape_size << " > " << y_shape_size << ").";

  VLOG(4) << "Try broadcast " << operand->id << " from shape ("
          << cinn::utils::Join(operand->shape, ",") << ") to shape ("
          << cinn::utils::Join(out_shape, ",") << ")";

  std::vector<int> broadcast_axes(x_shape_size, 0);
  if (x_shape_size > 1) {
    for (int i = 1; i <= x_shape_size; ++i) {
      CHECK((out_shape[y_shape_size - i] == operand->shape[x_shape_size - i]) ||
            (operand->shape[x_shape_size - i] == 1))
          << "We cannot broadcast from shape ("
          << cinn::utils::Join(operand->shape, ",") << ") to shape ("
          << cinn::utils::Join(out_shape, ",") << ")";
      broadcast_axes[x_shape_size - i] = y_shape_size - i;
    }
  } else {
    int axis = -1;
    auto x_shape = operand->shape.at(0);
    if (x_shape == 1) {
      // Can broadcast directly, default axis 0
      axis = 0;
    } else {
      // The broadcast axes is the index of the shape in out_shape when the
      // input dimension is 1
      for (int i = 0; i < y_shape_size; ++i) {
        if (out_shape[i] == x_shape) {
          axis = i;
          break;
        }
      }
      CHECK_NE(axis, -1) << "When we broadcast a 1-dimension shape, the number "
                            "should contained in the out_shape. "
                         << "We cannot broadcast from shape ("
                         << cinn::utils::Join(operand->shape, ",")
                         << ") to shape (" << cinn::utils::Join(out_shape, ",")
                         << ")";
    }
    broadcast_axes[0] = axis;
  }

  return BroadcastTo(operand, out_shape, broadcast_axes);
}

Variable NetBuilder::BroadcastTo(const Variable& operand,
                                 const std::vector<int>& out_shape,
                                 const std::vector<int>& broadcast_axes) {
  return CustomInstr(
             "broadcast_to",
             {operand},
             {{"out_shape", out_shape}, {"broadcast_axes", broadcast_axes}})
      .front();
}

Variable NetBuilder::Reshape(const Variable& operand,
                             const std::vector<int>& shape) {
  return CustomInstr("reshape", {operand}, {{"shape", shape}}).front();
}

Variable NetBuilder::Transpose(const Variable& operand,
                               const std::vector<int>& axis) {
  return CustomInstr(
             "transpose",
             {operand},
             {{"axis", utils::GetPositiveAxes(axis, operand->shape.size())}})
      .front();
}

Variable NetBuilder::Slice(const Variable& operand,
                           const std::vector<int>& axes,
                           const std::vector<int>& starts,
                           const std::vector<int>& ends,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& strides,
                           const std::vector<int>& decrease_axis) {
  return CustomInstr("slice",
                     {operand},
                     {{"axes", axes},
                      {"starts", starts},
                      {"ends", ends},
                      {"infer_flags", infer_flags},
                      {"strides", strides},
                      {"decrease_axis", decrease_axis}})
      .front();
}

Variable NetBuilder::SliceAssign(const Variable& input,
                                 const Variable& assign,
                                 const std::vector<int>& axes,
                                 const std::vector<int>& starts,
                                 const std::vector<int>& ends,
                                 const std::vector<int>& strides) {
  return CustomInstr("slice_assign",
                     {input, assign},
                     {{"axes", axes},
                      {"starts", starts},
                      {"ends", ends},
                      {"strides", strides}})
      .front();
}

Variable NetBuilder::Reverse(const Variable& operand,
                             const std::vector<int>& axis) {
  return CustomInstr(
             "reverse",
             {operand},
             {{"axis", utils::GetPositiveAxes(axis, operand->shape.size())}})
      .front();
}

Variable NetBuilder::Select(const Variable& condition,
                            const Variable& true_value,
                            const Variable& false_value) {
  return CustomInstr("select", {condition, true_value, false_value}, {})
      .front();
}

Variable NetBuilder::Gather(const Variable& operand,
                            const Variable& index,
                            int axis) {
  size_t x_ndim = operand->shape.size();
  if (axis < 0) {
    axis += static_cast<int>(x_ndim);
  }
  CHECK_LT(axis, x_ndim) << "Axis must be in [" << -x_ndim << ", " << x_ndim - 1
                         << ").";
  Variable transformed_index = index;
  // If we got 1-D Tensor, the first step is reshape, in order to keep
  // operand.rank == index.rank
  if (index->shape.size() == 1) {
    std::vector<int> index_reshape(x_ndim, 1);
    index_reshape[axis] = index->shape[0];
    transformed_index = Reshape(index, index_reshape);
  }
  // Then we need to broadcast transformed index
  auto broadcast_shape = operand->shape;
  broadcast_shape[axis] = transformed_index->shape[axis];
  transformed_index = BroadcastTo(transformed_index, broadcast_shape);
  return CustomInstr("gather", {operand, transformed_index}, {{"axis", axis}})
      .front();
}

Variable NetBuilder::ScatterAssign(const Variable& operand,
                                   const Variable& updates,
                                   const Variable& index,
                                   int axis) {
  return CustomInstr(
             "scatter_assign", {operand, updates, index}, {{"axis", axis}})
      .front();
}

Variable NetBuilder::ScatterAdd(const Variable& operand,
                                const Variable& updates,
                                const Variable& index,
                                int axis) {
  return CustomInstr("scatter_add", {operand, updates, index}, {{"axis", axis}})
      .front();
}

Variable NetBuilder::IsClose(const Variable& x,
                             const Variable& y,
                             float rtol,
                             float atol,
                             bool equal_nan) {
  return CustomInstr("isclose",
                     {x, y},
                     {{"rtol", rtol}, {"atol", atol}, {"equal_nan", equal_nan}})
      .front();
}

Variable NetBuilder::Mul(const Variable& a,
                         const Variable& b,
                         int x_num_col_dims,
                         int y_num_col_dims,
                         bool is_infer) {
  return CustomInstr("mul",
                     {a, b},
                     {{"x_num_col_dims", x_num_col_dims},
                      {"y_num_col_dims", y_num_col_dims},
                      {"is_infer", is_infer}})
      .front();
}

const std::vector<Variable>& NetBuilder::ElementwiseAddGrad(
    const Variable& dout, const Variable& x, const Variable& y, int axis) {
  return CustomInstr("elementwise_add_grad", {dout, x, y}, {{"axis", axis}});
}

Variable NetBuilder::Relu6(const Variable& a, float threshold) {
  return CustomInstr("relu6", {a}, {{"threshold", threshold}}).front();
}

Variable NetBuilder::ReluGrad(const Variable& lhs, const Variable& rhs) {
  return CustomInstr("relu_grad", {lhs, rhs}, {}).front();
}

Variable NetBuilder::GatherNd(const Variable& x, const Variable& index) {
  return CustomInstr("gather_nd", {x, index}, {}).front();
}

Variable NetBuilder::Cast(const Variable& operand, const std::string& dtype) {
  return CustomInstr("cast", {operand}, {{"dtype", dtype}}).front();
}

Variable NetBuilder::BitcastConvert(const Variable& operand,
                                    const std::string& dtype) {
  std::string input_data_type = common::Type2Str(operand->type);
  return CustomInstr("bitcast_convert",
                     {operand},
                     {{"dtype", dtype}, {"input_data_type", input_data_type}})
      .front();
}

Variable NetBuilder::OneHot(const Variable& indices,
                            const Variable& on_value,
                            const Variable& off_value,
                            const int depth,
                            const int axis,
                            const std::string& dtype) {
  return CustomInstr("one_hot",
                     {indices, on_value, off_value},
                     {{"depth", depth}, {"axis", axis}, {"dtype", dtype}})
      .front();
}

Variable NetBuilder::Squeeze(const Variable& operand,
                             const std::vector<int>& axes) {
  return CustomInstr("squeeze", {operand}, {{"axes", axes}}).front();
}

Variable NetBuilder::ExpandDims(const Variable& operand,
                                const cinn::utils::ShapeType& axes) {
  return CustomInstr("expand_dims", {operand}, {{"axes", axes}}).front();
}

Variable NetBuilder::Conv(const Variable& lhs,
                          const Variable& rhs,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations,
                          int groups,
                          const std::string& conv_type,
                          const std::string& data_format,
                          const std::string& padding_algorithm,
                          const std::vector<int>& output_shape) {
  return CustomInstr("conv2d",
                     {lhs, rhs},
                     {{"stride", strides},
                      {"padding", paddings},
                      {"dilation", dilations},
                      {"groups", groups},
                      {"conv_type", conv_type},
                      {"data_format", data_format},
                      {"padding_algorithm", padding_algorithm},
                      {"output_shape", output_shape}})
      .front();
}

std::vector<Variable> NetBuilder::ArgSort(const Variable& operand,
                                          const int& axis,
                                          const bool& is_ascend) {
  return CustomInstr(
      "argsort", {operand}, {{"axis", axis}, {"is_ascend", is_ascend}});
}

Variable NetBuilder::Sort(const Variable& operand,
                          const int& axis,
                          const bool& is_ascend) {
  return CustomInstr(
             "sort", {operand}, {{"axis", axis}, {"is_ascend", is_ascend}})
      .front();
}

Variable NetBuilder::Argmax(const Variable& x,
                            const int& axis,
                            const bool& keep_dim) {
  return CustomInstr("argmax", {x}, {{"axis", axis}, {"keep_dim", keep_dim}})
      .front();
}

Variable NetBuilder::Argmin(const Variable& x,
                            const int& axis,
                            const bool& keep_dim) {
  return CustomInstr("argmin", {x}, {{"axis", axis}, {"keep_dim", keep_dim}})
      .front();
}

Variable NetBuilder::LookupTable(const Variable& table,
                                 const Variable& ids,
                                 int64_t padding_idx) {
  return CustomInstr(
             "lookup_table", {table, ids}, {{"padding_idx", padding_idx}})
      .front();
}

Variable NetBuilder::Conv2d(const Variable& a,
                            const Variable& b,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& dilations,
                            int groups,
                            const std::string& data_format,
                            const std::string& padding_algorithm) {
  return Conv(a,
              b,
              strides,
              paddings,
              dilations,
              groups,
              "forward",
              data_format,
              padding_algorithm,
              {});
}

Variable NetBuilder::DepthwiseConv2d(const Variable& a,
                                     const Variable& b,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& dilations,
                                     int groups,
                                     const std::string& data_format,
                                     const std::string& padding_algorithm) {
  return CustomInstr("depthwise_conv2d",
                     {a, b},
                     {{"stride", strides},
                      {"padding", paddings},
                      {"dilation", dilations},
                      {"groups", groups},
                      {"data_format", data_format},
                      {"padding_algorithm", padding_algorithm}})
      .front();
}

std::vector<int> UpdatePool2dKernelSize(const std::vector<int>& x_shape,
                                        const std::vector<int>& ksize,
                                        const bool global_pooling,
                                        const std::string& data_format) {
  std::vector<int> new_ksize{ksize};
  // Setting h/w_axis according to data_format
  int height_axis = -1;
  int width_axis = -1;
  if (data_format == "NCHW") {
    height_axis = 2;
    width_axis = 3;
  } else if (data_format == "NHWC") {
    height_axis = 1;
    width_axis = 2;
  } else {
    LOG(FATAL) << "Unsupport data_format: " << data_format;
  }
  if (global_pooling) {
    new_ksize[0] = x_shape[height_axis];
    new_ksize[1] = x_shape[width_axis];
  }
  return new_ksize;
}

std::vector<int> UpdatePool2dPaddings(const std::vector<int>& paddings,
                                      const std::vector<int>& x_shape,
                                      const std::vector<int>& ksize,
                                      const std::vector<int>& stride,
                                      const bool global_pooling,
                                      const bool adaptive,
                                      const std::string& padding_algorithm,
                                      const std::string& data_format) {
  std::vector<int> new_paddings{paddings};
  if (paddings.size() == 2) {
    new_paddings.insert(new_paddings.end(), paddings.begin(), paddings.end());
  }
  CHECK_EQ(new_paddings.size(), 4)
      << "Padding size must be 2 or 4, but got: " << paddings.size();
  // Setting h/w_axis according to data_format
  int height_axis = -1;
  int width_axis = -1;
  if (data_format == "NCHW") {
    height_axis = 2;
    width_axis = 3;
  } else if (data_format == "NHWC") {
    height_axis = 1;
    width_axis = 2;
  } else {
    LOG(FATAL) << "Unsupport data_format: " << data_format;
  }
  // When padding_algorithm is VALID, set paddings to [0, 0, 0, 0].
  // When padding_algorithm is SAME, the calculation formula of padding is as
  // follows: output_h/w = ceil(input_h/w / stride_h/w) padding_sum_h/w =
  // (output_h/w - 1) * stride_h/w + kernel_h/w - input_h/w padding_top/left =
  // padding_sum_h/w / 2; padding_bottom/right = padding_sum_h/w -
  // padding_top/left
  if (padding_algorithm == "VALID") {
    new_paddings = {0, 0, 0, 0};
  } else if (padding_algorithm == "SAME") {
    int out_size_h = (x_shape[height_axis] + stride[0] - 1) / stride[0];
    int out_size_w = (x_shape[width_axis] + stride[1] - 1) / stride[1];
    int pad_sum_h = std::max(
        (out_size_h - 1) * stride[0] + ksize[0] - x_shape[height_axis], 0);
    int pad_sum_w = std::max(
        (out_size_w - 1) * stride[1] + ksize[1] - x_shape[width_axis], 0);
    int pad_top = pad_sum_h / 2;
    int pad_bottom = pad_sum_h - pad_top;
    int pad_left = pad_sum_w / 2;
    int pad_right = pad_sum_w - pad_left;
    new_paddings = {pad_top, pad_left, pad_bottom, pad_right};
  }
  // When global_pooling or adaptive is true, set paddings to [0, 0, 0, 0].
  if (global_pooling || adaptive) {
    new_paddings = {0, 0, 0, 0};
  }
  return new_paddings;
}

Variable NetBuilder::Pool2d(const Variable& a,
                            const std::string& pooling_type,
                            const std::vector<int>& ksize,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            bool ceil_mode,
                            bool exclusive,
                            bool global_pooling,
                            const std::string& data_format,
                            bool adaptive,
                            const std::string& padding_algorithm) {
  // Check input dim
  CHECK_EQ(a->shape.size(), 4)
      << "Input's dim must be 4, but " << a->id << "'s shape is ["
      << cinn::utils::Join(a->shape, ", ") << "].";
  // Transform pool_type
  std::string pool_type;
  std::transform(pooling_type.begin(),
                 pooling_type.end(),
                 std::back_inserter(pool_type),
                 [](unsigned char c) { return std::tolower(c); });
  CHECK(pool_type == "avg" || pool_type == "max")
      << "Pool_type must be avg or max, but got: " << pool_type;
  // Transform ksize
  std::vector<int> input_ksize{ksize};
  if (input_ksize.size() == 1) {
    input_ksize.insert(input_ksize.end(), ksize.begin(), ksize.end());
  }
  CHECK_EQ(input_ksize.size(), 2)
      << "Kernel_size length must be 1 or 2, but got: " << ksize.size();
  // Transform stride
  std::vector<int> new_strides{strides};
  if (new_strides.size() == 1) {
    new_strides.insert(new_strides.end(), strides.begin(), strides.end());
  }
  CHECK_EQ(new_strides.size(), 2)
      << "Stride length must be 1 or 2, but got: " << strides.size();
  CHECK(new_strides[0] > 0 && new_strides[1] > 0)
      << "the value of kernel size for pool2d should greater than 0.";
  // Transform data_format
  std::string new_data_format{data_format};
  if (new_data_format == "AnyLayout") {
    new_data_format.assign("NCHW");
  }
  CHECK(new_data_format == "NCHW" || new_data_format == "NHWC")
      << "Data_format must be AnyLayout/NCHW/NHWC, but got: " << data_format;
  // Check padding_algorithm
  CHECK(padding_algorithm == "EXPLICIT" || padding_algorithm == "SAME" ||
        padding_algorithm == "VALID")
      << "Padding_algorithm must be EXPLICIT/SAME/VALID, but got: "
      << padding_algorithm;
  utils::AttributeMap attrs = {{"pool_type", pool_type},
                               {"origin_kernel_size", input_ksize},
                               {"stride_size", new_strides},
                               {"origin_padding_size", paddings},
                               {"ceil_mode", ceil_mode},
                               {"exclusive", exclusive},
                               {"origin_global_pooling", global_pooling},
                               {"data_format", new_data_format},
                               {"origin_adaptive", adaptive},
                               {"padding_algorithm", padding_algorithm}};
  // In avg_pool2d, if global_pooling = false, adaptive = true and ksize is [1,
  // 1], we turn off adaptive and use global pooling instead
  if (pooling_type == "avg" && !global_pooling && adaptive &&
      input_ksize[0] == 1 && input_ksize[1] == 1) {
    VLOG(4) << "In avg_pool2d, got global_pooling = false, adaptive = true, "
               "ksize = [1, 1], turn off adaptive and "
               "trans to global_pooling";
    adaptive = false;
    global_pooling = true;
  }
  // Transform paddings
  auto new_paddings = UpdatePool2dPaddings(paddings,
                                           a->shape,
                                           input_ksize,
                                           new_strides,
                                           global_pooling,
                                           adaptive,
                                           padding_algorithm,
                                           new_data_format);
  // Update kernel_size
  auto new_ksize = UpdatePool2dKernelSize(
      a->shape, input_ksize, global_pooling, new_data_format);
  attrs["kernel_size"] = new_ksize;
  attrs["padding_size"] = new_paddings;
  attrs["adaptive"] = adaptive;
  attrs["global_pooling"] = global_pooling;
  return CustomInstr("pool2d", {a}, attrs).front();
}

Variable NetBuilder::Pool2dGrad(const Variable& x,
                                const Variable& y,
                                const Variable& dy,
                                const std::string& pooling_type,
                                const std::vector<int>& ksize,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool ceil_mode,
                                bool exclusive,
                                bool global_pooling,
                                const std::string& data_format,
                                bool adaptive,
                                const std::string& padding_algorithm) {
  // Transform pool_type
  std::string pool_type;
  std::transform(pooling_type.begin(),
                 pooling_type.end(),
                 std::back_inserter(pool_type),
                 [](unsigned char c) { return std::tolower(c); });
  CHECK(pool_type == "avg" || pool_type == "max")
      << "Pool_type must be avg or max, but got: " << pool_type;
  // Transform ksize
  std::vector<int> input_ksize{ksize};
  if (input_ksize.size() == 1) {
    input_ksize.insert(input_ksize.end(), ksize.begin(), ksize.end());
  }
  CHECK_EQ(input_ksize.size(), 2)
      << "Kernel_size length must be 1 or 2, but got: " << ksize.size();
  // Transform stride
  std::vector<int> new_strides{strides};
  if (new_strides.size() == 1) {
    new_strides.insert(new_strides.end(), strides.begin(), strides.end());
  }
  CHECK_EQ(new_strides.size(), 2)
      << "Stride length must be 1 or 2, but got: " << strides.size();
  CHECK(new_strides[0] > 0 && new_strides[1] > 0)
      << "the value of kernel size for pool2d should greater than 0.";
  // Transform data_format
  std::string new_data_format{data_format};
  if (new_data_format == "AnyLayout") {
    new_data_format.assign("NCHW");
  }
  CHECK(new_data_format == "NCHW" || new_data_format == "NHWC")
      << "Data_format must be AnyLayout/NCHW/NHWC, but got: " << data_format;
  // Check padding_algorithm
  CHECK(padding_algorithm == "EXPLICIT" || padding_algorithm == "SAME" ||
        padding_algorithm == "VALID")
      << "Padding_algorithm must be EXPLICIT/SAME/VALID, but got: "
      << padding_algorithm;
  // In avg_pool2d, if global_pooling = false, adaptive = true and ksize is [1,
  // 1], we turn off adaptive and use global pooling instead
  if (pooling_type == "avg" && !global_pooling && adaptive &&
      input_ksize[0] == 1 && input_ksize[1] == 1) {
    VLOG(4) << "In avg_pool2d, got global_pooling = false, adaptive = true, "
               "ksize = [1, 1], turn off adaptive and "
               "trans to global_pooling";
    adaptive = false;
    global_pooling = true;
  }
  // Transform paddings
  auto new_paddings = UpdatePool2dPaddings(paddings,
                                           x->shape,
                                           input_ksize,
                                           new_strides,
                                           global_pooling,
                                           adaptive,
                                           padding_algorithm,
                                           new_data_format);
  // Update kernel_size
  auto new_ksize = UpdatePool2dKernelSize(
      x->shape, input_ksize, global_pooling, new_data_format);
  return CustomInstr("pool2d_grad",
                     {x, y, dy},
                     {{"pool_type", pool_type},
                      {"kernel_size", new_ksize},
                      {"stride_size", new_strides},
                      {"padding_size", new_paddings},
                      {"ceil_mode", ceil_mode},
                      {"exclusive", exclusive},
                      {"global_pooling", global_pooling},
                      {"data_format", new_data_format},
                      {"adaptive", adaptive},
                      {"padding_algorithm", padding_algorithm}})
      .front();
}

Variable NetBuilder::Repeat(const Variable& x, int repeats, int axis) {
  return CustomInstr("repeat", {x}, {{"repeats", repeats}, {"axis", axis}})
      .front();
}

Variable NetBuilder::Resize(const Variable& x,
                            const std::vector<int>& out_shape,
                            const std::string& mode) {
  return CustomInstr("resize", {x}, {{"out_shape", out_shape}, {"mode", mode}})
      .front();
}

std::vector<Variable> NetBuilder::BatchNorm(const Variable& a,
                                            const Variable& scale,
                                            const Variable& bias,
                                            const Variable& mean,
                                            const Variable& variance,
                                            float epsilon,
                                            float momentum,
                                            const std::string& data_layout,
                                            bool is_test) {
  std::string op_type = is_test ? "batch_norm" : "batch_norm_train";
  return CustomInstr(op_type,
                     {a, scale, bias, mean, variance},
                     {{"epsilon", epsilon},
                      {"momentum", momentum},
                      {"data_layout", data_layout}});
}

// batch norm grad, output(grad_x, grad_scale, grad_bias)
std::vector<Variable> NetBuilder::BatchNormGrad(
    const Variable& dy,
    const Variable& x,
    const Variable& scale,
    const Variable& save_mean,
    const Variable& save_variance,
    const float epsilon,
    const std::string& data_layout) {
  return CustomInstr("batch_norm_grad",
                     {dy, x, scale, save_mean, save_variance},
                     {{"epsilon", epsilon}, {"data_layout", data_layout}});
}

Variable NetBuilder::Scale(const Variable& a,
                           float scale,
                           float bias,
                           bool bias_after_scale) {
  return CustomInstr("scale",
                     {a},
                     {{"scale", scale},
                      {"bias", bias},
                      {"bias_after_scale", bias_after_scale}})
      .front();
}

Variable NetBuilder::Softmax(const Variable& a,
                             const std::vector<int>& axes,
                             const std::string& mode,
                             const std::string& data_format) {
  return CustomInstr(
             "softmax",
             {a},
             {{"axes", axes}, {"mode", mode}, {"data_format", data_format}})
      .front();
}

Variable NetBuilder::DropoutInfer(const Variable& a,
                                  float dropout_prob,
                                  const std::string& dropout_implementation) {
  return CustomInstr("dropout_infer",
                     {a},
                     {{"dropout_prob", dropout_prob},
                      {"dropout_implementation", dropout_implementation}})
      .front();
}

Variable NetBuilder::Sum(const std::vector<Variable>& inputs) {
  return CustomInstr("sum", inputs, {}).front();
}

Variable NetBuilder::Arange(const float start,
                            const float stop,
                            const float step,
                            const std::string& dtype) {
  return CustomInstr("arange",
                     {},
                     {{"start", start},
                      {"stop", stop},
                      {"step", step},
                      {"dtype", dtype}})
      .front();
}

Variable NetBuilder::Flip(const Variable& operand,
                          const std::vector<int>& axes) {
  return CustomInstr(
             "reverse",
             {operand},
             {{"axis", utils::GetPositiveAxes(axes, operand->shape.size())}})
      .front();
}

Variable NetBuilder::Matmul(const Variable& x,
                            const Variable& y,
                            bool trans_x,
                            bool trans_y,
                            float alpha) {
  return CustomInstr(
             "matmul",
             {x, y},
             {{"trans_a", trans_x}, {"trans_b", trans_y}, {"alpha", alpha}})
      .front();
}

Variable NetBuilder::GaussianRandom(const std::vector<int>& shape,
                                    float mean,
                                    float std,
                                    int seed,
                                    const std::string& dtype) {
  return CustomInstr("gaussian_random",
                     {},
                     {{"shape", shape},
                      {"mean", mean},
                      {"std", std},
                      {"seed", seed},
                      {"dtype", dtype}})
      .front();
}

Variable NetBuilder::UniformRandom(const std::vector<int>& shape,
                                   float min,
                                   float max,
                                   int seed,
                                   const std::string& dtype,
                                   int diag_num,
                                   int diag_step,
                                   float diag_val) {
  auto uniform_out = CustomInstr("uniform_random",
                                 {},
                                 {{"shape", shape},
                                  {"min", min},
                                  {"max", max},
                                  {"seed", seed},
                                  {"dtype", dtype}})
                         .front();
  if (min == 0.0f && max == 1.0f) {
    return uniform_out;
  }
  auto uniform_range =
      FillConstant(shape, max - min, UniqName("uniform_range"), dtype);
  auto uniform_mul_out = Multiply(uniform_out, uniform_range);
  auto uniform_min = FillConstant(shape, min, UniqName("uniform_min"), dtype);
  auto uniform_res = Add(uniform_mul_out, uniform_min);
  if (diag_num > 0) {
    int numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    CHECK_GT(numel, (diag_num - 1) * (diag_step + 1))
        << "(diag_num - 1) * (diag_step + 1) should smaller than numel!";
    auto diag_index = Arange(0.0f,
                             static_cast<float>(diag_num * (diag_step + 1)),
                             static_cast<float>(diag_step + 1),
                             "int32");
    auto diag_val_tensor =
        FillConstant(diag_index->shape, diag_val, "diag_val", dtype);
    auto uniform_flatten = Reshape(uniform_res, {-1});
    auto uniform_scatter =
        ScatterAssign(uniform_flatten, diag_val_tensor, diag_index);
    uniform_res = Reshape(uniform_scatter, shape);
  }
  return uniform_res;
}

Variable NetBuilder::RandInt(const std::vector<int>& shape,
                             int min,
                             int max,
                             int seed,
                             const std::string& dtype) {
  CHECK_GT(max, min) << "max: " << max << "should greater than"
                     << "min: " << min;
  auto randint_out =
      CustomInstr(
          "randint", {}, {{"shape", shape}, {"seed", seed}, {"dtype", dtype}})
          .front();
  randint_out = Cast(randint_out, dtype);
  auto randint_range =
      FillConstant(shape, max - min, UniqName("randint_range"), dtype);
  auto randint_mod = Mod(randint_out, randint_range);
  auto randint_min = FillConstant(shape, min, UniqName("randint_min"), dtype);
  auto randint_ret = Add(randint_mod, randint_min);
  return randint_ret;
}

Variable NetBuilder::Cholesky(const Variable& x, bool upper) {
  auto cholesky_out = CustomInstr("cholesky", {x}, {{"upper", upper}}).front();
  // Set upper/lower triangle of matrices to 0
  auto x_ndim = x->shape.size();
  CHECK_GE(x_ndim, 2)
      << "The input matrix x shape size should >= 2! Please check again.";
  CHECK_EQ(x->shape[x_ndim - 1], x->shape[x_ndim - 2])
      << "The input matrix x's last 2 dimensions must be the same! Please "
         "check again.";
  int m = x->shape[x_ndim - 1];
  auto m_tensor = FillConstant({m * m}, m);
  auto index = Arange(0.0f, static_cast<float>(m * m), 1.0f, "int32");
  auto index_row = Mod(index, m_tensor);
  auto index_col = FloorDivide(index, m_tensor);
  auto mask = upper ? GreaterEqual(index_row, index_col)
                    : LessEqual(index_row, index_col);
  auto mask_mat = Reshape(mask, {m, m});
  auto mask_full = BroadcastTo(mask_mat, x->shape);
  auto zeros = FillConstant(x->shape, 0.0f, "zeros", common::Type2Str(x->type));
  auto out = Select(mask_full, cholesky_out, zeros);
  return out;
}

Variable NetBuilder::TriangularSolve(const Variable& input1,
                                     const Variable& input2,
                                     bool left_side,
                                     bool upper,
                                     bool transpose_a,
                                     bool unit_diagonal) {
  // broadcast
  std::vector<Variable> inputs{input1, input2};
  {
    auto a_ndim = input1->shape.size();
    auto b_ndim = input2->shape.size();
    CHECK_GE(a_ndim, 2)
        << "The input matrix A shape size should >= 2! Please check again.";
    CHECK_GE(b_ndim, 2)
        << "The input matrix B shape size should >= 2! Please check again.";
    std::vector<int> input1_shape_cut(input1->shape.begin(),
                                      input1->shape.end() - 2);
    std::vector<int> input2_shape_cut(input2->shape.begin(),
                                      input2->shape.end() - 2);
    std::vector<int> common_shape;
    hlir::pe::GetBroadcastOutShape(
        input1_shape_cut, input2_shape_cut, &common_shape);

    // broadcast input1
    std::vector<int> input1_shape(common_shape.begin(), common_shape.end());
    input1_shape.push_back(input1->shape[a_ndim - 2]);
    input1_shape.push_back(input1->shape[a_ndim - 1]);
    inputs[0] = BroadcastTo(input1, input1_shape);

    // broadcast input2
    std::vector<int> input2_shape(common_shape.begin(), common_shape.end());
    input2_shape.push_back(input2->shape[b_ndim - 2]);
    input2_shape.push_back(input2->shape[b_ndim - 1]);
    inputs[1] = BroadcastTo(input2, input2_shape);
  }

  return CustomInstr("triangular_solve",
                     inputs,
                     {{"left_side", left_side},
                      {"upper", upper},
                      {"transpose_a", transpose_a},
                      {"unit_diagonal", unit_diagonal}})
      .front();
}

std::vector<Variable> NetBuilder::TopK(const Variable& x,
                                       int k,
                                       int axis,
                                       bool largest) {
  return CustomInstr(
      "top_k", {x}, {{"k", k}, {"axis", axis}, {"largest", largest}});
}

}  // namespace frontend
}  // namespace cinn

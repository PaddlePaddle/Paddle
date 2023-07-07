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

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/frontend/paddle/cpp/desc_api.h"
#include "paddle/cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

enum class EltwiseType {
  kUnk = 0,
  kAdd,
  kDiv,
  kMul,
  kSub,
  kPow,
  kMod,
  kMax,
  kMin
};

template <EltwiseType Type>
std::string GetEltwiseTypeString();

#define EXPAND_ELTWISETYPE_STRING(Type, str)              \
  template <>                                             \
  std::string GetEltwiseTypeString<EltwiseType::Type>() { \
    return str;                                           \
  }

EXPAND_ELTWISETYPE_STRING(kAdd, " + ")
EXPAND_ELTWISETYPE_STRING(kDiv, " / ")
EXPAND_ELTWISETYPE_STRING(kMul, " * ")
EXPAND_ELTWISETYPE_STRING(kSub, " - ")
EXPAND_ELTWISETYPE_STRING(kPow, " pow ")
EXPAND_ELTWISETYPE_STRING(kMod, " % ")
EXPAND_ELTWISETYPE_STRING(kMax, " max ")
EXPAND_ELTWISETYPE_STRING(kMin, " min ")
#undef EXPAND_ELTWISETYPE_STRING

template <EltwiseType Type>
struct OpBuilder {};

#define ELTWISE_SPEC(enum_t, function)                                        \
  template <>                                                                 \
  struct OpBuilder<enum_t> {                                                  \
    constexpr static Variable (NetBuilder::*func)(const Variable&,            \
                                                  const Variable&,            \
                                                  int){&function}; /*NOLINT*/ \
  }
ELTWISE_SPEC(EltwiseType::kAdd, NetBuilder::Add);
ELTWISE_SPEC(EltwiseType::kDiv, NetBuilder::Divide);
ELTWISE_SPEC(EltwiseType::kMul, NetBuilder::Multiply);
ELTWISE_SPEC(EltwiseType::kSub, NetBuilder::Subtract);
ELTWISE_SPEC(EltwiseType::kPow, NetBuilder::Pow);
ELTWISE_SPEC(EltwiseType::kMod, NetBuilder::Mod);
ELTWISE_SPEC(EltwiseType::kMax, NetBuilder::Max);
ELTWISE_SPEC(EltwiseType::kMin, NetBuilder::Min);
#undef ELTWISE_SPEC

void AddOpMapper(const paddle::cpp::OpDesc& op_desc,
                 const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  VLOG(4) << out_name << " = " << x_name << " + " << y_name;

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Add(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

template <EltwiseType Type>
void ElementwiseOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  VLOG(5) << "Elementwise operator mapping type: " << static_cast<int>(Type);
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  VLOG(4) << out_name << " = " << x_name << GetEltwiseTypeString<Type>()
          << y_name << " at " << axis;

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);
  auto out = (ctx.Builder()->*OpBuilder<Type>::func)(x, y, axis);

  ctx.AddVar(out_name, out, true);
  ctx.AddVarModelToProgram(out_name, out->id, true);
}

void ElementwiseAddGradOpMapper(const paddle::cpp::OpDesc& op_desc,
                                const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Input(paddle::GradVarName("Out")).size(), 1UL);
  auto dout_name = op_desc.Input(paddle::GradVarName("Out")).front();

  std::string dx_name, dy_name;
  bool has_dx = op_desc.Output(paddle::GradVarName("X")).size() > 0UL;
  if (has_dx) {
    dx_name = op_desc.Output(paddle::GradVarName("X")).front();
  }
  bool has_dy = op_desc.Output(paddle::GradVarName("Y")).size() > 0UL;
  if (has_dy) {
    dy_name = op_desc.Output(paddle::GradVarName("Y")).front();
  }

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  VLOG(4) << "{X@GRAD=" << dx_name << ", Y@GRAD=" << dy_name
          << "}=elementwise_add_grad(X=" << x_name << ", Y=" << y_name
          << ", OUT@GRAD=" << dout_name << ", axis=" << axis << ")";

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);
  auto dout = ctx.GetVar(dout_name);
  auto outs = ctx.Builder()->ElementwiseAddGrad(dout, x, y, axis);
  CHECK_EQ(outs.size(), 2) << "elementwise_add_grad should return 2 variables";

  if (has_dx) {
    auto dx = outs.front();
    ctx.AddVar(dx_name, dx);
    ctx.AddVarModelToProgram(dx_name, dx->id, true);
  }
  if (has_dy) {
    auto dy = outs.back();
    ctx.AddVar(dy_name, dy);
    ctx.AddVarModelToProgram(dy_name, dy->id, true);
  }
}

void SumOpMapper(const paddle::cpp::OpDesc& op_desc,
                 const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    xs.emplace_back(ctx.GetVar(name));
  }

  VLOG(4) << out_name << " = " << cinn::utils::Join(x_names, " + ");

  auto out = ctx.Builder()->Sum(xs);

  ctx.AddVar(out_name, out, true);
  ctx.AddVarModelToProgram(out_name, out->id, true);
}

void CastOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("out_dtype"))
      << "The cast op should has [out_dtype] attribute!";
  auto dtype = utils::GetPaddleDtype(
      op_desc, "out_dtype", paddle::cpp::VarDescAPI::Type::FP32);
  CHECK(!dtype.empty()) << "The op \"cast\"'s attribute \"out_dtype\" should "
                           "not be unknown type! Please check.";

  VLOG(4) << out_name << " = cast(X:" << x_name << ", out_dtype=" << dtype
          << ")";

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Cast(x, dtype);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void PowOpMapper(const paddle::cpp::OpDesc& op_desc,
                 const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x = ctx.GetVar(x_name);
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  absl::optional<Variable> y;
  if (op_desc.HasInput("FactorTensor") &&
      !op_desc.Input("FactorTensor").empty()) {
    CHECK_EQ(op_desc.Input("FactorTensor").size(), 1UL);
    auto y_name = op_desc.Input("FactorTensor").front();
    y = ctx.GetVar(y_name);

  } else if (op_desc.HasAttr("factor")) {
    auto factor = utils::GetAttrOrDefault<float>(op_desc, "factor");
    y = ctx.Builder()->FillConstant(x->shape,
                                    factor,
                                    cinn::UniqName(x_name + "_factor"),
                                    common::Type2Str(x->type));
  } else {
    LOG(FATAL) << "Cannot found [FactorTensor] input or [factor] attribute in "
                  "paddle.pow! Please check.";
  }

  VLOG(4) << out_name << " = pow(" << x_name << ", " << y.value()->id << ")";
  CHECK_EQ(x->type, y.value()->type)
      << "The data type of pow's inputs should be equal, but here x:" << x->type
      << " != y:" << y.value()->type;

  auto out = ctx.Builder()->Pow(x, y.value());

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void FloorDivideOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);

  VLOG(4) << out_name << " = ⌊ " << x_name << " / " << y_name << " ⌋";
  CHECK_EQ(x->type, y->type) << "Type of input x and y must be the same.";
  CHECK(x->type.is_int()) << "Type of inputs must be int32 or int64.";

  auto out = ctx.Builder()->FloorDivide(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_elementwise) {
  using cinn::frontend::paddle_mappers::AddOpMapper;
  using cinn::frontend::paddle_mappers::CastOpMapper;
  using cinn::frontend::paddle_mappers::ElementwiseAddGradOpMapper;
  using cinn::frontend::paddle_mappers::ElementwiseOpMapper;
  using cinn::frontend::paddle_mappers::EltwiseType;
  using cinn::frontend::paddle_mappers::FloorDivideOpMapper;
  using cinn::frontend::paddle_mappers::PowOpMapper;
  using cinn::frontend::paddle_mappers::SumOpMapper;

  CINN_REGISTER_OP_MAPPER(add, AddOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_add,
                          ElementwiseOpMapper<EltwiseType::kAdd>)
  CINN_REGISTER_OP_MAPPER(elementwise_add_grad, ElementwiseAddGradOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_mul,
                          ElementwiseOpMapper<EltwiseType::kMul>)
  CINN_REGISTER_OP_MAPPER(elementwise_div,
                          ElementwiseOpMapper<EltwiseType::kDiv>)
  CINN_REGISTER_OP_MAPPER(elementwise_sub,
                          ElementwiseOpMapper<EltwiseType::kSub>)
  CINN_REGISTER_OP_MAPPER(elementwise_pow,
                          ElementwiseOpMapper<EltwiseType::kPow>)
  CINN_REGISTER_OP_MAPPER(elementwise_mod,
                          ElementwiseOpMapper<EltwiseType::kMod>)
  CINN_REGISTER_OP_MAPPER(elementwise_max,
                          ElementwiseOpMapper<EltwiseType::kMax>)
  CINN_REGISTER_OP_MAPPER(elementwise_min,
                          ElementwiseOpMapper<EltwiseType::kMin>)
  CINN_REGISTER_OP_MAPPER(sum, SumOpMapper)
  CINN_REGISTER_OP_MAPPER(cast, CastOpMapper)
  CINN_REGISTER_OP_MAPPER(pow, PowOpMapper)
  CINN_REGISTER_OP_MAPPER(
      grad_add,
      ElementwiseOpMapper<EltwiseType::kAdd>)  // special elementwise_add for
                                               // gradient accumulation
  CINN_REGISTER_OP_MAPPER(elementwise_floordiv, FloorDivideOpMapper)
  return true;
}

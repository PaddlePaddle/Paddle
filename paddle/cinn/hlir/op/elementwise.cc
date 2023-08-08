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

#include "paddle/cinn/hlir/pe/elementwise.h"

#include <iostream>

#include "absl/types/optional.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/utils/functional.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;
using PeFunc = std::function<std::vector<ir::Tensor>(
    const ir::Tensor &A, const std::string &out_name)>;

#define StrategyForUnary(op_name__, pe__)                                      \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(                               \
      const framework::NodeAttr &attrs,                                        \
      const std::vector<ir::Tensor> &inputs,                                   \
      const std::vector<Type> &out_type,                                       \
      const std::vector<std::vector<int>> &output_shapes,                      \
      const Target &target) {                                                  \
    return StrategyForElementwise(                                             \
        attrs, inputs, out_type, output_shapes, target, #op_name__, pe::pe__); \
  }

std::shared_ptr<OpStrategy> StrategyForElementwise(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    const PeFunc &pe_func) {
  framework::CINNCompute unary_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input argument of " << op_name
                             << " compute is empty! Please check.";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 1U)
            << "1 input tensor for " << op_name << " compute";
        CHECK_EQ(pack_args.size(), 2U);
        CHECK(pack_args[1].is_string());
        std::string tensor_name = pack_args[1].operator std::string();
        Expr A_expr = pack_args[0];
        CHECK(A_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();
        auto out = pe_func(A, tensor_name);
        auto stages = CreateStages({A});
        std::vector<CINNValue> res;
        for (auto &t : out) {
          stages->InsertLazily(t);
          res.push_back(CINNValue(t));
        }
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(unary_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy." + op_name + ".x86",
                    1);

  return strategy;
}

std::vector<shape_t> InferShapeForElementwise(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL);
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForElementwise(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<Type> InferDtypeForElementwiseBool(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  return {Bool()};
}

std::vector<std::vector<std::string>> InferLayoutForElementwise(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U)
      << "The input's layouts size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForScale(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  float scale = 1.f;
  float bias = 0.f;
  bool bias_after_scale = true;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "scale") {
      scale = absl::get<float>(iter.second);
    } else if (iter.first == "bias") {
      bias = absl::get<float>(iter.second);
    } else if (iter.first == "bias_after_scale") {
      bias_after_scale = absl::get<bool>(iter.second);
    }
  }
  framework::CINNCompute scale_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input arguments of scale compute is empty! Please check.";
        CINNValuePack pack_args = args[0];
        CHECK(!pack_args.empty())
            << "The input tensors of scale compute is empty! Please check.";
        Expr A_expr = pack_args[0];
        CHECK(A_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();
        ir::Tensor out;
        CHECK_EQ(pack_args.size(), 2);
        CHECK(pack_args[1].is_string());
        std::string tensor_name = pack_args[1].operator std::string();

        if (bias_after_scale) {
          out = Compute(
              A->shape,
              [=](const std::vector<Expr> &indice) {
                return ir::Cast::Make(A->type(), Expr(scale)) * A(indice) +
                       ir::Cast::Make(A->type(), Expr(bias));
              },
              tensor_name);
        } else {
          out = Compute(
              A->shape,
              [=](const std::vector<Expr> &indice) {
                return ir::Cast::Make(A->type(), Expr(scale)) *
                       (A(indice) + ir::Cast::Make(A->type(), Expr(bias)));
              },
              tensor_name);
        }
        auto stages = CreateStages({out});
        *ret = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.scale.x86",
                    1);

  return strategy;
}

Expr GetScalarExpr(const framework::NodeAttr::attr_t &attr) {
  Expr scalar;
  struct Visitor {
    Expr &scalar_;
    explicit Visitor(Expr &scalar) : scalar_(scalar) {}
    void operator()(float v) { scalar_ = Expr(v); }
    void operator()(double v) { scalar_ = Expr(v); }
    void operator()(int32_t v) { scalar_ = Expr(v); }
    void operator()(int64_t v) { scalar_ = Expr(v); }
    void operator()(bool v) { scalar_ = Expr(v); }
    void operator()(const std::string &v) { scalar_ = Expr(v); }
    void operator()(const std::vector<int> &) {
      LOG(FATAL) << "wrong type std::vector<int>";
    }
    void operator()(const std::vector<int64_t> &) {
      LOG(FATAL) << "wrong type std::vector<int64_t>";
    }
    void operator()(const std::vector<float> &) {
      LOG(FATAL) << "wrong type std::vector<float>";
    }
    void operator()(const std::vector<double> &) {
      LOG(FATAL) << "wrong type std::vector<double>";
    }
    void operator()(const std::vector<bool> &) {
      LOG(FATAL) << "wrong type std::vector<bool>";
    }
    void operator()(const std::vector<std::string> &) {
      LOG(FATAL) << "wrong type std::vector<std::string>";
    }
  };
  absl::visit(Visitor{scalar}, attr);
  return scalar;
}

std::shared_ptr<OpStrategy> StrategyForConstScalar(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute const_scalar_compute([=](lang::Args args,
                                                  lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of const_float compute is empty! Please check.";
    auto scalar = GetScalarExpr(attrs.attr_store.at("value"));
    auto scalar_type = out_type.at(0);
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 1U);
    CHECK(pack_args[0].is_string());
    std::string tensor_name = pack_args[0].operator std::string();

    auto out = lang::Compute(
        {Expr(1)},
        [=](const std::vector<Expr> &indice) {
          auto res = (scalar_type == scalar->type())
                         ? scalar
                         : ir::Cast::Make(scalar_type, scalar);
          return res;
        },
        tensor_name);
    CHECK(out.defined()) << "can't create const scalar with the given type "
                         << out_type[0];
    auto stages = CreateStages({out});
    *ret = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(const_scalar_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.const_scalar.x86",
                    1);

  return strategy;
}

std::vector<shape_t> InferShapeForConstScalar(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  return {{1}};
}

std::vector<Type> InferDtypeForConstScalar(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  Type out_type;
  if (attrs.find("dtype") != attrs.end()) {
    auto dtype_str = absl::get<std::string>(attrs.at("dtype"));
    if (!dtype_str.empty()) {
      out_type = common::Str2Type(dtype_str);
    }
  } else {
    auto scalar = GetScalarExpr(attrs.at("value"));
    out_type = scalar->type();
  }
  VLOG(3) << "scalar type: " << out_type;
  return {out_type};
}

std::vector<std::vector<std::string>> InferLayoutForConstScalar(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  return {{"C"}, input_layouts};
}

std::shared_ptr<OpStrategy> StrategyForSum(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  LOG(FATAL) << "The operator will be decomposed into several primitive "
                "operators. Please Use Decomposer Program Pass.";
}

std::vector<shape_t> InferShapeForSum(const std::vector<shape_t> &inputs_shape,
                                      const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty()) << "At least 1 input tensor for sum operator.";
  auto shape = inputs_shape[0];
  for (size_t i = 1; i < inputs_shape.size(); ++i) {
    if (inputs_shape[i] != shape) {
      LOG(FATAL) << "The input shapes must be the same. But received: the i-th("
                 << i << ") input shape is "
                 << utils::Join(inputs_shape[i], ",")
                 << " and the first input shape is " << utils::Join(shape, ",");
    }
  }
  std::vector<shape_t> out_shape{shape};

  return out_shape;
}

std::vector<Type> InferDtypeForSum(const std::vector<Type> &inputs_type,
                                   const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "At least 1 input tensor for sum operator.";
  auto type = inputs_type[0];
  for (size_t i = 1; i < inputs_type.size(); ++i) {
    if (inputs_type[i] != type) {
      LOG(FATAL) << "The input types must be the same. But received: the i-th("
                 << i << ") input type is " << inputs_type[i]
                 << " and the first input type is " << type;
    }
  }
  std::vector<Type> res{type};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForFillConstant(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute fill_constant_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input argument of fill_constant compute "
                                "is empty! Please check.";
        bool force_cpu = false;
        CHECK(attrs.attr_store.count("shape"));
        auto shape = absl::get<std::vector<int>>(attrs.attr_store.at("shape"));
        CHECK(attrs.attr_store.count("value"));
        auto value = GetScalarExpr(attrs.attr_store.at("value"));
        CHECK(attrs.attr_store.count("force_cpu"));
        force_cpu = absl::get<bool>(attrs.attr_store.at("force_cpu"));

        if (force_cpu && target != common::DefaultHostTarget()) {
          LOG(WARNING) << "The attribute \"force_cpu\" of \"fill_constant\" "
                          "not supported in CINN! The \"fill_constant\"'s "
                          "output tensor will placed on "
                       << target;
        }

        CINNValuePack arg_pack = args[0];
        CHECK_EQ(arg_pack.size(), 1U);
        CHECK(arg_pack[0].is_string());
        std::string tensor_name = arg_pack[0].operator std::string();
        CHECK(!shape.empty()) << "shape attr is empty!";
        auto shape_exprs = ToCinnExprs(shape);
        auto out = lang::Compute(
            shape_exprs,
            [=](const std::vector<Expr> &indice) {
              return ir::Cast::Make(out_type[0], value);
            },
            tensor_name);
        CHECK(out.defined())
            << "can't create fill_constant with the given type " << out_type[0];
        auto stages = CreateStages({out});
        *ret = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(fill_constant_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.fill_constant.x86",
                    1);

  return strategy;
}

std::vector<shape_t> InferShapeForFillConstant(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK(attrs.count("shape"));
  auto shape = absl::get<std::vector<int>>(attrs.at("shape"));
  return {shape};
}

std::vector<Type> InferDtypeForFillConstant(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  common::Type out_type;
  CHECK(attrs.count("value"));
  if (attrs.find("dtype") != attrs.end()) {
    // attribute [dtype] are given
    auto dtype_str = absl::get<std::string>(attrs.at("dtype"));
    out_type = common::Str2Type(dtype_str);
    VLOG(3) << "FillConstant output dtype (from [dtype]): " << dtype_str;
  } else {
    // attribute [dtype] no given, inferred by value's type
    auto scalar = GetScalarExpr(attrs.at("value"));
    out_type = scalar->type();
    VLOG(3) << "FillConstant scalar type (from [value]): "
            << common::Type2Str(out_type);
  }
  return {out_type};
}

std::vector<std::vector<std::string>> InferLayoutForFillConstant(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  return {{""}, input_layouts};
}

#define EXPAND_ATTR_TYPE(MACRO) \
  MACRO(bool)                   \
  MACRO(int)                    \
  MACRO(int64_t)                \
  MACRO(double)                 \
  MACRO(float)

std::shared_ptr<OpStrategy> StrategyForAssignValue(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute assign_value_compute([=](lang::Args args,
                                                  lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of assign_value compute is empty! Please check.";
    CHECK(attrs.attr_store.count("values"))
        << "assign_value should set attribute [values]! Please check.";
    const auto &value = attrs.attr_store.at("values");

    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 1U);
    CHECK(arg_pack[0].is_string());
    std::string tensor_name = arg_pack[0].operator std::string();

    absl::optional<ir::Tensor> out;
#define EXPAND_VALUE_TO_TENSOR(TYPE)                                          \
  else if (absl::get_if<TYPE>(&value)) { /*NOLINT*/                           \
    out = pe::AssignValue(                                                    \
        std::vector<TYPE>{absl::get<TYPE>(value)}, out_type[0], tensor_name); \
  }                                                                           \
  else if (absl::get_if<std::vector<TYPE>>(&value)) { /*NOLINT*/              \
    out = pe::AssignValue(                                                    \
        absl::get<std::vector<TYPE>>(value), out_type[0], tensor_name);       \
  }

    if (false) {  // NOLINT
    }
    EXPAND_ATTR_TYPE(EXPAND_VALUE_TO_TENSOR)
    else {  // NOLINT
      LOG(FATAL) << "Assign value not support the type " << out_type[0];
    }
#undef EXPAND_VALUE_TO_TENSOR

    CHECK(out && out.value().defined())
        << "can't create assign_value with the given type " << out_type[0];

    auto stages = CreateStages({out.value()});
    *ret =
        CINNValuePack{{CINNValue(Expr(out.value().get())), CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(assign_value_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.assign_value.x86",
                    1);

  return strategy;
}

std::vector<shape_t> InferShapeForAssignValue(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK(attrs.count("values"))
      << "assign_value should set attribute [values]! Please check.";
  const auto &value = attrs.at("values");

  shape_t shape;
#define EXPAND_ATTR_TO_GET_SHAPE(TYPE)                              \
  else if (absl::get_if<TYPE>(&value)) { /*NOLINT*/                 \
    shape.emplace_back(1);                                          \
  }                                                                 \
  else if (absl::get_if<std::vector<TYPE>>(&value)) { /*NOLINT*/    \
    shape.emplace_back(absl::get<std::vector<TYPE>>(value).size()); \
  }

  if (false) {  // NOLINT
  }
  EXPAND_ATTR_TYPE(EXPAND_ATTR_TO_GET_SHAPE)
  else {  // NOLINT
    LOG(FATAL) << "assign_value not support the type!";
  }
#undef EXPAND_ATTR_TO_GET_SHAPE

  VLOG(3) << "The output shape of assign_value is ["
          << cinn::utils::Join(shape, ", ") << "]";

  return {shape};
}

std::vector<Type> InferDtypeForAssignValue(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  Type out_type;
  if (attrs.find("dtype") != attrs.end()) {
    // attribute [dtype] are given
    auto dtype_str = absl::get<std::string>(attrs.at("dtype"));
    if (!dtype_str.empty()) {
      // if the [dtype] is not empty, output as the given type
      out_type = common::Str2Type(dtype_str);
    }
  }

  // attribute [dtype] not given or is empty
  if (out_type.is_unk()) {
    // infer from [values]'s dtype
    CHECK(attrs.count("values"))
        << "assign_value should set attribute [values]! Please check.";
    const auto &value = attrs.at("values");

#define EXPAND_ATTR_TO_GET_DTYPE(TYPE)                           \
  else if (absl::get_if<TYPE>(&value)) { /*NOLINT*/              \
    out_type = common::type_of<TYPE>();                          \
  }                                                              \
  else if (absl::get_if<std::vector<TYPE>>(&value)) { /*NOLINT*/ \
    out_type = common::type_of<TYPE>();                          \
  }

    if (false) {  // NOLINT
    }
    EXPAND_ATTR_TYPE(EXPAND_ATTR_TO_GET_DTYPE)
    else {  // NOLINT
      LOG(FATAL) << "assign_value not support the type!";
    }
#undef EXPAND_ATTR_TO_GET_DTYPE
  }

  VLOG(3) << "The data type of assign_value is " << out_type;

  return {out_type};
}

std::vector<std::vector<std::string>> InferLayoutForAssignValue(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  return {{""}, input_layouts};
}

#undef EXPAND_ATTR_TYPE

StrategyForUnary(exp, Exp);
StrategyForUnary(erf, Erf);
StrategyForUnary(sqrt, Sqrt);
StrategyForUnary(log, Log);
StrategyForUnary(floor, Floor);
StrategyForUnary(ceil, Ceil);
StrategyForUnary(round, Round);
StrategyForUnary(tanh, Tanh);
StrategyForUnary(log2, Log2);
StrategyForUnary(log10, Log10);
StrategyForUnary(trunc, Trunc);
StrategyForUnary(cos, Cos);
StrategyForUnary(cosh, Cosh);
StrategyForUnary(tan, Tan);
StrategyForUnary(sin, Sin);
StrategyForUnary(sinh, Sinh);
StrategyForUnary(acos, Acos);
StrategyForUnary(acosh, Acosh);
StrategyForUnary(asin, Asin);
StrategyForUnary(asinh, Asinh);
StrategyForUnary(atan, Atan);
StrategyForUnary(atanh, Atanh);

StrategyForUnary(isnan, IsNan);
StrategyForUnary(isfinite, IsFinite);
StrategyForUnary(isinf, IsInf);
StrategyForUnary(bitwise_not, BitwiseNot);

StrategyForUnary(negative, Negative);
StrategyForUnary(identity, Identity);
StrategyForUnary(logical_not, LogicalNot);
StrategyForUnary(sign, Sign);
StrategyForUnary(abs, Abs);
StrategyForUnary(rsqrt, Rsqrt);
StrategyForUnary(sigmoid, Sigmoid);
StrategyForUnary(cbrt, Cbrt);
StrategyForUnary(clz, Clz);
StrategyForUnary(popc, Popc);

#undef StrategyForUnary

std::shared_ptr<framework::OpStrategy> StrategyForSqueeze(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  const std::vector<int> &axes =
      attrs.attr_store.count("axes")
          ? absl::get<std::vector<int>>(attrs.attr_store.at("axes"))
          : std::vector<int>{};

  framework::CINNCompute squeeze_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input arguments of Squeeze compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U)
        << "at least 1 input tensors for Squeeze compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto stages = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    CHECK_EQ(pack_args.size(), 2U);
    std::string tensor_name = pack_args[1].operator std::string();

    ir::Tensor out = pe::Squeeze(tensor_A, axes, tensor_name);
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty())
        << "Output type of Squeeze is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(squeeze_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.squeeze.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForSqueeze(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U);
  const std::vector<int> &axes =
      attrs.count("axes") ? absl::get<std::vector<int>>(attrs.at("axes"))
                          : std::vector<int>{};
  VLOG(4) << "The [axis] value used in Squeeze: "
          << cinn::utils::Join(axes, ",");

  const auto &posi_axes = utils::GetPositiveAxes(axes, inputs_shape[0].size());
  std::vector<int> output_shape;
  if (posi_axes.size()) {
    for (int idx = 0; idx < inputs_shape[0].size(); ++idx) {
      // if can't find idx in axis
      if (std::find(posi_axes.begin(), posi_axes.end(), idx) ==
          posi_axes.end()) {
        output_shape.push_back(inputs_shape[0][idx]);
      } else {
        CHECK_EQ(inputs_shape[0][idx], 1);
      }
    }
  } else {
    for (int idx = 0; idx < inputs_shape[0].size(); ++idx) {
      if (inputs_shape[0][idx] != 1) {
        output_shape.push_back(inputs_shape[0][idx]);
      }
    }
  }

  VLOG(4) << "The output calculated in Squeeze: "
          << cinn::utils::Join(output_shape, ", ");
  return {output_shape};
}

std::shared_ptr<OpStrategy> StrategyForExpandDims(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  const std::vector<int> &axes =
      attrs.attr_store.count("axes")
          ? absl::get<std::vector<int>>(attrs.attr_store.at("axes"))
          : std::vector<int>{};

  framework::CINNCompute expand_dims_compute{[=](lang::Args args,
                                                 lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input args are empty! Please check again.";
    CINNValuePack input_args = args[0];
    int input_size = input_args.size();
    CHECK_GE(input_size, 1U)
        << "Require 1 input tensors for expand_dims compute.";
    Expr x = input_args[0];
    CHECK(x.as_tensor());

    CHECK_EQ(input_args.size(), 2U);
    CHECK(input_args[1].is_string());
    std::string tensor_name = input_args[1].operator std::string();

    auto out =
        pe::ExpandDims(x.as_tensor_ref(), axes, output_shapes[0], tensor_name);
    auto stages = CreateStages({x.as_tensor_ref()});
    stages->InsertLazily(out);
    std::vector<CINNValue> res{CINNValue(out), CINNValue(stages)};
    *ret = CINNValuePack{res};
  }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(expand_dims_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.expand_dims.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForExpandDims(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty())
      << "At least 1 input tensor for expand_dims operator.";

  CHECK_EQ(inputs_shape.size(), 1U);
  const std::vector<int> &axes =
      attrs.count("axes") ? absl::get<std::vector<int>>(attrs.at("axes"))
                          : std::vector<int>{};
  VLOG(4) << "The [axes] value used in ExpandDims: "
          << cinn::utils::Join(axes, ",");

  std::vector<int> out_shape(inputs_shape[0].size() + axes.size(), 1);
  const auto &posi_axes = utils::GetPositiveAxes(axes, out_shape.size());

  int shape_pos = 0, axes_pos = 0;
  for (int i = 0; i < out_shape.size(); ++i) {
    if (axes_pos < posi_axes.size() && posi_axes[axes_pos] == i) {
      out_shape[i] = 1;
      ++axes_pos;
    } else if (shape_pos < inputs_shape[0].size()) {
      out_shape[i] = inputs_shape[0][shape_pos];
      ++shape_pos;
    }
  }

  VLOG(4) << "The output calculated in ExpandDims: "
          << cinn::utils::Join(out_shape, ", ");
  return {out_shape};
}

std::shared_ptr<OpStrategy> StrategyForReshape(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute reshape_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input arguments of Reshape compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U)
        << "at least 1 input tensors for Reshape compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto attr_store = attrs.attr_store;
    CHECK(attr_store.count("shape")) << "find no attr of shape";
    std::vector<int> new_shape =
        absl::get<std::vector<int>>(attr_store.at("shape"));
    auto tensor_A = A.as_tensor_ref();
    auto stages = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    CHECK_EQ(pack_args.size(), 2);
    CHECK(pack_args[1].is_string());
    std::string tensor_name = pack_args[1].operator std::string();

    ir::Tensor out = pe::Reshape(tensor_A, output_shapes[0], tensor_name);
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty())
        << "Output type of Reshape is empty! Please check.\n";
    res.push_back(CINNValue(stages));

    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(reshape_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.reshape.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForReshape(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U)
      << "The input's shape size should be 1! Please check again.";
  std::vector<int> output_shape;
  for (auto &iter : attrs) {
    if (iter.first == "shape") {
      output_shape = absl::get<std::vector<int>>(iter.second);
      break;
    }
  }
  int tensor_size = 1;
  for (auto i : inputs_shape[0]) {
    tensor_size *= i;
  }
  int flag_index = -1;
  for (int i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] > 0) {
      CHECK_EQ(tensor_size % output_shape[i], 0)
          << "Incompatible input shape and output shape in op reshape: "
          << tensor_size << ", " << output_shape[i];
      tensor_size /= output_shape[i];
    } else if (output_shape[i] == 0) {
      CHECK_LT(i, inputs_shape[0].size())
          << "In op reshape, when attribute shape[i] == 0, shape[i] = "
             "input_shape[i]. But now the size of input_shape "
             "<= i, which is incompatible. Please check!";
      output_shape[i] = inputs_shape[0][i];
      CHECK_EQ(tensor_size % output_shape[i], 0)
          << "Incompatible input shape and output shape in op reshape: "
          << tensor_size << ", " << output_shape[i];
      tensor_size /= output_shape[i];
    } else if (output_shape[i] == -1 && flag_index == -1) {
      flag_index = i;
    } else if (output_shape[i] == -1) {
      LOG(FATAL) << "More than one -1 in output_shape of op reshape.";
    } else {
      LOG(FATAL) << "Unsupported output_shape " << output_shape[i];
    }
  }
  if (flag_index >= 0) output_shape[flag_index] = tensor_size;
  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForCast(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute cast_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input arguments of Cast compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 1U)
            << "at least 1 input tensors for Cast compute\n";
        Expr A = pack_args[0];
        CHECK(A.as_tensor());
        CHECK(!output_shapes.empty());
        auto tensor_A = A.as_tensor_ref();
        auto stages = CreateStages({tensor_A});
        VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
                << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
        CHECK_EQ(pack_args.size(), 2U);
        std::string tensor_name = pack_args[1].operator std::string();
        ir::Tensor out = pe::Cast(tensor_A, out_type[0], tensor_name);
        std::vector<CINNValue> res;
        stages->InsertLazily(out);
        res.push_back(CINNValue(out));
        CHECK(!out_type.empty())
            << "Output type of Cast is empty! Please check.\n";
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cast_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.reshape.x86",
                    1);
  return strategy;
}

std::vector<Type> InferDtypeForCast(const std::vector<Type> &inputs_type,
                                    const framework::AttrMapType &attrs) {
  CHECK(attrs.count("dtype"));
  return {common::Str2Type(absl::get<std::string>(attrs.at("dtype")))};
}

std::shared_ptr<framework::OpStrategy> StrategyForArange(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  CHECK(attr_store.count("start"));
  CHECK(attr_store.count("stop"));
  CHECK(attr_store.count("step"));
  CHECK(attr_store.count("dtype"));

  auto start = absl::get<float>(attr_store.at("start"));
  auto stop = absl::get<float>(attr_store.at("stop"));
  auto step = absl::get<float>(attr_store.at("step"));
  auto dtype = common::Str2Type(absl::get<std::string>(attr_store.at("dtype")));

  framework::CINNCompute arange_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input argument of arange compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];

        CHECK_EQ(pack_args.size(), 1U);
        std::string tensor_name = pack_args[0].operator std::string();

        auto out = pe::Arange(start, stop, step, dtype, tensor_name);
        std::vector<common::CINNValue> res;
        auto stages = CreateStages({out});
        res.push_back(common::CINNValue(out));
        res.push_back(common::CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(arange_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.reshape.x86",
                    1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForArange(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK(attrs.count("start"));
  CHECK(attrs.count("stop"));
  CHECK(attrs.count("step"));
  float start = absl::get<float>(attrs.at("start"));
  float stop = absl::get<float>(attrs.at("stop"));
  float step = absl::get<float>(attrs.at("step"));
  CHECK_NE(step, 0.0f) << "The value of step can't be 0!";

  int num = static_cast<int>(std::ceil((stop - start) / step));
  CHECK(num) << "Invalid arange parameters, start = " << start
             << ", stop = " << stop << ", step = " << step
             << ", cause num_elem = " << num << " which is negative.";
  return {{num}};
}

std::vector<Type> InferDtypeForArange(const std::vector<Type> &inputs_type,
                                      const framework::AttrMapType &attrs) {
  CHECK(attrs.count("dtype"));
  return {common::Str2Type(absl::get<std::string>(attrs.at("dtype")))};
}

std::vector<Type> InferDtypeForLogicalNot(const std::vector<Type> &inputs_type,
                                          const framework::AttrMapType &attrs) {
  return {common::Bool()};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(elementwise_ops) {
#define CINN_REGISTER_UNARY(op__, op_stragegy__)                           \
  CINN_REGISTER_OP(op__)                                                   \
      .describe(#op__ " function")                                         \
      .set_num_inputs(1)                                                   \
      .set_num_outputs(1)                                                  \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                  \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)      \
      .set_attr("infershape",                                              \
                MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))  \
      .set_attr("inferdtype",                                              \
                MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))  \
      .set_attr("inferlayout",                                             \
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise)) \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                     \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise) \
      .set_support_level(4);

  CINN_REGISTER_UNARY(exp, Exp);
  CINN_REGISTER_UNARY(erf, Erf);
  CINN_REGISTER_UNARY(sqrt, Sqrt);
  CINN_REGISTER_UNARY(log, Log);
  CINN_REGISTER_UNARY(floor, Floor);
  CINN_REGISTER_UNARY(ceil, Ceil);
  CINN_REGISTER_UNARY(round, Round);
  CINN_REGISTER_UNARY(tanh, Tanh);
  CINN_REGISTER_UNARY(log2, Log2);
  CINN_REGISTER_UNARY(log10, Log10);
  CINN_REGISTER_UNARY(trunc, Trunc);
  CINN_REGISTER_UNARY(cos, Cos);
  CINN_REGISTER_UNARY(cosh, Cosh);
  CINN_REGISTER_UNARY(tan, Tan);
  CINN_REGISTER_UNARY(sin, Sin);
  CINN_REGISTER_UNARY(sinh, Sinh);
  CINN_REGISTER_UNARY(acos, Acos);
  CINN_REGISTER_UNARY(acosh, Acosh);
  CINN_REGISTER_UNARY(asin, Asin);
  CINN_REGISTER_UNARY(asinh, Asinh);
  CINN_REGISTER_UNARY(atan, Atan);
  CINN_REGISTER_UNARY(atanh, Atanh);
  CINN_REGISTER_UNARY(bitwise_not, BitwiseNot)

  CINN_REGISTER_UNARY(negative, Negative)
  CINN_REGISTER_UNARY(identity, Identity)
  CINN_REGISTER_UNARY(sign, Sign)
  CINN_REGISTER_UNARY(abs, Abs)
  CINN_REGISTER_UNARY(rsqrt, Rsqrt)
  CINN_REGISTER_UNARY(sigmoid, Sigmoid)
  CINN_REGISTER_UNARY(cbrt, Cbrt);
  CINN_REGISTER_UNARY(clz, Clz);
  CINN_REGISTER_UNARY(popc, Popc);

#undef CINN_REGISTER_UNARY

#define CINN_REGISTER_COMPARE(op__, op_stragegy__)                            \
  CINN_REGISTER_OP(op__)                                                      \
      .describe(#op__ " function")                                            \
      .set_num_inputs(1)                                                      \
      .set_num_outputs(1)                                                     \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                     \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)         \
      .set_attr("infershape",                                                 \
                MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))     \
      .set_attr("inferdtype",                                                 \
                MakeOpFunction(cinn::hlir::op::InferDtypeForElementwiseBool)) \
      .set_attr("inferlayout",                                                \
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))    \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                        \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)    \
      .set_support_level(4);

  CINN_REGISTER_COMPARE(isnan, IsNan)
  CINN_REGISTER_COMPARE(isfinite, IsFinite)
  CINN_REGISTER_COMPARE(isinf, IsInf)

#undef CINN_REGISTER_COMPARE

  CINN_REGISTER_OP(scale)
      .describe("Putting scale and bias to the input Tensor")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForScale)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(const_scalar)
      .describe("create const scalar with the given value")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForConstScalar)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForConstScalar))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForConstScalar))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForConstScalar))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(sum)
      .describe("Sum the input tensors.")
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSum)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForSum))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForSum))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  CINN_REGISTER_OP(fill_constant)
      .describe("create tensor with the given value, type and shape")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForFillConstant)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForFillConstant))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForFillConstant))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForFillConstant))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(assign_value)
      .describe("create tensor with the given value, type and shape")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForAssignValue)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForAssignValue))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForAssignValue))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForAssignValue))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(squeeze)
      .describe("The operator is used to squeeze input tensor's dims")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSqueeze)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForSqueeze))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(expand_dims)
      .describe("This operator is used to expand input tensor's dims.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForExpandDims)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForExpandDims))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(reshape)
      .describe("This operator is used to reshape input tensor X.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForReshape)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForReshape))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(cast)
      .describe("This operator is used to cast input tensor's type to target.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForCast)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForCast))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(arange)
      .describe("Returns evenly spaced values within a given interval.")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForArange)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForArange))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForArange))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(gelu)
      .describe("The implement of gelu.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForElementwise))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  CINN_REGISTER_OP(logical_not)
      .describe("Logical not function")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForLogicalNot)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForLogicalNot))
      .set_attr("inferlayout",
                MakeOpFunction(cinn::hlir::op::InferLayoutForElementwise))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}

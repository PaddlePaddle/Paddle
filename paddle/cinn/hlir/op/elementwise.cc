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
#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/dialect/operator/ir/symbol_bindings.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/core/enforce.h"

namespace cinn {
namespace hlir {
namespace op {
using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
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
  }                                                                            \
  std::shared_ptr<OpStrategy> StrategyFor##pe__##Symbolic(                     \
      const framework::NodeAttr &attrs,                                        \
      const std::vector<ir::Tensor> &inputs,                                   \
      const std::vector<Type> &out_type,                                       \
      const std::vector<std::vector<ir::Dim>> &output_shapes,                  \
      const Target &target) {                                                  \
    return StrategyForElementwiseSymbolic(                                     \
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
  framework::CINNCompute unary_compute([=](lang::Args args,
                                           lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument("The input argument of %s compute is "
                                          "empty! Please check.",
                                          op_name));
    CINNValuePack pack_args = args[0];

    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater "
                          "than or equal to 1, but got %d.",
                          pack_args.size()));

    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("the size of pack_args should be"
                                          "equal to 2, but got %d.",
                                          pack_args.size()));
    PADDLE_ENFORCE_EQ(pack_args[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "the type of pack_args[1] should be string!"
                          "Please check."));
    std::string tensor_name = pack_args[1].operator std::string();
    Expr A_expr = pack_args[0];
    PADDLE_ENFORCE(
        A_expr.as_tensor(),
        ::common::errors::InvalidArgument("The pack_args[0] should be tensor!"
                                          "Please check."));
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out = pe_func(A, tensor_name);
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(unary_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy." + op_name + ".x86",
                    1);

  return strategy;
}
std::shared_ptr<OpStrategy> StrategyForElementwiseSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    const PeFunc &pe_func) {
  framework::CINNCompute unary_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE_EQ(!args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The input argument of %s compute is empty!"
                              "Please check.",
                              op_name));
        CINNValuePack pack_args = args[0];
        PADDLE_ENFORCE_GE(pack_args.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "the size of pack_args should be greater "
                              "than or equal to 1, but got %d.",
                              pack_args.size()));

        PADDLE_ENFORCE_EQ(
            pack_args.size(),
            2U,
            ::common::errors::InvalidArgument("the size of pack_args should be "
                                              "equal to 2, but got %d.",
                                              pack_args.size()));
        PADDLE_ENFORCE_EQ(pack_args[1].is_string(),
                          true,
                          ::common::errors::InvalidArgument(
                              "the type of pack_args[1] should be string!"
                              "Please check."));
        std::string tensor_name = pack_args[1].operator std::string();
        Expr A_expr = pack_args[0];
        PADDLE_ENFORCE(
            A_expr.as_tensor(),
            ::common::errors::InvalidArgument("The pack_args[0] should be "
                                              "tensor! Please check."));
        ir::Tensor A = A_expr.as_tensor_ref();
        auto out = pe_func(A, tensor_name);
        std::vector<CINNValue> res;
        for (auto &t : out) {
          res.push_back(CINNValue(t));
        }
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      unary_compute, lang::PackedFunc(), "strategy." + op_name + ".x86", 1);

  return strategy;
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
  framework::CINNCompute scale_compute([=](lang::Args args,
                                           lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument("The input argument of scale compute "
                                          "is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        !pack_args.empty(),
        true,
        ::common::errors::InvalidArgument("The input tensors of scale compute "
                                          "is empty! Please check."));
    Expr A_expr = pack_args[0];
    PADDLE_ENFORCE(A_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] is not a tensor! Please check."));
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor out;
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 2, but got %d.",
                                          pack_args.size()));
    PADDLE_ENFORCE_EQ(pack_args[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "the type of pack_args[1] should be string! "
                          "Please check."));
    std::string tensor_name = pack_args[1].operator std::string();

    // Paddle upscale float16 or bfloat16 compute to float32,
    // we made CINN consistent with this behavior of Paddle
    bool should_upscale_fp32 =
        A->type() == cinn::common::F16() || A->type() == cinn::common::BF16();

    out = Compute(
        A->shape,
        [=](const std::vector<Expr> &indice) {
          Expr cast_scale = should_upscale_fp32
                                ? Expr(scale)
                                : ir::Cast::Make(A->type(), Expr(scale));
          Expr cast_bias = should_upscale_fp32
                               ? Expr(bias)
                               : ir::Cast::Make(A->type(), Expr(bias));
          Expr cast_A_indice =
              should_upscale_fp32
                  ? ir::Cast::Make(cinn::common::F32(), A(indice))
                  : A(indice);
          Expr add_result = bias_after_scale
                                ? cast_scale * cast_A_indice + cast_bias
                                : cast_scale * (cast_A_indice + cast_bias);
          return should_upscale_fp32 ? ir::Cast::Make(A->type(), add_result)
                                     : add_result;
        },
        tensor_name);

    *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.scale.x86",
                    1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForScaleSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
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
        PADDLE_ENFORCE_EQ(!args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The input argument of scale compute is empty! "
                              "Please check."));
        CINNValuePack pack_args = args[0];
        PADDLE_ENFORCE_EQ(!pack_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The input tensors of scale compute is empty! "
                              "Please check."));
        Expr A_expr = pack_args[0];
        PADDLE_ENFORCE(A_expr.as_tensor(),
                       ::common::errors::InvalidArgument(
                           "The pack_args[0] is not a tensor! Please check."));
        ir::Tensor A = A_expr.as_tensor_ref();
        ir::Tensor out;
        PADDLE_ENFORCE_EQ(
            pack_args.size(),
            2,
            ::common::errors::InvalidArgument("the size of pack_args should be "
                                              "equal to 2, but got %d.",
                                              pack_args.size()));
        PADDLE_ENFORCE_EQ(pack_args[1].is_string(),
                          true,
                          ::common::errors::InvalidArgument(
                              "the type of pack_args[1] should be string! "
                              "Please check."));
        std::string tensor_name = pack_args[1].operator std::string();

        // Paddle upscale float16 or bfloat16 compute to float32,
        // we made CINN consistent with this behavior of Paddle
        bool should_upscale_fp32 = A->type() == cinn::common::F16() ||
                                   A->type() == cinn::common::BF16();

        out = Compute(
            A->shape,
            [=](const std::vector<Expr> &indice) {
              Expr cast_A_indice =
                  should_upscale_fp32
                      ? ir::Cast::Make(cinn::common::F32(), A(indice))
                      : A(indice);

              Expr cast_scale = should_upscale_fp32
                                    ? Expr(scale)
                                    : ir::Cast::Make(A->type(), Expr(scale));
              Expr cast_bias = should_upscale_fp32
                                   ? Expr(bias)
                                   : ir::Cast::Make(A->type(), Expr(bias));
              Expr add_result;
              if (scale == 1.0f) {
                if (bias == 0.0f) {
                  add_result = cast_A_indice;
                } else {
                  add_result = cast_A_indice + cast_bias;
                }
              } else {
                if (bias == 0.0f) {
                  add_result = cast_scale * cast_A_indice;
                } else {
                  add_result = bias_after_scale
                                   ? cast_scale * cast_A_indice + cast_bias
                                   : cast_scale * (cast_A_indice + cast_bias);
                }
              }
              return should_upscale_fp32 ? ir::Cast::Make(A->type(), add_result)
                                         : add_result;
            },
            tensor_name);

        *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute, lang::PackedFunc(), "strategy.scale.x86", 1);

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
      PADDLE_THROW(
          ::common::errors::InvalidArgument("wrong type std::vector<int>"));
    }
    void operator()(const std::vector<int64_t> &) {
      PADDLE_THROW(
          ::common::errors::InvalidArgument("wrong type std::vector<int64_t>"));
    }
    void operator()(const std::vector<float> &) {
      PADDLE_THROW(
          ::common::errors::InvalidArgument("wrong type std::vector<float>"));
    }
    void operator()(const std::vector<double> &) {
      PADDLE_THROW(
          ::common::errors::InvalidArgument("wrong type std::vector<double>"));
    }
    void operator()(const std::vector<bool> &) {
      PADDLE_THROW(
          ::common::errors::InvalidArgument("wrong type std::vector<bool>"));
    }
    void operator()(const std::vector<std::string> &) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "wrong type std::vector<std::string>"));
    }
    void operator()(const std::vector<symbol::DimExpr> &) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "wrong type std::vector<symbol::DimExpr>"));
    }
    void operator()(const std::vector<cinn::dialect::SymbolBinding> &) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "wrong type std::vector<cinn::dialect::SymbolBinding>"));
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
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of const_scalar compute is empty! "
            "Please check."));
    auto scalar = GetScalarExpr(attrs.attr_store.at("value"));
    auto scalar_type = out_type.at(0);
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        1U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 1, but got %d.",
                                          pack_args.size()));
    PADDLE_ENFORCE_EQ(pack_args[0].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "the type of pack_args[0] should be string! "
                          "Please check."));
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
    PADDLE_ENFORCE_EQ(
        out.defined(),
        true,
        ::common::errors::InvalidArgument(
            "can't create const scalar with the given type %s", out_type[0]));
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(const_scalar_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.const_scalar.x86",
                    1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForSum(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  PADDLE_THROW(::common::errors::Fatal(
      "The operator will be decomposed into several primitive "
      "operators. Please Use Decomposer Program Pass."));
}

std::shared_ptr<OpStrategy> StrategyForFillConstant(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute fill_constant_compute([=](lang::Args args,
                                                   lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument("The input argument of fill_constant "
                                          "compute is empty! Please check."));
    bool force_cpu = false;
    PADDLE_ENFORCE_EQ(attrs.attr_store.count("shape"),
                      true,
                      ::common::errors::InvalidArgument(
                          "The attribute shape of fill_constant is not found! "
                          "Please check."));
    auto shape = absl::get<std::vector<int>>(attrs.attr_store.at("shape"));
    PADDLE_ENFORCE_EQ(attrs.attr_store.count("value"),
                      true,
                      ::common::errors::InvalidArgument(
                          "The attribute value of fill_constant is not found! "
                          "Please check."));
    auto value = GetScalarExpr(attrs.attr_store.at("value"));
    PADDLE_ENFORCE_EQ(
        attrs.attr_store.count("force_cpu"),
        true,
        ::common::errors::InvalidArgument(
            "The attribute force_cpu of fill_constant is not found! "
            "Please check."));
    force_cpu = absl::get<bool>(attrs.attr_store.at("force_cpu"));

    if (force_cpu && target != cinn::common::DefaultHostTarget()) {
      LOG(WARNING) << "The attribute force_cpu of fill_constant "
                      "not supported in CINN! The fill_constant's "
                      "output tensor will placed on "
                   << target;
    }

    CINNValuePack arg_pack = args[0];
    PADDLE_ENFORCE_EQ(
        arg_pack.size(),
        1U,
        ::common::errors::InvalidArgument("the size of arg_pack should be "
                                          "equal to 1, but got %d.",
                                          arg_pack.size()));
    PADDLE_ENFORCE_EQ(arg_pack[0].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "the type of arg_pack[0] should be string! "
                          "Please check."));
    std::string tensor_name = arg_pack[0].operator std::string();
    PADDLE_ENFORCE_EQ(
        !shape.empty(),
        true,
        ::common::errors::InvalidArgument("shape attr is empty!"));
    auto shape_exprs = ToCinnExprs(shape);
    auto out = lang::Compute(
        shape_exprs,
        [=](const std::vector<Expr> &indice) {
          return ir::Cast::Make(out_type[0], value);
        },
        tensor_name);
    PADDLE_ENFORCE_EQ(
        out.defined(),
        true,
        ::common::errors::InvalidArgument(
            "can't create fill_constant with the given type %s", out_type[0]));
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(fill_constant_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.fill_constant.x86",
                    1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForFillConstantSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute fill_constant_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE_EQ(!args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The input argument of fill_constant compute "
                              "is empty! Please check."));
        bool force_cpu = false;
        auto shape = output_shapes[0];
        PADDLE_ENFORCE_EQ(attrs.attr_store.count("value"),
                          true,
                          ::common::errors::InvalidArgument(
                              "The attribute value of fill_constant "
                              "is not found! Please check."));
        auto value = GetScalarExpr(attrs.attr_store.at("value"));

        CINNValuePack arg_pack = args[0];
        PADDLE_ENFORCE_EQ(
            arg_pack.size(),
            1U,
            ::common::errors::InvalidArgument("the size of arg_pack should be "
                                              "equal to 1, but got %d.",
                                              arg_pack.size()));
        PADDLE_ENFORCE_EQ(arg_pack[0].is_string(),
                          true,
                          ::common::errors::InvalidArgument(
                              "the type of arg_pack[0] should be string! "
                              "Please check."));
        std::string tensor_name = arg_pack[0].operator std::string();
        PADDLE_ENFORCE_EQ(
            !shape.empty(),
            true,
            ::common::errors::InvalidArgument("shape attr is empty!"));
        auto shape_exprs = ToCinnExprs(shape);
        auto out = lang::Compute(
            shape_exprs,
            [=](const std::vector<Expr> &indice) {
              return ir::Cast::Make(out_type[0], value);
            },
            tensor_name);
        PADDLE_ENFORCE_EQ(out.defined(),
                          true,
                          ::common::errors::InvalidArgument(
                              "can't create fill_constant with the given type "
                              "%s",
                              out_type[0]));
        *ret = CINNValuePack{{CINNValue(out)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(fill_constant_compute,
                    lang::PackedFunc(),
                    "strategy.fill_constant.x86",
                    1);

  return strategy;
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
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input argument of assign_value compute is "
                          "empty! Please check."));
    PADDLE_ENFORCE_EQ(attrs.attr_store.count("values"),
                      true,
                      ::common::errors::InvalidArgument(
                          "The attributevalues of assign_value is not found! "
                          "Please check."));
    const auto &value = attrs.attr_store.at("values");

    CINNValuePack arg_pack = args[0];
    PADDLE_ENFORCE_EQ(
        arg_pack.size(),
        1U,
        ::common::errors::InvalidArgument("the size of arg_pack should be "
                                          "equal to 1, but got %d.",
                                          arg_pack.size()));
    PADDLE_ENFORCE_EQ(arg_pack[0].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "the type of arg_pack[0] should be string! "
                          "Please check."));
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
      std::stringstream ss;
      ss << "Assign value not support the type " << out_type[0];
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    }
#undef EXPAND_VALUE_TO_TENSOR

    PADDLE_ENFORCE(
        out.has_value(),
        ::common::errors::InvalidArgument(
            "can't create assign_value with the given type %s", out_type[0]));

    *ret = CINNValuePack{{CINNValue(Expr(out.value().get()))}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(assign_value_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.assign_value.x86",
                    1);

  return strategy;
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
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Squeeze compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(
        pack_args.size(),
        1U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 1, but got %d.",
                                          pack_args.size()));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(
        !output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output_shapes of Squeeze is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 2, but got %d.",
                                          pack_args.size()));
    std::string tensor_name = pack_args[1].operator std::string();

    ir::Tensor out = pe::Squeeze(tensor_A, axes, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Squeeze is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(squeeze_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.squeeze.x86",
                    1);
  return strategy;
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
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of ExpandDims compute is empty! "
                          "Please check."));
    CINNValuePack input_args = args[0];
    int input_size = input_args.size();
    PADDLE_ENFORCE_GE(
        input_size,
        1U,
        ::common::errors::InvalidArgument(
            "the input_size should be greater than or equal to 1, but got %d",
            input_size));
    Expr x = input_args[0];
    PADDLE_ENFORCE(x.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The input_args[0] should be tensor! Please check."));

    PADDLE_ENFORCE_EQ(
        input_args.size(),
        2U,
        ::common::errors::InvalidArgument("the size of input_args should be "
                                          "equal to 2, but got %d.",
                                          input_args.size()));
    PADDLE_ENFORCE_EQ(input_args[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "the type of input_args[1] should be string! "
                          "Please check."));
    std::string tensor_name = input_args[1].operator std::string();

    auto out =
        pe::ExpandDims(x.as_tensor_ref(), axes, output_shapes[0], tensor_name);

    std::vector<CINNValue> res{CINNValue(out)};
    *ret = CINNValuePack{res};
  }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(expand_dims_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.expand_dims.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReshape(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute reshape_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Reshape compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 1, but got %d.",
                          pack_args.size()));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(
        !output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output_shapes of Reshape is empty! Please check."));
    auto attr_store = attrs.attr_store;
    PADDLE_ENFORCE(attr_store.count("shape"),
                   ::common::errors::InvalidArgument("find no attr of shape"));
    std::vector<int> new_shape =
        absl::get<std::vector<int>>(attr_store.at("shape"));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE_EQ(pack_args[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "the type of pack_args[1] should be string! "
                          "Please check."));
    std::string tensor_name = pack_args[1].operator std::string();

    ir::Tensor out = pe::Reshape(tensor_A, output_shapes[0], tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Reshape is empty! Please check."));

    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(reshape_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.reshape.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReshapeSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute reshape_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Reshape compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 1, but got %d.",
                          pack_args.size()));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(
        !output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output_shapes of Reshape is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    std::string tensor_name;
    if (pack_args.size() == 4 || pack_args.size() == 3) {
      PADDLE_ENFORCE_EQ(pack_args[2].is_string(),
                        true,
                        ::common::errors::InvalidArgument(
                            "the type of pack_args[2] should be string! "
                            "Please check."));
      tensor_name = pack_args[2].operator std::string();
    } else {
      PADDLE_ENFORCE_EQ(pack_args[1].is_string(),
                        true,
                        ::common::errors::InvalidArgument(
                            "the type of pack_args[1] should be string! "
                            "Please check."));
      tensor_name = pack_args[1].operator std::string();
    }

    ir::Tensor out = pe::Reshape(tensor_A, output_shapes[0], tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Reshape is empty! Please check."));

    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      reshape_compute, lang::PackedFunc(), "strategy.reshape.x86", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForCast(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute cast_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Cast compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 1, but got %d.",
                          pack_args.size()));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(!output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output_shapes of Cast is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 2, but got %d.",
                                          pack_args.size()));
    std::string tensor_name = pack_args[1].operator std::string();
    ir::Tensor out = pe::Cast(tensor_A, out_type[0], tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Cast is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cast_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.reshape.x86",
                    1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForCastSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute cast_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Cast compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 1, but got %d.",
                          pack_args.size()));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(!output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output_shapes of Cast is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 2, but got %d.",
                                          pack_args.size()));
    std::string tensor_name = pack_args[1].operator std::string();
    ir::Tensor out = pe::Cast(tensor_A, out_type[0], tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Cast is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cast_compute, lang::PackedFunc(), "strategy.cast.x86", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForYieldStore(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute cast_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Cast compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 1, but got %d.",
                          pack_args.size()));

    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(!output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output_shapes of Cast is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 2, but got %d.",
                                          pack_args.size()));
    std::string tensor_name = pack_args[1].operator std::string();
    ir::Tensor out = pe::Store(tensor_A, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Cast is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cast_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.reshape.x86",
                    1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForYieldStoreSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute cast_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Cast compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 1, but got %d.",
                          pack_args.size()));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(!output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output_shapes of Cast is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 2, but got %d.",
                                          pack_args.size()));
    std::string tensor_name = pack_args[1].operator std::string();
    ir::Tensor out = pe::Store(tensor_A, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Cast is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(cast_compute, lang::PackedFunc(), "strategy.store.x86", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForGenerateShapeSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  PADDLE_ENFORCE(
      attrs.attr_store.count("output_dim_exprs"),
      ::common::errors::InvalidArgument("Expected attribute output_dim_exprs "
                                        "in strategy for generate shape op"));
  PADDLE_ENFORCE(
      attrs.attr_store.count("symbol_bindings"),
      ::common::errors::InvalidArgument("Expected attribute symbol_bindings "
                                        "in strategy for generate shape op"));
  auto output_dim_exprs = absl::get<std::vector<symbol::DimExpr>>(
      attrs.attr_store.at("output_dim_exprs"));
  auto symbol_bindings = absl::get<cinn::dialect::SymbolBindings>(
      attrs.attr_store.at("symbol_bindings"));

  framework::CINNCompute generate_shape_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE(!args.empty(),
                       ::common::errors::InvalidArgument(
                           "Invalid argument. The input arguments of "
                           "generate_shape compute is empty! Please check."));
        CINNValuePack pack_args = args[0];
        PADDLE_ENFORCE_GE(pack_args->size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "At least 1 input tensors for generate_shape "
                              "compute, but now get %d.",
                              pack_args->size()));

        std::string tensor_name = pack_args.back().operator std::string();
        ir::Tensor out = pe::GenerateShape(inputs,
                                           symbol_bindings,
                                           output_dim_exprs,
                                           output_shapes[0],
                                           out_type,
                                           tensor_name);
        std::vector<CINNValue> res;
        res.push_back(CINNValue(out));
        PADDLE_ENFORCE(!out_type.empty(),
                       ::common::errors::InvalidArgument(
                           "Invalid argument. The output type of "
                           "generate_shape is empty! Please check."));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      generate_shape_compute, lang::PackedFunc(), "strategy.store.x86", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForArange(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  PADDLE_ENFORCE_EQ(
      attr_store.count("start"),
      true,
      ::common::errors::InvalidArgument(
          "No start attribute in attrs.attr_store! Please check."));
  PADDLE_ENFORCE_EQ(
      attr_store.count("stop"),
      true,
      ::common::errors::InvalidArgument(
          "No stop attribute in attrs.attr_store! Please check."));
  PADDLE_ENFORCE_EQ(
      attr_store.count("step"),
      true,
      ::common::errors::InvalidArgument(
          "No step attribute in attrs.attr_store! Please check."));
  PADDLE_ENFORCE_EQ(
      attr_store.count("dtype"),
      true,
      ::common::errors::InvalidArgument(
          "No dtype attribute in attrs.attr_store! Please check."));

  auto start = absl::get<float>(attr_store.at("start"));
  auto stop = absl::get<float>(attr_store.at("stop"));
  auto step = absl::get<float>(attr_store.at("step"));
  auto dtype =
      cinn::common::Str2Type(absl::get<std::string>(attr_store.at("dtype")));

  framework::CINNCompute arange_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE(!args.empty(),
                       ::common::errors::InvalidArgument(
                           "The input argument of arange compute is empty! "
                           "Please check."));
        CINNValuePack pack_args = args[0];

        PADDLE_ENFORCE_EQ(
            pack_args.size(),
            1U,
            ::common::errors::InvalidArgument("the size of pack_args should be "
                                              "equal to 1, but got %d.",
                                              pack_args.size()));
        std::string tensor_name = pack_args[0].operator std::string();

        auto out = pe::Arange(start, stop, step, dtype, tensor_name);
        std::vector<cinn::common::CINNValue> res;
        res.push_back(cinn::common::CINNValue(out));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(arange_compute,
                    GetElementwiseScheduleFunc(output_shapes, target),
                    "strategy.reshape.x86",
                    1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForArangeSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  PADDLE_ENFORCE_GT(attr_store.count("start"),
                    0U,
                    ::common::errors::InvalidArgument(
                        "No start attribute in arange Op! Please check."));
  PADDLE_ENFORCE_GT(attr_store.count("stop"),
                    0U,
                    ::common::errors::InvalidArgument(
                        "No stop attribute in arange Op! Please check."));
  PADDLE_ENFORCE_GT(attr_store.count("step"),
                    0U,
                    ::common::errors::InvalidArgument(
                        "No step attribute in arange Op! Please check."));
  PADDLE_ENFORCE_GT(attr_store.count("dtype"),
                    0U,
                    ::common::errors::InvalidArgument(
                        "No dtype attribute in arange Op! Please check."));

  auto start = absl::get<float>(attr_store.at("start"));
  auto stop = absl::get<float>(attr_store.at("stop"));
  auto step = absl::get<float>(attr_store.at("step"));
  auto dtype =
      cinn::common::Str2Type(absl::get<std::string>(attr_store.at("dtype")));

  framework::CINNCompute arange_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of arange compute is empty! Please check."));
    CINNValuePack pack_args = args[0];

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "The number of input argument of arange should be at "
                          "last 1. Please check."));
    std::string tensor_name = pack_args[0].operator std::string();

    auto out = pe::Arange(start, stop, step, dtype, tensor_name);
    std::vector<cinn::common::CINNValue> res;
    res.push_back(cinn::common::CINNValue(out));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      arange_compute, lang::PackedFunc(), "strategy.reshape.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForTril(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute tril_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(args.size(),
                      size_t(1),
                      ::common::errors::InvalidArgument(
                          "The input arguments of tril compute is empty"));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      size_t(1),
                      ::common::errors::InvalidArgument(
                          "only 1 input tensor for tril compute"));
    Expr A = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        A.as_tensor(),
        ::common::errors::InvalidArgument(
            "first input argument in tril should be tensor"));
    int diagonal = absl::get<int>(attrs.attr_store.at("diagonal"));
    auto tensor_A = A.as_tensor_ref();

    PADDLE_ENFORCE_NE(output_shapes.size(),
                      size_t(0),
                      ::common::errors::InvalidArgument(
                          "output shape of tril should not be empty."));
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      size_t(2),
                      ::common::errors::InvalidArgument(
                          "args of tril compute should be equal to 2"));
    PADDLE_ENFORCE_EQ(pack_args[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The second argument of tril should be string"));
    std::string tensor_name = pack_args[1].operator std::string();

    ir::Tensor out =
        pe::Tril(tensor_A, diagonal, output_shapes[0], tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Reshape is empty! Please check.\n"));

    *ret = CINNValuePack{res};
  });
  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(tril_compute, lang::PackedFunc(), "strategy.tril.x86", 1);

  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForAssignOutSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute assign_out_compute([=](lang::Args args,
                                                lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of AssignOut compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        3U,
        ::common::errors::InvalidArgument("the size of pack_args should be "
                                          "equal to 3, but got %d.",
                                          pack_args.size()));
    Expr x = pack_args[0];
    PADDLE_ENFORCE(x.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    Expr out = pack_args[1];
    PADDLE_ENFORCE(out.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[1] should be tensor! Please check."));
    PADDLE_ENFORCE_EQ(!output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output_shapes of AssignOut is empty! Please "
                          "check."));
    auto tensor_x = x.as_tensor_ref();
    auto tensor_out = out.as_tensor_ref();

    std::string tensor_name = pack_args[2].operator std::string();
    auto new_out = Compute(
        tensor_x->shape,
        [=](const std::vector<Expr> &indice) { return tensor_x(indice); },
        tensor_name);

    PADDLE_ENFORCE_EQ(
        !out_type.empty(),
        true,
        ::common::errors::InvalidArgument(
            "Output type of AssignOut is empty! Please check.\n"));
    if (!tensor_out->buffer.defined()) {
      tensor_out->WithBuffer(out_type.front());
    }
    new_out->Bind(tensor_out->buffer);

    std::vector<CINNValue> res{CINNValue(new_out)};
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      assign_out_compute, lang::PackedFunc(), "strategy.default", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForIsClose(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<shape_t> &output_shapes,
    const Target &target) {
  float rtol = 1e-05f, atol = 1e-08f;
  bool equal_nan = false;
  int axis = -1;

  if (attrs.attr_store.count("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  if (attrs.attr_store.count("rtol")) {
    rtol = absl::get<float>(attrs.attr_store.at("rtol"));
  }
  if (attrs.attr_store.count("atol")) {
    atol = absl::get<float>(attrs.attr_store.at("atol"));
  }
  if (attrs.attr_store.count("equal_nan")) {
    equal_nan = absl::get<bool>(attrs.attr_store.at("equal_nan"));
  }

  framework::CINNCompute isclose_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of isclose compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    int input_size = pack_args.size();

    // the last pack argument is the output tensor name
    std::string tensor_name = pack_args.back().operator std::string();
    --input_size;
    PADDLE_ENFORCE_EQ(
        input_size,
        2,
        ::common::errors::InvalidArgument(
            "the input_size should be 2, but got %d.", input_size));

    // the input tensor are in front
    Expr x_expr = pack_args[0];
    PADDLE_ENFORCE(x_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    auto x_tensor = x_expr.as_tensor_ref();

    Expr y_expr = pack_args[1];
    PADDLE_ENFORCE(y_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[1] should be tensor! Please check."));
    auto y_tensor = y_expr.as_tensor_ref();

    auto out = pe::IsClose(
        x_tensor, y_tensor, axis, rtol, atol, equal_nan, tensor_name);

    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(isclose_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.assertisclose",
                    1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForIsCloseSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  float rtol = 1e-05f, atol = 1e-08f;
  bool equal_nan = false;
  int axis = -1;

  if (attrs.attr_store.count("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  if (attrs.attr_store.count("rtol")) {
    rtol = absl::get<float>(attrs.attr_store.at("rtol"));
  }
  if (attrs.attr_store.count("atol")) {
    atol = absl::get<float>(attrs.attr_store.at("atol"));
  }
  if (attrs.attr_store.count("equal_nan")) {
    equal_nan = absl::get<bool>(attrs.attr_store.at("equal_nan"));
  }

  framework::CINNCompute isclose_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of isclose compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    int input_size = pack_args.size();

    // the last pack argument is the output tensor name
    std::string tensor_name = pack_args.back().operator std::string();
    --input_size;
    PADDLE_ENFORCE_EQ(
        input_size,
        2,
        ::common::errors::InvalidArgument(
            "the input_size should be 2, but got %d.", input_size));

    // the input tensor are in front
    Expr x_expr = pack_args[0];
    PADDLE_ENFORCE(x_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[0] should be tensor! Please check."));
    auto x_tensor = x_expr.as_tensor_ref();

    Expr y_expr = pack_args[1];
    PADDLE_ENFORCE(y_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The pack_args[1] should be tensor! Please check."));
    auto y_tensor = y_expr.as_tensor_ref();

    auto out = pe::IsClose(
        x_tensor, y_tensor, axis, rtol, atol, equal_nan, tensor_name);

    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      isclose_compute, lang::PackedFunc(), "strategy.assertisclose", 1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(elementwise_ops) {
#define CINN_REGISTER_UNARY(op__, op_strategy__)                           \
  CINN_REGISTER_OP(op__)                                                   \
      .describe(#op__ " function")                                         \
      .set_num_inputs(1)                                                   \
      .set_num_outputs(1)                                                  \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                  \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)      \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(          \
          "CINNStrategySymbolic",                                          \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)            \
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

#define CINN_REGISTER_COMPARE(op__, op_strategy__)                         \
  CINN_REGISTER_OP(op__)                                                   \
      .describe(#op__ " function")                                         \
      .set_num_inputs(1)                                                   \
      .set_num_outputs(1)                                                  \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                  \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)      \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(          \
          "CINNStrategySymbolic",                                          \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)            \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                     \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise) \
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
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForScaleSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(const_scalar)
      .describe("create const scalar with the given value")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForConstScalar)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(sum)
      .describe("Sum the input tensors.")
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSum)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  CINN_REGISTER_OP(fill_constant)
      .describe("create tensor with the given value, type and shape")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForFillConstant)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic",
          cinn::hlir::op::StrategyForFillConstantSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(assign_value)
      .describe("create tensor with the given value, type and shape")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForAssignValue)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(squeeze)
      .describe("The operator is used to squeeze input tensor's dims")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSqueeze)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(expand_dims)
      .describe("This operator is used to expand input tensor's dims.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForExpandDims)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(reshape)
      .describe("This operator is used to reshape input tensor X.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForReshape)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForReshapeSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(cast)
      .describe("This operator is used to cast input tensor's type to target.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForCast)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForCastSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(yield_store)
      .describe("This operator is used to cast input tensor's type to target.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForYieldStore)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForYieldStoreSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(generate_shape)
      .describe("This operator is used to cast input tensor's type to target.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic",
          cinn::hlir::op::StrategyForGenerateShapeSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  CINN_REGISTER_OP(arange)
      .describe("Returns evenly spaced values within a given interval.")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForArange)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForArangeSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(logical_not)
      .describe("Logical not function")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForLogicalNot)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForLogicalNotSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  CINN_REGISTER_OP(tril)
      .describe(
          "Filters out the upper portion of an input tensor on one side of a "
          "diagonal")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForTril)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  CINN_REGISTER_OP(assign_out_)
      .describe("Copy the value of the first parameter to the second one")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForAssignOutSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise);

  CINN_REGISTER_OP(isclose)
      .describe(
          "This operator checks if all x and y satisfy the condition: |x - y| "
          "<= atol + rtol * |y|")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForIsClose)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForIsCloseSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}

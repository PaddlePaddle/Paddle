// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/anf_expr_builder.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/code_gen/arg_source_ctx.h"
#include "paddle/ap/include/code_gen/kernel_arg_id.h"

namespace ap::code_gen {

template <typename BirNode>
struct ArgSourceHelper {
  const ArgSourceCtx<BirNode>& arg_source_ctx;

  adt::Result<axpr::Function<axpr::SerializableValue>>
  MakeRuntimeKerneArgsGetter(
      const std::list<KernelArgId<BirNode>>& kernel_arg_ids) const {
    auto GetBody =
        [&](axpr::LetVar* dispatch_ctx) -> adt::Result<axpr::LetVar*> {
      std::vector<axpr::LetVar> items;
      items.reserve(kernel_arg_ids.size());
      for (const auto& kernel_arg_id : kernel_arg_ids) {
        ADT_LET_CONST_REF(elt, MakeRuntimeGetter(dispatch_ctx, kernel_arg_id));
        items.emplace_back(*elt);
      }
      auto* ctx = dispatch_ctx->ctx();
      auto* ret_ptr = &ctx->Var(ctx->NewTmpVarName());
      *ret_ptr = ctx->Var(axpr::kBuiltinList()).Call(items);
      return ret_ptr;
    };
    ADT_LET_CONST_REF(lambda, CreateLambda("ctx", GetBody));
    return axpr::Function<axpr::SerializableValue>{lambda, std::nullopt};
  }

  adt::Result<axpr::Function<axpr::SerializableValue>>
  MakeRuntimeKerneArgGetter(const KernelArgId<BirNode>& kernel_arg_id) const {
    auto GetBody =
        [&](axpr::LetVar* dispatch_ctx) -> adt::Result<axpr::LetVar*> {
      return MakeRuntimeGetter(dispatch_ctx, kernel_arg_id);
    };
    ADT_LET_CONST_REF(lambda, CreateLambda("ctx", GetBody));
    return axpr::Function<axpr::SerializableValue>{lambda, std::nullopt};
  }

 private:
  adt::Result<axpr::LetVar*> MakeRuntimeGetter(
      axpr::LetVar* dispatch_ctx,
      const KernelArgId<BirNode>& kernel_arg_id) const {
    return kernel_arg_id.Match([&](const auto& impl) {
      return MakeRuntimeGetterImpl(dispatch_ctx, impl);
    });
  }

  adt::Result<axpr::LetVar*> MakeRuntimeGetterImpl(
      axpr::LetVar* dispatch_ctx,
      const InTensorDataPtrKernelArgId<BirNode>& kernel_arg_id) const {
    const auto& opt_in_tensor_source =
        arg_source_ctx->GetInputTensorSource(kernel_arg_id->ir_value);
    ADT_CHECK(opt_in_tensor_source.has_value());
    const auto* in_tensor_source = opt_in_tensor_source.value();
    ADT_LET_CONST_REF(
        tensor_var_ptr,
        MakeGetterAnfExprByInTensorSource(dispatch_ctx, *in_tensor_source));
    return &tensor_var_ptr->Attr("data_ptr");
  }

  adt::Result<axpr::LetVar*> MakeRuntimeGetterImpl(
      axpr::LetVar* dispatch_ctx,
      const OutTensorDataPtrKernelArgId<BirNode>& kernel_arg_id) const {
    const auto& opt_out_tensor_source =
        arg_source_ctx->GetOutputTensorSource(kernel_arg_id->ir_value);
    ADT_CHECK(opt_out_tensor_source.has_value());
    const auto* out_tensor_source = opt_out_tensor_source.value();
    ADT_LET_CONST_REF(
        tensor_var_ptr,
        MakeGetterAnfExprByOutTensorSource(dispatch_ctx, *out_tensor_source));
    return &tensor_var_ptr->Attr("data_ptr");
  }

  template <typename GetVarPtrT>
  adt::Result<axpr::Lambda<axpr::CoreExpr>> CreateLambda(
      const std::string& dispatch_ctx_name, const GetVarPtrT& GetVarPtr) const {
    axpr::LambdaExprBuilder lmbd;
    auto GetBody =
        [&](axpr::LetContext& ctx) -> adt::Result<axpr::AnfExpr> {  // NOLINT
      ADT_LET_CONST_REF(var_ptr, GetVarPtr(&ctx.Var(dispatch_ctx_name)));
      return static_cast<axpr::AnfExpr>(*var_ptr);
    };
    ADT_LET_CONST_REF(anf_expr, lmbd.TryLambda({dispatch_ctx_name}, GetBody));
    const auto& core_expr = ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return lambda;
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByInTensorSource(
      axpr::LetVar* dispatch_ctx,
      const InTensorSource& in_tensor_source) const {
    auto& inputs_var = dispatch_ctx->Attr("inputs");
    return MakeGetterAnfExprByTensorSource(&inputs_var,
                                           in_tensor_source.tensor_source);
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByOutTensorSource(
      axpr::LetVar* dispatch_ctx,
      const OutTensorSource& out_tensor_source) const {
    auto& outputs_var = dispatch_ctx->Attr("outputs");
    return MakeGetterAnfExprByTensorSource(&outputs_var,
                                           out_tensor_source.tensor_source);
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByTensorSource(
      axpr::LetVar* in_out_tensors, const TensorSource& tensor_source) const {
    using RetT = adt::Result<axpr::LetVar*>;
    return tensor_source.Match(
        [&](const NativeIrValueSource& native) -> RetT {
          return &in_out_tensors->At(native.native_ir_value_index);
        },
        [&](const PackedIrValueSource& packed) -> RetT {
          return &in_out_tensors->At(packed.packed_ir_value_index)
                      .At(packed.tensor_member_index);
        });
  }

  adt::Result<axpr::LetVar*> MakeRuntimeGetterImpl(
      axpr::LetVar* dispatch_ctx,
      const DimExprKernelArgId<BirNode>& kernel_arg_id) const {
    ADT_CHECK(arg_source_ctx->HasDirectOrIndirectDimExprSource(
        kernel_arg_id->dim_expr));
    return MakeGetterAnfExprByDimExpr(dispatch_ctx, kernel_arg_id->dim_expr);
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimSource(
      axpr::LetVar* dispatch_ctx, const DimSource& dim_source) const {
    using RetT = adt::Result<axpr::LetVar*>;
    return dim_source.Match(
        [&](const ShapeDimSource& shape_dim_source) -> RetT {
          ADT_LET_CONST_REF(tensor_var_ptr,
                            MakeGetterAnfExprByInOutTensorSource(
                                dispatch_ctx, shape_dim_source.tensor_source));
          auto* ctx = dispatch_ctx->ctx();
          auto* dim_expr =
              &tensor_var_ptr->Attr("shape").At(shape_dim_source.dim_axis);
          auto* data_value = &ctx->Var("DataValue").Call(*dim_expr);
          auto* ret = &ctx->Var(ctx->NewTmpVarName());
          *ret = data_value->Attr("cast").Call(
              ctx->Var("DataType").Attr("const_int64"));
          return ret;
        },
        [&](const DataDimSource& data_dim_source) -> RetT {
          return adt::errors::TypeError{"DataDimSource is not supported yet."};
        });
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByInOutTensorSource(
      axpr::LetVar* dispatch_ctx,
      const InOutTensorSource& in_out_tensor_source) const {
    using RetT = adt::Result<axpr::LetVar*>;
    return in_out_tensor_source.Match(
        [&](const InTensorSource& in_tensor_source) -> RetT {
          return MakeGetterAnfExprByInTensorSource(dispatch_ctx,
                                                   in_tensor_source);
        },
        [&](const OutTensorSource& out_tensor_source) -> RetT {
          return MakeGetterAnfExprByOutTensorSource(dispatch_ctx,
                                                    out_tensor_source);
        });
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExpr(
      axpr::LetVar* dispatch_ctx, const symbol::DimExpr& dim_expr) const {
    const auto& opt_dim_source = arg_source_ctx->GetDimExprSource(dim_expr);
    if (opt_dim_source.has_value()) {
      return MakeGetterAnfExprByDimSource(dispatch_ctx,
                                          *opt_dim_source.value());
    }
    return dim_expr.Match([&](const auto& impl) {
      return MakeGetterAnfExprByDimExprImpl(dispatch_ctx, impl);
    });
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx, int64_t c) const {
    auto* ctx = dispatch_ctx->ctx();
    auto* ret_var = &ctx->Var(ctx->NewTmpVarName());
    *ret_var = ctx->Int64(c);
    return ret_var;
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx, const std::string& symbol) const {
    return adt::errors::NotImplementedError{
        "Dead code. Symbols have been handled in MakeGetterAnfExprByDimExpr()"};
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx,
      const symbol::Negative<symbol::DimExpr>& dim_expr) const {
    return adt::errors::NotImplementedError{
        "Dead code. Negative dim_exprs have been handled in "
        "MakeGetterAnfExprByDimExprImpl(dispatch_ctx, const "
        "symbol::Add<symbol::DimExpr>&)"};
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx,
      const symbol::Add<symbol::DimExpr>& dim_expr) const {
    const auto& [operands] = dim_expr;
    ADT_CHECK(operands->size() > 0);
    auto* ctx = dispatch_ctx->ctx();
    ADT_LET_CONST_REF(
        init_var_ptr,
        MakeGetterAnfExprByDimExpr(dispatch_ctx, operands->at(0)));
    axpr::LetVar* ret_var_ptr = init_var_ptr;
    for (int i = 1; i < operands->size(); ++i) {
      const auto& operand = operands->at(i);
      auto* tmp_var_ptr = &ctx->Var(ctx->NewTmpVarName());
      if (operand.template Has<symbol::Negative<symbol::DimExpr>>()) {
        const auto& [operand_operand] =
            *operand.template Get<symbol::Negative<symbol::DimExpr>>();
        ADT_LET_CONST_REF(
            operand_operand_var_ptr,
            MakeGetterAnfExprByDimExpr(dispatch_ctx, operand_operand));
        *tmp_var_ptr = ctx->Call(
            axpr::kBuiltinSub(), *ret_var_ptr, *operand_operand_var_ptr);
      } else {
        ADT_LET_CONST_REF(operand_var_ptr,
                          MakeGetterAnfExprByDimExpr(dispatch_ctx, operand));
        *tmp_var_ptr =
            ctx->Call(axpr::kBuiltinAdd(), *ret_var_ptr, *operand_var_ptr);
      }
      ret_var_ptr = tmp_var_ptr;
    }
    return ret_var_ptr;
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx,
      const symbol::Mul<symbol::DimExpr>& dim_expr) const {
    const auto& [operands] = dim_expr;
    ADT_CHECK(operands->size() > 0);
    auto* ctx = dispatch_ctx->ctx();
    ADT_LET_CONST_REF(
        init_var_ptr,
        MakeGetterAnfExprByDimExpr(dispatch_ctx, operands->at(0)));
    axpr::LetVar* ret_var_ptr = init_var_ptr;
    for (int i = 1; i < operands->size(); ++i) {
      const auto& operand = operands->at(i);
      auto* tmp_var_ptr = &ctx->Var(ctx->NewTmpVarName());
      ADT_LET_CONST_REF(operand_var_ptr,
                        MakeGetterAnfExprByDimExpr(dispatch_ctx, operand));
      *tmp_var_ptr =
          ctx->Call(axpr::kBuiltinMul(), *ret_var_ptr, *operand_var_ptr);
      ret_var_ptr = tmp_var_ptr;
    }
    return ret_var_ptr;
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx,
      const symbol::Div<symbol::DimExpr>& dim_expr) const {
    const auto& operand_lhs = (*dim_expr).lhs;
    const auto& operand_rhs = (*dim_expr).rhs;
    auto* ctx = dispatch_ctx->ctx();
    ADT_LET_CONST_REF(init_var_ptr,
                      MakeGetterAnfExprByDimExpr(dispatch_ctx, operand_lhs));
    axpr::LetVar* ret_var_ptr = init_var_ptr;
    auto* tmp_var_ptr = &ctx->Var(ctx->NewTmpVarName());
    ADT_LET_CONST_REF(operand_var_ptr,
                      MakeGetterAnfExprByDimExpr(dispatch_ctx, operand_rhs));
    *tmp_var_ptr =
        ctx->Call(axpr::kBuiltinDiv(), *ret_var_ptr, *operand_var_ptr);
    ret_var_ptr = tmp_var_ptr;
    return ret_var_ptr;
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx,
      const symbol::Max<symbol::DimExpr>& dim_expr) const {
    const auto& [operands] = dim_expr;
    ADT_CHECK(operands->size() > 0);
    auto* ctx = dispatch_ctx->ctx();
    ADT_LET_CONST_REF(
        init_var_ptr,
        MakeGetterAnfExprByDimExpr(dispatch_ctx, operands->at(0)));
    axpr::LetVar* ret_var_ptr = init_var_ptr;
    for (int i = 1; i < operands->size(); ++i) {
      const auto& operand = operands->at(i);
      auto* tmp_var_ptr = &ctx->Var(ctx->NewTmpVarName());
      ADT_LET_CONST_REF(operand_var_ptr,
                        MakeGetterAnfExprByDimExpr(dispatch_ctx, operand));
      *tmp_var_ptr = ctx->Call("max", *ret_var_ptr, *operand_var_ptr);
      ret_var_ptr = tmp_var_ptr;
    }
    return ret_var_ptr;
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx,
      const symbol::Min<symbol::DimExpr>& dim_expr) const {
    const auto& [operands] = dim_expr;
    ADT_CHECK(operands->size() > 0);
    auto* ctx = dispatch_ctx->ctx();
    ADT_LET_CONST_REF(
        init_var_ptr,
        MakeGetterAnfExprByDimExpr(dispatch_ctx, operands->at(0)));
    axpr::LetVar* ret_var_ptr = init_var_ptr;
    for (int i = 1; i < operands->size(); ++i) {
      const auto& operand = operands->at(i);
      auto* tmp_var_ptr = &ctx->Var(ctx->NewTmpVarName());
      ADT_LET_CONST_REF(operand_var_ptr,
                        MakeGetterAnfExprByDimExpr(dispatch_ctx, operand));
      *tmp_var_ptr = ctx->Call("min", *ret_var_ptr, *operand_var_ptr);
      ret_var_ptr = tmp_var_ptr;
    }
    return ret_var_ptr;
  }

  adt::Result<axpr::LetVar*> MakeGetterAnfExprByDimExprImpl(
      axpr::LetVar* dispatch_ctx,
      const symbol::Broadcast<symbol::DimExpr>& dim_expr) const {
    const auto& [operands] = dim_expr;
    ADT_CHECK(operands->size() > 0);
    auto* ctx = dispatch_ctx->ctx();
    ADT_LET_CONST_REF(
        init_var_ptr,
        MakeGetterAnfExprByDimExpr(dispatch_ctx, operands->at(0)));
    axpr::LetVar* ret_var_ptr = init_var_ptr;
    for (int i = 1; i < operands->size(); ++i) {
      const auto& operand = operands->at(i);
      auto* tmp_var_ptr = &ctx->Var(ctx->NewTmpVarName());
      ADT_LET_CONST_REF(operand_var_ptr,
                        MakeGetterAnfExprByDimExpr(dispatch_ctx, operand));
      *tmp_var_ptr = ctx->Call("broadcast", *ret_var_ptr, *operand_var_ptr);
      ret_var_ptr = tmp_var_ptr;
    }
    return ret_var_ptr;
  }
};

}  // namespace ap::code_gen

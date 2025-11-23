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

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/packed_args.h"
#include "paddle/ap/include/code_gen/arg_source_helper.h"
#include "paddle/ap/include/code_gen/cuda_code_gen_util.h"
#include "paddle/ap/include/code_gen/dim_expr_kernel_arg_id_method_class.h"
#include "paddle/ap/include/code_gen/in_tensor_data_ptr_kernel_arg_id_method_class.h"
#include "paddle/ap/include/code_gen/ir_op.h"
#include "paddle/ap/include/code_gen/kernel_arg_id_helper.h"
#include "paddle/ap/include/code_gen/op_code_gen_ctx.h"
#include "paddle/ap/include/code_gen/out_tensor_data_ptr_kernel_arg_id_method_class.h"
#include "paddle/ap/include/code_module/code_module.h"
#include "paddle/ap/include/ir_match/native_or_ref_ir_value.h"
#include "paddle/ap/include/registry/registry_singleton.h"

namespace ap::code_gen {

using ap::axpr::BuiltinBinaryFunc;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFunc;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;

template <typename ValueT, typename BirNode>
struct CodeGenCtxMethodClass {
  using This = CodeGenCtxMethodClass;
  using Self = CodeGenCtx<BirNode>;

  static adt::Result<ValueT> StaticMakeAndCheckOutTensorDataPtrKernelArgId(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return This{}.MakeAndCheckOutTensorDataPtrKernelArgId(self, args);
  }

  adt::Result<ValueT> MakeAndCheckOutTensorDataPtrKernelArgId(
      const Self& self, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1)
        << adt::errors::TypeError{std::string() +
                                  "out_tensor_data_ptr_kernel_"
                                  "arg_id() takes 1 argument but " +
                                  std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(ir_value, CastToBirValue(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of "
               "out_tensor_data_ptr_kernel_arg_id() should be "
               "'NativeIrValue' or 'RefIrValue' (not '" +
               axpr::GetTypeName(args.at(0)) + "')."};
    ADT_RETURN_IF_ERR(CheckOutTensorDataPtrRuntimeAvailable(self, ir_value));
    OutTensorDataPtrKernelArgId<BirNode> uninitialized{ir_value, std::nullopt};
    ArgSourceHelper<BirNode> helper{self->arg_source_ctx};
    ADT_LET_CONST_REF(runtime_getter,
                      helper.MakeRuntimeKerneArgGetter(uninitialized));
    OutTensorDataPtrKernelArgId<BirNode> kernel_arg_id{ir_value,
                                                       runtime_getter};
    axpr::BuiltinClassInstance<ValueT> instance{
        GetOutTensorDataPtrKernelArgIdClass<ValueT, BirNode>(), kernel_arg_id};
    return instance;
  }

  adt::Result<adt::Ok> CheckOutTensorDataPtrRuntimeAvailable(
      const Self& self, const BirNode& ir_value) {
    ADT_CHECK(self->arg_source_ctx->GetOutputTensorSource(ir_value).has_value())
        << adt::errors::TypeError{
               std::string() +
               "out_tensor_data_ptr_kernel_arg_id() failed. "
               "please check whether the ir_value is an output value of the "
               "current ap_pattern_fusion_op defined in drr result pattern "
               "lambda."};
    return adt::Ok{};
  }

  static adt::Result<ValueT> StaticMakeAndCheckInTensorDataPtrKernelArgId(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return This{}.MakeAndCheckInTensorDataPtrKernelArgId(self, args);
  }

  adt::Result<ValueT> MakeAndCheckInTensorDataPtrKernelArgId(
      const Self& self, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1)
        << adt::errors::TypeError{std::string() +
                                  "in_tensor_data_ptr_kernel_"
                                  "arg_id() takes 1 argument but " +
                                  std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(ir_value, CastToBirValue(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of "
               "in_tensor_data_ptr_kernel_arg_id() should be "
               "'NativeIrValue' or 'RefIrValue' (not '" +
               axpr::GetTypeName(args.at(0)) + "')."};
    ADT_RETURN_IF_ERR(CheckInTensorDataPtrRuntimeAvailable(self, ir_value));
    InTensorDataPtrKernelArgId<BirNode> uninitialized{ir_value, std::nullopt};
    ArgSourceHelper<BirNode> helper{self->arg_source_ctx};
    ADT_LET_CONST_REF(runtime_getter,
                      helper.MakeRuntimeKerneArgGetter(uninitialized));
    InTensorDataPtrKernelArgId<BirNode> kernel_arg_id{ir_value, runtime_getter};
    axpr::BuiltinClassInstance<ValueT> instance{
        GetInTensorDataPtrKernelArgIdClass<ValueT, BirNode>(), kernel_arg_id};
    return instance;
  }

  adt::Result<adt::Ok> CheckInTensorDataPtrRuntimeAvailable(
      const Self& self, const BirNode& ir_value) {
    ADT_CHECK(self->arg_source_ctx->GetInputTensorSource(ir_value).has_value())
        << adt::errors::TypeError{
               std::string() +
               "in_tensor_data_ptr_kernel_arg_id() failed. "
               "please check whether the ir_value is an input value of the "
               "current ap_pattern_fusion_op defined in drr result pattern "
               "lambda."};
    return adt::Ok{};
  }

  adt::Result<BirNode> CastToBirValue(const ValueT& val) {
    ADT_LET_CONST_REF(
        instance, val.template CastTo<axpr::BuiltinClassInstance<ValueT>>());
    if (instance.template Has<typename BirNode::native_value_type>()) {
      ADT_LET_CONST_REF(
          ret, instance.template TryGet<typename BirNode::native_value_type>());
      return ret;
    }
    if (instance.template Has<typename BirNode::ref_value_type>()) {
      ADT_LET_CONST_REF(
          ret, instance.template TryGet<typename BirNode::ref_value_type>());
      return ret;
    }
    return adt::errors::NotImplementedError{
        std::string() +
        "CastToBirValue() failed. only 'NativeIrValue' and 'RefIrValue' "
        "argument is expected, but '" +
        axpr::GetTypeName(val) + "' found."};
  }

  static adt::Result<ValueT> StaticMakeAndCheckDimExprKernelArgId(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return This{}.MakeAndCheckDimExprKernelArgId(self, args);
  }

  adt::Result<ValueT> MakeAndCheckDimExprKernelArgId(
      const Self& self, const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "dim_expr_kernel_arg_id() takes 1 arguments but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(dim_expr, args.at(0).template CastTo<symbol::DimExpr>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of dim_expr_kernel_arg_id() "
                                  "should be 'DimExpr' (not '" +
                                  axpr::GetTypeName(args.at(0)) + "')."};
    ADT_RETURN_IF_ERR(CheckDimExprRuntimeAvailable(self, dim_expr));
    DimExprKernelArgId<BirNode> uninitialized{dim_expr, std::nullopt};
    ArgSourceHelper<BirNode> helper{self->arg_source_ctx};
    ADT_LET_CONST_REF(runtime_getter,
                      helper.MakeRuntimeKerneArgGetter(uninitialized));
    DimExprKernelArgId<BirNode> kernel_arg_id{dim_expr, runtime_getter};
    axpr::BuiltinClassInstance<ValueT> instance{
        GetDimExprKernelArgIdClass<ValueT, BirNode>(), kernel_arg_id};
    return instance;
  }

  adt::Result<adt::Ok> CheckDimExprRuntimeAvailable(
      const Self& self, const symbol::DimExpr& dim_expr) {
    ADT_CHECK(self->arg_source_ctx->HasDirectOrIndirectDimExprSource(dim_expr))
        << adt::errors::ValueError{
               std::string() +
               "DimExpr could not evaluated in runtime. value: " +
               symbol::ToString(dim_expr)};
    return adt::Ok{};
  }
};

template <typename ValueT, typename BirNode>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetCodeGenCtxClass() {
  using ImplMethods = CodeGenCtxMethodClass<ValueT, BirNode>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("CodeGenCtx", [&](const auto& Define) {
        Define("dim_expr_kernel_arg_id",
               &ImplMethods::StaticMakeAndCheckDimExprKernelArgId);
        Define("in_tensor_data_ptr_kernel_arg_id",
               &ImplMethods::StaticMakeAndCheckInTensorDataPtrKernelArgId);
        Define("out_tensor_data_ptr_kernel_arg_id",
               &ImplMethods::StaticMakeAndCheckOutTensorDataPtrKernelArgId);
      }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::code_gen

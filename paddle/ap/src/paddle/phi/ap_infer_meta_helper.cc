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

#include "paddle/ap/include/paddle/phi/ap_infer_meta_helper.h"
#include <mutex>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/axpr/value_method_class.h"
#include "paddle/ap/include/paddle/builtin_frame_util.h"
#include "paddle/ap/include/paddle/const_meta_tensor_ptr.h"
#include "paddle/ap/include/paddle/const_meta_tensor_ptr_method_class.h"
#include "paddle/ap/include/paddle/const_std_vector_const_meta_tensor_ptr_ptr_method_class.h"
#include "paddle/ap/include/paddle/ddim.h"
#include "paddle/ap/include/paddle/ddim_method_class.h"
#include "paddle/ap/include/paddle/meta_tensor_ptr.h"
#include "paddle/ap/include/paddle/meta_tensor_ptr_method_class.h"
#include "paddle/ap/include/paddle/std_vector_meta_tensor_ptr_ptr_method_class.h"

namespace phi {

namespace {

namespace adt = ap::adt;

using CoreExpr = ap::axpr::CoreExpr;
using Lambda = ap::axpr::Lambda<CoreExpr>;

adt::Result<adt::Ok> InferMetaByLambda(
    const Lambda& lambda,
    const std::vector<const MetaTensor*>* inputs,
    std::vector<MetaTensor*>* outputs) {
  ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::paddle::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_RETURN_IF_ERR(interpreter.Interpret(
      lambda,
      {ap::paddle::GetConstStdVectorConstMetaTensorPtrPtrClass().New(inputs),
       ap::paddle::GetStdVectorMetaTensorPtrPtrClass().New(outputs)}));
  return adt::Ok{};
}

adt::Result<adt::Ok> InferMetaByLambda(
    const Lambda& lambda,
    const ::paddle::optional<std::vector<const MetaTensor*>>& inputs,
    const ap::axpr::AttrMap<ap::axpr::Value>& attrs,
    const std::vector<MetaTensor*>& outputs) {
  ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::paddle::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  adt::List<ap::axpr::Value> inputs_val{};
  if (inputs.is_initialized()) {
    inputs_val->reserve(inputs->size());
    for (const auto& input : *inputs) {
      inputs_val->emplace_back(
          ap::paddle::GetConstMetaTensorPtrClass().New(input));
    }
  }
  adt::List<ap::axpr::Value> outputs_val{};
  outputs_val->reserve(outputs.size());
  for (const auto& output : outputs) {
    outputs_val->emplace_back(ap::paddle::GetMetaTensorPtrClass().New(output));
  }
  ADT_RETURN_IF_ERR(
      interpreter.Interpret(lambda, {inputs_val, attrs, outputs_val}));
  return adt::Ok{};
}

template <typename RetT>
using MakeT = adt::Result<RetT> (*)(const std::string& str);

template <typename T, MakeT<T> Make>
adt::Result<T> CacheResult(const std::string& serialized_attributes) {
  static std::unordered_map<std::string, adt::Result<T>> cache;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = cache.find(serialized_attributes);
  if (iter == cache.end()) {
    iter =
        cache.emplace(serialized_attributes, Make(serialized_attributes)).first;
  }
  ADT_LET_CONST_REF(ret, iter->second);
  return ret;
}

adt::Result<Lambda> MakeLambda(const std::string& serialized_attributes) {
  ADT_LET_CONST_REF(anf_expr,
                    ap::axpr::MakeAnfExprFromJsonString(serialized_attributes));
  const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
  ADT_LET_CONST_REF(atomic,
                    core_expr.TryGet<ap::axpr::Atomic<ap::axpr::CoreExpr>>())
      << adt::errors::TypeError{
             std::string() +
             "serialized_attributes can not be converted to atomic AnfExpr."};
  ADT_LET_CONST_REF(lambda,
                    atomic.TryGet<ap::axpr::Lambda<ap::axpr::CoreExpr>>());
  return lambda;
}

constexpr auto CastToLambda = &CacheResult<Lambda, &MakeLambda>;

adt::Result<ap::axpr::AttrMap<ap::axpr::Value>> MakeAttrMap(
    const std::string& serialized_attributes) {
  ADT_LET_CONST_REF(anf_expr,
                    ap::axpr::MakeAnfExprFromJsonString(serialized_attributes));
  const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
  std::vector<ap::axpr::tVar<std::string>> args{};
  ap::axpr::Lambda<ap::axpr::CoreExpr> lambda{args, core_expr};
  ap::memory::Guard guard{};
  ap::axpr::Interpreter interpreter(
      ap::paddle::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(ret, interpreter.Interpret(lambda, {}));
  ADT_LET_CONST_REF(attrs,
                    ret.template CastTo<ap::axpr::AttrMap<ap::axpr::Value>>());
  return attrs;
}

constexpr auto CastToAttrMap =
    &CacheResult<ap::axpr::AttrMap<ap::axpr::Value>, &MakeAttrMap>;

adt::Result<Lambda> MakeInferMetaLambda(
    const std::string& infer_meta_func_name) {
  auto dot_pos = infer_meta_func_name.find('.');
  ADT_CHECK(dot_pos != std::string::npos);
  const auto& module_name = infer_meta_func_name.substr(0, dot_pos);
  const auto& func_name = infer_meta_func_name.substr(dot_pos + 1);
  ADT_CHECK(func_name.find('.') == std::string::npos);
  ap::axpr::LambdaExprBuilder lmd;
  const ap::axpr::AnfExpr anf_expr =
      lmd.Lambda({"inputs", "attrs", "mut_outputs"}, [&](auto& ctx) {
        auto& infer_hooks = ctx.Var("import").Call(ctx.String(module_name));
        auto& method = infer_hooks.Attr(func_name);
        auto& inputs = ctx.Var("inputs");
        auto& attrs = ctx.Var("attrs");
        auto& mut_outputs = ctx.Var("mut_outputs");
        auto& ret = method.Call(inputs, attrs, mut_outputs);
        return ret;
      });
  const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
  const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
  return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
}

constexpr auto GetInferMetaLambda = &CacheResult<Lambda, &MakeInferMetaLambda>;

adt::Result<adt::Ok> InferMetaByAxprHookImpl(
    const std::string& infer_meta_func_name,
    const ::paddle::optional<std::vector<const MetaTensor*>>& inputs,
    const ap::axpr::AttrMap<ap::axpr::Value>& attrs,
    const std::vector<MetaTensor*>& outputs) {
  ADT_LET_CONST_REF(lambda, GetInferMetaLambda(infer_meta_func_name));
  return InferMetaByLambda(lambda, inputs, attrs, outputs);
}

}  // namespace

adt::Result<adt::Ok> ApInferMetaHelper::InferMeta(
    const std::string& serialized_attributes,
    const std::vector<const MetaTensor*>* inputs,
    std::vector<MetaTensor*>* outputs) {
  ADT_LET_CONST_REF(lambda, CastToLambda(serialized_attributes));
  return InferMetaByLambda(lambda, inputs, outputs);
}

adt::Result<adt::Ok> ApInferMetaHelper::InferMetaByAxprHook(
    const ::paddle::optional<std::vector<const MetaTensor*>>& inputs,
    const std::string& infer_meta_func_name,
    const std::string& serialized_attributes,
    const std::vector<MetaTensor*>& outputs) {
  ADT_LET_CONST_REF(attrs, CastToAttrMap(serialized_attributes));
  return InferMetaByAxprHookImpl(infer_meta_func_name, inputs, attrs, outputs);
}

}  // namespace phi

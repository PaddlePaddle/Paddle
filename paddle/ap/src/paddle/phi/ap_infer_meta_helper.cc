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
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/interpreter.h"
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

adt::Result<Lambda> MakeLambda(const std::string& lambda_str) {
  ADT_LET_CONST_REF(anf_expr, ap::axpr::MakeAnfExprFromJsonString(lambda_str));
  const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
  ADT_LET_CONST_REF(atomic,
                    core_expr.TryGet<ap::axpr::Atomic<ap::axpr::CoreExpr>>())
      << adt::errors::TypeError{
             std::string() +
             "lambda_str can not be converted to atomic AnfExpr."};
  ADT_LET_CONST_REF(lambda,
                    atomic.TryGet<ap::axpr::Lambda<ap::axpr::CoreExpr>>());
  return lambda;
}

using MakeLambdaT = adt::Result<Lambda> (*)(const std::string& lambda_str);

template <MakeLambdaT Make>
adt::Result<Lambda> CacheConvertResult(const std::string& lambda_str) {
  static std::unordered_map<std::string, adt::Result<Lambda>> cache;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = cache.find(lambda_str);
  if (iter == cache.end()) {
    iter = cache.emplace(lambda_str, Make(lambda_str)).first;
  }
  ADT_LET_CONST_REF(lambda, iter->second);
  return lambda;
}

constexpr MakeLambdaT CastToLambda = &CacheConvertResult<&MakeLambda>;

}  // namespace

adt::Result<adt::Ok> ApInferMetaHelper::InferMeta(
    const std::string& lambda_str,
    const std::vector<const MetaTensor*>* inputs,
    std::vector<MetaTensor*>* outputs) {
  ADT_LET_CONST_REF(lambda, CastToLambda(lambda_str));
  return InferMetaByLambda(lambda, inputs, outputs);
}

}  // namespace phi

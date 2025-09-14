// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/builtin_serializable_attr_map_to_axpr_helper.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/value.h"

namespace ap::axpr {

struct BuiltinSerializableAttrMapToAxprHelperImpl {
  using AnfExpr = axpr::AnfExpr;

  adt::Result<AnfExpr> Convert(
      ap::axpr::LetContext* ctx,
      const ap::axpr::AttrMap<ap::axpr::SerializableValue>& attr_map) const {
    return GetCodeFromBuiltinSerializableAttrMap(ctx, attr_map);
  }

 private:
  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMap(
      ap::axpr::LetContext* ctx,
      const ap::axpr::AttrMap<ap::axpr::SerializableValue>& attr_map) const {
    std::map<std::string, AnfExpr> kwargs;
    for (const auto& [keyword, val] : attr_map->storage) {
      ADT_LET_CONST_REF(val_anf,
                        GetCodeFromBuiltinSerializableAttrMapItem(ctx, val));
      kwargs[keyword] = val_anf;
    }
    return ctx->Apply("BuiltinSerializableAttrMap", {}, kwargs);
  }

  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMapItem(
      ap::axpr::LetContext* ctx,
      const ap::axpr::SerializableValue& item) const {
    return item.Match(
        [&](const adt::Nothing&) -> adt::Result<AnfExpr> {
          return ctx->None();
        },
        [&](bool c) -> adt::Result<AnfExpr> { return ctx->Bool(c); },
        [&](int64_t c) -> adt::Result<AnfExpr> { return ctx->Int64(c); },
        [&](double c) -> adt::Result<AnfExpr> { return ctx->Double(c); },
        [&](const std::string& str) -> adt::Result<AnfExpr> {
          return ctx->String(str);
        },
        [&](const adt::List<ap::axpr::SerializableValue>& l)
            -> adt::Result<AnfExpr> {
          return GetCodeFromBuiltinSerializableAttrMapList(ctx, l);
        },
        [&](const ap::axpr::AttrMap<ap::axpr::SerializableValue>& object)
            -> adt::Result<AnfExpr> {
          return GetCodeFromBuiltinSerializableAttrMap(ctx, object);
        },
        [&](const ap::axpr::Function<ap::axpr::SerializableValue>& function)
            -> adt::Result<AnfExpr> {
          const auto& lambda = function->lambda;
          const AnfExpr& anf_expr = ap::axpr::ConvertCoreExprToAnfExpr(lambda);
          AnfExpr ret{ctx->Attr(anf_expr, "__function__")};
          return ret;
        },
        [&](const axpr::BuiltinFuncVoidPtr& func) -> adt::Result<AnfExpr> {
          const auto& name_info =
              axpr::BuiltinFuncNameMgr::Singleton()->OptGet(func.func_ptr);
          ADT_CHECK(name_info.has_value());
          if (name_info.value().module_name.has_value()) {
            const auto& module_name =
                ctx->String(name_info.value().module_name.value());
            const auto& func_name = name_info.value().func_name;
            return ctx->Var("import").Call(module_name).Attr(func_name);
          } else {
            const auto& func_name = name_info.value().func_name;
            return ctx->Var(func_name);
          }
        },
        [&](const axpr::BuiltinHighOrderFuncVoidPtr& func)
            -> adt::Result<AnfExpr> {
          const auto& name_info =
              axpr::BuiltinFuncNameMgr::Singleton()->OptGet(func.func_ptr);
          ADT_CHECK(name_info.has_value());
          if (name_info.value().module_name.has_value()) {
            const auto& module_name =
                ctx->String(name_info.value().module_name.value());
            const auto& func_name = name_info.value().func_name;
            return ctx->Var("import").Call(module_name).Attr(func_name);
          } else {
            const auto& func_name = name_info.value().func_name;
            return ctx->Var(func_name);
          }
        },
        [&](const TypeImplBuiltinClassInstance& impl) -> adt::Result<AnfExpr> {
          auto* ptr =
              reinterpret_cast<const ClassOps<axpr::Value>*>(impl.class_ops);
          return ctx->Var(ptr->class_attrs()->Name());
        },
        [&](const TypeImplClassInstance<SerializableValue>&)
            -> adt::Result<AnfExpr> {
          return adt::errors::NotImplementedError{
              "serialization of axpr::ClassAttrs<SerializableValueT> not "
              "implemented"};
        },
        [&](const auto& impl) -> adt::Result<AnfExpr> {
          return ctx->Var(
              impl.template CastToAxprType<axpr::Value, SerializableValue>()
                  .Name());
        });
  }

  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMapList(
      ap::axpr::LetContext* ctx,
      const adt::List<ap::axpr::SerializableValue>& list) const {
    std::vector<AnfExpr> elt_anf_exprs;
    for (const auto& elt : *list) {
      ADT_LET_CONST_REF(elt_anf_expr,
                        GetCodeFromBuiltinSerializableAttrMapItem(ctx, elt));
      elt_anf_exprs.emplace_back(elt_anf_expr);
    }
    return ctx->Call(ap::axpr::kBuiltinList(), elt_anf_exprs);
  }
};

adt::Result<AnfExpr> BuiltinSerializableAttrMapToAxprHelper::Convert(
    ap::axpr::LetContext* ctx,
    const ap::axpr::AttrMap<ap::axpr::SerializableValue>& attr_map) const {
  return BuiltinSerializableAttrMapToAxprHelperImpl{}.Convert(ctx, attr_map);
}

}  // namespace ap::axpr

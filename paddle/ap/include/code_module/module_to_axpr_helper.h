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
#include "paddle/ap/include/axpr/builtin_serializable_attr_map_to_axpr_helper.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/code_module/code_module.h"

namespace ap::code_module {

struct ModuleToAxprHelper {
  using AnfExpr = axpr::AnfExpr;

  adt::Result<AnfExpr> ConvertModuleToAnfExpr(axpr::LetContext* ctx,
                                              const CodeModule& m) const {
    return ConvertModuleToAnfExprImpl(ctx, m);
  }

  adt::Result<AnfExpr> ConvertModuleToAnfExpr(const CodeModule& m) const {
    auto ConstructLambdaBody = [&](auto& ctx) -> adt::Result<AnfExpr> {
      return ConvertModuleToAnfExprImpl(&ctx, m);
    };
    return ap::axpr::LambdaExprBuilder{}.TryLambda({}, ConstructLambdaBody);
  }

 private:
  adt::Result<AnfExpr> ConvertModuleToAnfExprImpl(axpr::LetContext* ctx,
                                                  const CodeModule& m) const {
    auto ConvertArgType = [&](auto& ctx, const auto& arg_type) -> AnfExpr {
      return arg_type.Match(
          [&](const ap::axpr::DataType& data_type) -> AnfExpr {
            const auto& var = ctx->Var("DataType").Attr(data_type.Name());
            return ap::axpr::tVar<std::string>{var.name()};
          },
          [&](const ap::axpr::PointerType& pointer_type) -> AnfExpr {
            const auto& var = ctx->Var("PointerType").Attr(pointer_type.Name());
            return ap::axpr::tVar<std::string>{var.name()};
          });
    };
    auto ConvertFuncDeclareCall = [&](auto& ctx,
                                      const auto& func_declare) -> AnfExpr {
      const auto& ret_val_anf_expr =
          ConvertArgType(ctx, func_declare->ret_type);
      const auto& func_name = ctx->String(func_declare->func_id);
      std::vector<AnfExpr> elts;
      elts.reserve(func_declare->arg_types->size());
      for (const auto& arg_type : *func_declare->arg_types) {
        elts.emplace_back(ConvertArgType(ctx, arg_type));
      }
      const auto& arg_type_anf_expr = ctx->Call(ap::axpr::kBuiltinList(), elts);
      return ctx->Call(
          "FuncDeclare", ret_val_anf_expr, func_name, arg_type_anf_expr);
    };
    auto ConvertFuncDeclareList = [&](auto& ctx) -> AnfExpr {
      std::vector<AnfExpr> elts;
      elts.reserve(m->func_declares->size());
      for (const auto& func_declare : *m->func_declares) {
        elts.emplace_back(ConvertFuncDeclareCall(ctx, func_declare));
      }
      return ctx->Call(ap::axpr::kBuiltinList(), elts);
    };
    auto ConvertSourceCodeConstruction =
        [&](auto* ctx) -> adt::Result<AnfExpr> {
      return m->source_code.Match(
          [&](const ap::code_module::Project& project) -> adt::Result<AnfExpr> {
            return ConvertProjectConstruct(ctx, project);
          },
          [&](const ap::code_module::Package& package) -> adt::Result<AnfExpr> {
            return ConvertPackageConstruct(ctx, package);
          });
    };
    const auto& declare = ConvertFuncDeclareList(ctx);
    ADT_LET_CONST_REF(source_code, ConvertSourceCodeConstruction(ctx));
    return ctx->Call("CodeModule", declare, source_code);
  }

  adt::Result<AnfExpr> ConvertProjectConstruct(
      ap::axpr::LetContext* ctx,
      const ap::code_module::Project& project) const {
    const auto& attrs = project->others;
    ADT_LET_CONST_REF(others_anf_expr,
                      GetCodeFromBuiltinSerializableAttrMap(ctx, attrs));
    std::map<std::string, AnfExpr> kwargs{
        {"nested_files", ConvertProjectNestedFiles(ctx, project->nested_files)},
        {"compile_cmd", AnfExpr{ctx->String(project->compile_cmd)}},
        {"so_relative_path", AnfExpr{ctx->String(project->so_relative_path)}},
        {"others", others_anf_expr},
    };
    return ctx->Apply("Project", {}, kwargs);
  }

  adt::Result<AnfExpr> ConvertPackageConstruct(
      ap::axpr::LetContext* ctx,
      const ap::code_module::Package& package) const {
    const auto& attrs = package->others;
    ADT_LET_CONST_REF(others_anf_expr,
                      GetCodeFromBuiltinSerializableAttrMap(ctx, attrs));
    const auto& api_so_path = package->api_wrapper_so_relative_path;
    const auto& main_so_path = package->main_so_relative_path;
    std::map<std::string, AnfExpr> kwargs{
        {"nested_files", ConvertProjectNestedFiles(ctx, package->nested_files)},
        {"api_wrapper_so_relative_path", AnfExpr{ctx->String(api_so_path)}},
        {"main_so_relative_path", AnfExpr{ctx->String(main_so_path)}},
        {"others", others_anf_expr},
    };
    return ctx->Apply("Package", {}, kwargs);
  }

  AnfExpr ConvertProjectNestedFiles(ap::axpr::LetContext* ctx,
                                    const ap::code_module::File& file) const {
    return file.Match(
        [&](const ap::code_module::FileContent& file_content) -> AnfExpr {
          const auto& str = file_content->file_content;
          return ctx->Var("Project").Attr("FileContent").Call(ctx->String(str));
        },
        [&](const ap::code_module::SoftLink& soft_link) -> AnfExpr {
          const auto& str = soft_link->target_relative_path;
          return ctx->Var("Project").Attr("SoftLink").Call(ctx->String(str));
        },
        [&](const ap::code_module::Directory<ap::code_module::File>& dir)
            -> AnfExpr {
          std::vector<AnfExpr> args;
          for (const auto& [k, v] : dir.dentry2file->storage) {
            const auto& v_anf_expr = ConvertProjectNestedFiles(ctx, v);
            args.emplace_back(ctx->Call(
                ap::axpr::kBuiltinList(), ctx->String(k), v_anf_expr));
          }
          return ctx->Apply(ctx->Var("Project").Attr("Directory"), args);
        });
  }

  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMap(
      ap::axpr::LetContext* ctx,
      const ap::axpr::AttrMap<ap::axpr::SerializableValue>& attr_map) const {
    return axpr::BuiltinSerializableAttrMapToAxprHelper{}.Convert(ctx,
                                                                  attr_map);
  }
};

}  // namespace ap::code_module

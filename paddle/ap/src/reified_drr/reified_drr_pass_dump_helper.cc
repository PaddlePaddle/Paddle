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

#include "paddle/ap/include/reified_drr/reified_drr_pass_dump_helper.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/code_module/module_compile_helper.h"
#include "paddle/ap/include/fs/fs.h"
#include "paddle/ap/include/reified_drr/reified_res_ptn_axpr_maker.h"
#include "paddle/ap/include/reified_drr/reified_src_ptn_axpr_maker.h"

namespace ap::reified_drr {

struct ReifiedDrrPassDumpHelperImpl {
  drr::DrrCtx abstract_drr_ctx_;
  DrrNodeAttrToAnfExprHelper* attr2axpr_helper_;
  MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper_;
  std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
      const std::string&)>
      CodeGenResult4FusedOpName_;
  int64_t nice_;

  struct DumpCtx {
    std::optional<std::string> dump_dir;
    std::optional<axpr::AnfExpr> reified_drr_pass_class_lambda_anf_expr;
  };

  // Returns reified drr_pass_class lambda
  adt::Result<axpr::AnfExpr> Dump() {
    DumpCtx dump_ctx;
    ADT_LET_CONST_REF(anf_expr, ConvertToModuleAnfExpr(&dump_ctx));
    if (dump_ctx.dump_dir.has_value()) {
      const auto& reified_drr_json = anf_expr.DumpToJsonString();
      const auto& reified_drr_json_path =
          dump_ctx.dump_dir.value() + "/reified_drr.json";
      ADT_RETURN_IF_ERR(
          fs::WriteFileContent(reified_drr_json_path, reified_drr_json));
    }
    ADT_CHECK(dump_ctx.reified_drr_pass_class_lambda_anf_expr.has_value());
    return dump_ctx.reified_drr_pass_class_lambda_anf_expr.value();
  }

  adt::Result<axpr::AnfExpr> ConvertToModuleAnfExpr(DumpCtx* dump_ctx) {
    axpr::LambdaExprBuilder lmd;
    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      ADT_RETURN_IF_ERR(DefineAxprModule(&ctx, dump_ctx));
      return ctx.None();
    };
    return lmd.TryLet(GetBody);
  }

  adt::Result<adt::Ok> DefineAxprModule(axpr::LetContext* ctx,
                                        DumpCtx* dump_ctx) {
    ADT_LET_CONST_REF(make_drr_ctx_anf_expr, DefineMakeDrrCtxLambda(dump_ctx));
    ADT_LET_CONST_REF(drr_pass_class_anf_expr,
                      DefineDrrPassClass(ctx, make_drr_ctx_anf_expr));
    auto GetBody = [&](auto& let_ctx) -> adt::Result<axpr::AnfExpr> {
      return DefineDrrPassClass(&let_ctx, make_drr_ctx_anf_expr);
    };
    ADT_LET_CONST_REF(lambda, axpr::LambdaExprBuilder{}.TryLambda({}, GetBody));
    dump_ctx->reified_drr_pass_class_lambda_anf_expr = lambda;
    ADT_RETURN_IF_ERR(
        InsertRegisterReifiedDrrPass(ctx, drr_pass_class_anf_expr));
    return adt::Ok{};
  }

  adt::Result<axpr::AnfExpr> DefineMakeDrrCtxLambda(DumpCtx* dump_ctx) {
    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      auto& drr_ctx = ctx.Var("DrrCtx").Call();
      ADT_LET_CONST_REF(src_ptn_func, DefineSourcePatternFunc());
      ADT_LET_CONST_REF(constraint_lambda, DefineConstraintLambda());
      ADT_LET_CONST_REF(res_ptn_func,
                        DefineOrGetResultPatternFunc(
                            src_ptn_func, constraint_lambda, dump_ctx));
      drr_ctx.Attr("set_drr_pass_type")
          .Call(ctx.String("reified_drr_pass_type"));
      drr_ctx.Attr("init_source_pattern").Call(src_ptn_func);
      const auto& constraint_func_name = ctx.NewTmpVarName();
      ctx.Var(constraint_func_name) = constraint_lambda;
      const auto& constraint_func =
          ctx.Var(constraint_func_name).Attr("__function__");
      drr_ctx.Attr("init_constraint_func").Call(constraint_func);
      drr_ctx.Attr("init_result_pattern").Call(res_ptn_func);
      return drr_ctx;
    };
    return axpr::LambdaExprBuilder{}.TryLambda({"self"}, GetBody);
  }

  adt::Result<axpr::AnfExpr> DefineSourcePatternFunc() {
    ADT_CHECK(abstract_drr_ctx_->source_pattern_ctx.has_value());
    ADT_CHECK(abstract_drr_ctx_->source_pattern_ctx.value() ==
              matched_src_ptn_ctx_helper_->src_ptn_ctx());
    ReifiedSrcPtnAxprMaker maker{attr2axpr_helper_,
                                 matched_src_ptn_ctx_helper_};
    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      auto* op_pattern_ctx = &ctx.Var("o");
      auto* tensor_pattern_ctx = &ctx.Var("t");
      ADT_RETURN_IF_ERR(maker.GenAnfExprForSrcPtnCtxOps(op_pattern_ctx));
      ADT_RETURN_IF_ERR(maker.GenAnfExprForSrcPtnCtxValues(tensor_pattern_ctx));
      ADT_RETURN_IF_ERR(maker.GenAnfExprForSrcPtnCtxOpValueConnections(
          op_pattern_ctx, tensor_pattern_ctx));
      return ctx.None();
    };
    return axpr::LambdaExprBuilder{}.TryLambda({"o", "t"}, GetBody);
  }

  adt::Result<axpr::AnfExpr> DefineConstraintLambda() {
    axpr::LambdaExprBuilder lmbd;
    return lmbd.Lambda({"o", "t", "ir_helper"},
                       [&](auto& ctx) { return ctx.Bool(true); });
  }

  adt::Result<axpr::AnfExpr> DefineOrGetResultPatternFunc(
      const axpr::AnfExpr& src_ptn_func,
      const axpr::AnfExpr& constraint_func,
      DumpCtx* dump_ctx) {
    std::string src_ptn_func_json = src_ptn_func.DumpToJsonString();
    std::string constraint_func_json = constraint_func.DumpToJsonString();
    std::hash<std::string> str_hash{};
    std::size_t pattern_hash_value = adt::hash_combine(
        str_hash(src_ptn_func_json), str_hash(constraint_func_json));
    ADT_CHECK(abstract_drr_ctx_->pass_name.has_value());
    const std::string relative_dump_dir =
        DecodeIntoDirectoryName(abstract_drr_ctx_->pass_name.value()) + "_" +
        std::to_string(pattern_hash_value);
    ADT_LET_CONST_REF(dump_root_dir, GetDumpDir());
    const std::string& src_ptn_func_json_path =
        dump_root_dir + "/" + relative_dump_dir + "/source_pattern_func.json";
    const std::string& constraint_func_json_path =
        dump_root_dir + "/" + relative_dump_dir + "/constraint_func.json";
    const std::string& res_ptn_func_json_path =
        dump_root_dir + "/" + relative_dump_dir + "/result_pattern_func.json";
    if (fs::FileExists(res_ptn_func_json_path)) {
      std::string old_src_ptn_func_json;
      ADT_RETURN_IF_ERR(
          fs::ReadFileContent(src_ptn_func_json_path, &old_src_ptn_func_json));
      std::string old_constraint_func_json;
      ADT_RETURN_IF_ERR(fs::ReadFileContent(constraint_func_json_path,
                                            &old_constraint_func_json));
      ADT_CHECK(old_src_ptn_func_json == src_ptn_func_json);
      ADT_CHECK(old_constraint_func_json == constraint_func_json);
      std::string res_ptn_func_json;
      ADT_RETURN_IF_ERR(
          fs::ReadFileContent(res_ptn_func_json_path, &res_ptn_func_json));
      ADT_LET_CONST_REF(res_ptn_func,
                        axpr::MakeAnfExprFromJsonString(res_ptn_func_json));
      return res_ptn_func;
    } else {
      dump_ctx->dump_dir = dump_root_dir + "/" + relative_dump_dir;
      code_module::ModuleCompileHelper compile_helper{dump_root_dir,
                                                      relative_dump_dir};
      using RetT = adt::Result<code_gen::CodeGenResult<axpr::Value>>;
      auto CodeGenResult4FusedOpName =
          [&](const std::string& op_unique_name) -> RetT {
        ADT_LET_CONST_REF(code_gen_result,
                          CodeGenResult4FusedOpName_(op_unique_name));
        ADT_LET_CONST_REF(package_module,
                          compile_helper.CompileProjectModuleToPackageModule(
                              code_gen_result->code_module));
        return code_gen::CodeGenResult<axpr::Value>{
            package_module,
            code_gen_result->kernel_dispatch_func,
            code_gen_result->kernel_dispatch_const_data};
      };
      ADT_LET_CONST_REF(res_ptn_func,
                        DefineResultPatternFunc(CodeGenResult4FusedOpName));
      const auto& res_ptn_func_json = res_ptn_func.DumpToJsonString();
      ADT_RETURN_IF_ERR(
          fs::WriteFileContent(src_ptn_func_json_path, src_ptn_func_json));
      ADT_RETURN_IF_ERR(fs::WriteFileContent(constraint_func_json_path,
                                             constraint_func_json));
      ADT_RETURN_IF_ERR(
          fs::WriteFileContent(res_ptn_func_json_path, res_ptn_func_json));
      return res_ptn_func;
    }
  }

  std::string DecodeIntoDirectoryName(std::string str) {
    for (size_t i = 0; i < str.size(); ++i) {
      if (str.at(i) >= 'A' && str.at(i) <= 'Z') {
        continue;
      }
      if (str.at(i) >= 'a' && str.at(i) <= 'z') {
        continue;
      }
      if (str.at(i) >= '0' && str.at(i) <= '9') {
        continue;
      }
      str[i] = '_';
    }
    return str;
  }

  adt::Result<std::string> GetDumpDir() {
    const char* dump_dir = std::getenv("AP_PACKAGE_DUMP_DIR");
    ADT_CHECK(dump_dir != nullptr);
    return std::string(dump_dir);
  }

  adt::Result<axpr::AnfExpr> DefineResultPatternFunc(
      const std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
          const std::string&)>& CodeGenResult4FusedOpName) {
    ADT_CHECK(abstract_drr_ctx_->result_pattern_ctx.has_value());
    ReifiedResPtnAxprMaker maker(abstract_drr_ctx_->result_pattern_ctx.value(),
                                 CodeGenResult4FusedOpName);
    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      auto* op_pattern_ctx = &ctx.Var("o");
      auto* tensor_pattern_ctx = &ctx.Var("t");
      ADT_RETURN_IF_ERR(maker.GenAnfExprForResPtnCtxOps(op_pattern_ctx));
      ADT_RETURN_IF_ERR(maker.GenAnfExprForResPtnCtxOpValueConnections(
          op_pattern_ctx, tensor_pattern_ctx));
      return ctx.None();
    };
    return axpr::LambdaExprBuilder{}.TryLambda({"o", "t"}, GetBody);
  }

  adt::Result<axpr::AnfExpr> DefineDrrPassClass(
      axpr::LetContext* ctx, const axpr::AnfExpr& make_drr_lambda) {
    const auto& class_name = ctx->String(std::string("ReifiedDrrPass"));
    const auto& superclasses = ctx->Var(axpr::kBuiltinList()).Call();
    const auto& make_drr_func_name = ctx->NewTmpVarName();
    ctx->Var(make_drr_func_name) = make_drr_lambda;
    const auto& methods = [&] {
      std::vector<axpr::AnfExpr> args{};
      std::map<std::string, axpr::AnfExpr> kwargs{
          {"make_drr_ctx", ctx->Var(make_drr_func_name).Attr("__function__")}};
      return ctx->Var("BuiltinSerializableAttrMap").Apply(args, kwargs);
    }();
    return ctx->Var("type").Call(class_name, superclasses, methods);
  }

  adt::Result<adt::Ok> InsertRegisterReifiedDrrPass(
      axpr::LetContext* ctx, const axpr::AnfExpr& drr_pass_class) {
    ADT_LET_CONST_REF(pass_name_val, GetPassName(drr_pass_class));
    const auto& pass_name = ctx->String(pass_name_val);
    const auto& nice = ctx->Int64(nice_);
    ctx->Var("Registry")
        .Attr("classic_drr_pass")
        .Call(pass_name, nice, drr_pass_class);
    return adt::Ok{};
  }

  adt::Result<std::string> GetPassName(const axpr::AnfExpr& drr_pass_class) {
    ADT_CHECK(abstract_drr_ctx_->pass_name.has_value());
    std::size_t hash_value = GetHashValue(drr_pass_class);
    return abstract_drr_ctx_->pass_name.value() + "_reified_" +
           std::to_string(hash_value);
  }

  std::size_t GetHashValue(const axpr::AnfExpr& anf_expr) {
    const auto& serialized = anf_expr.DumpToJsonString();
    return std::hash<std::string>()(serialized);
  }
};

bool ReifiedDrrPassDumpHelper::DumpEnabled() {
  return std::getenv("AP_PACKAGE_DUMP_DIR") != nullptr;
}

// Returns reified drr_pass_class lambda
adt::Result<axpr::AnfExpr> ReifiedDrrPassDumpHelper::Dump(
    const drr::DrrCtx& abstract_drr_ctx,
    DrrNodeAttrToAnfExprHelper* attr2axpr_helper,
    MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper,
    const std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
        const std::string&)>& CodeGenResult4FusedOpName,
    int64_t nice) const {
  ReifiedDrrPassDumpHelperImpl impl{abstract_drr_ctx,
                                    attr2axpr_helper,
                                    matched_src_ptn_ctx_helper,
                                    CodeGenResult4FusedOpName,
                                    nice};
  return impl.Dump();
}

}  // namespace ap::reified_drr

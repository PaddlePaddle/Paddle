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

#include "paddle/ap/include/axpr/core_expr_util.h"
#include <unordered_map>
#include <unordered_set>

namespace ap::axpr {

namespace {

struct SimplifyTmpVarNameCtx {
  const std::string tmp_var_name_prefix{"___"};
  std::unordered_map<std::string, std::string> old_var_name2new_var_name;

  bool IsTmpVarName(const std::string& var_name) const {
    return var_name.size() >= 3 &&
           var_name.substr(0, 3) == this->tmp_var_name_prefix;
  }

  std::string ReplaceVarName(const std::string& var_name) {
    if (!IsTmpVarName(var_name)) return var_name;
    const auto& iter = this->old_var_name2new_var_name.find(var_name);
    if (iter != this->old_var_name2new_var_name.end()) return iter->second;
    std::size_t seq_no = this->old_var_name2new_var_name.size();
    std::string new_var_name = tmp_var_name_prefix + std::to_string(seq_no);
    this->old_var_name2new_var_name[var_name] = new_var_name;
    return new_var_name;
  }
};

axpr::CoreExpr SimplifyCoreExprTmpVarName(const axpr::CoreExpr& core_expr,
                                          SimplifyTmpVarNameCtx* ctx);

axpr::tVar<std::string> SimplifyVarCoreExprTmpVarName(
    const axpr::tVar<std::string>& var, SimplifyTmpVarNameCtx* ctx) {
  return axpr::tVar<std::string>{ctx->ReplaceVarName(var.value())};
}

axpr::Symbol SimplifySymbolCoreExprTmpVarName(const axpr::Symbol& symbol,
                                              SimplifyTmpVarNameCtx* ctx) {
  return symbol.Match(
      [&](const axpr::tVar<std::string>& var) -> axpr::Symbol {
        return SimplifyVarCoreExprTmpVarName(var, ctx);
      },
      [&](const builtin_symbol::Symbol&) -> axpr::Symbol { return symbol; });
}

axpr::Lambda<axpr::CoreExpr> SimplifyLambdaCoreExprTmpVarName(
    const axpr::Lambda<axpr::CoreExpr>& lambda, SimplifyTmpVarNameCtx* ctx) {
  std::vector<axpr::tVar<std::string>> args;
  args.reserve(lambda->args.size());
  for (const auto& arg : lambda->args) {
    args.emplace_back(SimplifyVarCoreExprTmpVarName(arg, ctx));
  }
  const auto& body = SimplifyCoreExprTmpVarName(lambda->body, ctx);
  return axpr::Lambda<axpr::CoreExpr>{args, body};
}

axpr::Atomic<axpr::CoreExpr> SimplifyAtomicCoreExprTmpVarName(
    const axpr::Atomic<axpr::CoreExpr>& atomic, SimplifyTmpVarNameCtx* ctx) {
  return atomic.Match(
      [&](const axpr::Symbol& symbol) -> axpr::Atomic<axpr::CoreExpr> {
        return SimplifySymbolCoreExprTmpVarName(symbol, ctx);
      },
      [&](const axpr::Lambda<axpr::CoreExpr>& lambda)
          -> axpr::Atomic<axpr::CoreExpr> {
        return SimplifyLambdaCoreExprTmpVarName(lambda, ctx);
      },
      [&](const auto&) -> axpr::Atomic<axpr::CoreExpr> { return atomic; });
}

axpr::CoreExpr SimplifyCoreExprTmpVarNameImpl(
    const axpr::Atomic<axpr::CoreExpr>& atomic, SimplifyTmpVarNameCtx* ctx) {
  return SimplifyAtomicCoreExprTmpVarName(atomic, ctx);
}

axpr::CoreExpr SimplifyCoreExprTmpVarNameImpl(
    const axpr::ComposedCallAtomic<axpr::CoreExpr>& call,
    SimplifyTmpVarNameCtx* ctx) {
  const auto inner_func =
      SimplifyAtomicCoreExprTmpVarName(call->inner_func, ctx);
  std::vector<axpr::Atomic<axpr::CoreExpr>> args;
  args.reserve(call->args.size());
  for (const auto& arg : call->args) {
    args.emplace_back(SimplifyAtomicCoreExprTmpVarName(arg, ctx));
  }
  const auto outer_func =
      SimplifyAtomicCoreExprTmpVarName(call->outer_func, ctx);
  return axpr::ComposedCallAtomic<axpr::CoreExpr>{outer_func, inner_func, args};
}

axpr::CoreExpr SimplifyCoreExprTmpVarName(const axpr::CoreExpr& core_expr,
                                          SimplifyTmpVarNameCtx* ctx) {
  return core_expr.Match([&](const auto& impl) {
    return SimplifyCoreExprTmpVarNameImpl(impl, ctx);
  });
}

}  // namespace

axpr::CoreExpr SimplifyTmpVarName(const axpr::CoreExpr& core_expr) {
  SimplifyTmpVarNameCtx ctx{};
  return SimplifyCoreExprTmpVarName(core_expr, &ctx);
}

}  // namespace ap::axpr

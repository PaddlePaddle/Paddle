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

#include <iomanip>
#include <sstream>
#include "paddle/ap/include/axpr/anf_expr.h"

namespace ap::axpr {

struct AnfExprHelper {
  adt::Result<std::string> FunctionToString(
      const Lambda<AnfExpr>& lambda) const {
    return FunctionToString("unnamed", lambda);
  }

 private:
  adt::Result<std::string> FunctionToString(
      const std::string& func_name, const Lambda<AnfExpr>& lambda) const {
    std::ostringstream ss;
    auto Generate = [&](const std::string& str) { ss << str << "\n"; };
    ADT_RETURN_IF_ERR(SerializeFunction(Generate, func_name, lambda));
    return ss.str();
  }

  adt::Result<adt::Ok> SerializeFunction(
      const std::function<void(const std::string&)>& Generate,
      const std::string& func_name,
      const Lambda<AnfExpr>& lambda) const {
    {
      std::ostringstream ss;
      ss << "def " << func_name << "(";
      int i = 0;
      for (const auto& arg : lambda->args) {
        if (i++ > 0) {
          ss << ", ";
        }
        ss << arg.value();
      }
      ss << "):";
      Generate(ss.str());
    }
    {
      auto BodyGenerate = [&](const std::string& str) {
        Generate(std::string("    ") + str);
      };
      ADT_RETURN_IF_ERR(SerializeLastExprInLambda(BodyGenerate, lambda->body));
    }
    return adt::Ok{};
  }

  struct LambdaBodySerializeCtx {
    std::function<void(const std::string&)> Generate;

    std::size_t auto_id_in_body = 0;

    std::size_t GetAutoIdInBody() { return auto_id_in_body++; }
  };

  adt::Result<adt::Ok> SerializeLastExprInLambda(
      const std::function<void(const std::string&)>& Generate,
      const AnfExpr& lambda_body) const {
    LambdaBodySerializeCtx ctx{Generate};
    return SerializeLastExprInLambda(&ctx, lambda_body);
  }

  adt::Result<adt::Ok> SerializeLastExprInLambda(
      LambdaBodySerializeCtx* ctx, const AnfExpr& lambda_body) const {
    return lambda_body.Match([&](const auto& impl) -> adt::Result<adt::Ok> {
      return SerializeLastExprInLambdaImpl(ctx, impl);
    });
  }

  adt::Result<adt::Ok> SerializeLastExprInLambdaImpl(
      LambdaBodySerializeCtx* ctx, const Atomic<AnfExpr>& atomic) const {
    ADT_LET_CONST_REF(
        atomic_str,
        atomic.Match([&](const auto& impl) -> adt::Result<std::string> {
          return AtomicToStringImpl(ctx, impl);
        }));
    ctx->Generate(std::string() + "return " + atomic_str);
    return adt::Ok{};
  }

  adt::Result<std::string> AtomicToString(LambdaBodySerializeCtx* ctx,
                                          const Atomic<AnfExpr>& atomic) const {
    return atomic.Match([&](const auto& impl) -> adt::Result<std::string> {
      return AtomicToStringImpl(ctx, impl);
    });
  }

  adt::Result<adt::Ok> SerializeLastExprInLambdaImpl(
      LambdaBodySerializeCtx* ctx, const Combined<AnfExpr>& combined) const {
    ADT_LET_CONST_REF(
        combined_str,
        combined.Match([&](const auto& impl) -> adt::Result<std::string> {
          return CombinedToStringImpl(ctx, impl);
        }));
    ctx->Generate(std::string() + "return " + combined_str);
    return adt::Ok{};
  }

  adt::Result<std::string> CombinedToString(
      LambdaBodySerializeCtx* ctx, const Combined<AnfExpr>& combined) const {
    return combined.Match([&](const auto& impl) -> adt::Result<std::string> {
      return CombinedToStringImpl(ctx, impl);
    });
  }

  adt::Result<adt::Ok> SerializeLastExprInLambdaImpl(
      LambdaBodySerializeCtx* ctx, const Let<AnfExpr>& let) const {
    for (const auto& [var, combined] : let->bindings) {
      ADT_LET_CONST_REF(combined_str, CombinedToString(ctx, combined));
      ctx->Generate(var.value() + " = " + combined_str);
    }
    return SerializeLastExprInLambda(ctx, let->body);
  }

  adt::Result<std::string> CombinedToStringImpl(
      LambdaBodySerializeCtx* ctx, const Call<AnfExpr>& call) const {
    std::ostringstream ss;
    ADT_LET_CONST_REF(func_name, AtomicToString(ctx, call->func));
    ss << func_name << "(";
    int i = 0;
    for (const auto& arg : call->args) {
      if (i++ > 0) {
        ss << ", ";
      }
      ADT_LET_CONST_REF(arg_str, AtomicToString(ctx, arg));
      ss << arg_str;
    }
    ss << ")";
    return ss.str();
  }

  adt::Result<std::string> CombinedToStringImpl(
      LambdaBodySerializeCtx* ctx, const If<AnfExpr>& if_expr) const {
    std::ostringstream ss;
    ADT_LET_CONST_REF(cond, AtomicToString(ctx, if_expr->cond));
    ADT_LET_CONST_REF(true_expr, AnfExprToString(ctx, if_expr->true_expr));
    ADT_LET_CONST_REF(false_expr, AnfExprToString(ctx, if_expr->false_expr));
    ss << true_expr << " if " << cond << " else " << false_expr;
    return ss.str();
  }

  adt::Result<std::string> AnfExprToString(LambdaBodySerializeCtx* ctx,
                                           const AnfExpr& anf_expr) const {
    return anf_expr.Match(
        [&](const Atomic<AnfExpr>& atomic) -> adt::Result<std::string> {
          return AtomicToString(ctx, atomic);
        },
        [&](const Combined<AnfExpr>& combined) -> adt::Result<std::string> {
          return CombinedToString(ctx, combined);
        },
        [&](const Let<AnfExpr>&) -> adt::Result<std::string> {
          return adt::errors::TypeError{
              "Let is not supported in AnfExprToString()."};
        });
  }

  adt::Result<std::string> AtomicToStringImpl(LambdaBodySerializeCtx* ctx,
                                              const adt::Nothing&) const {
    return std::string("None");
  }

  adt::Result<std::string> AtomicToStringImpl(LambdaBodySerializeCtx* ctx,
                                              bool c) const {
    return std::string(c ? "True" : "False");
  }

  adt::Result<std::string> AtomicToStringImpl(LambdaBodySerializeCtx* ctx,
                                              int64_t c) const {
    return std::to_string(c);
  }

  adt::Result<std::string> AtomicToStringImpl(LambdaBodySerializeCtx* ctx,
                                              double c) const {
    return std::to_string(c);
  }

  adt::Result<std::string> AtomicToStringImpl(LambdaBodySerializeCtx* ctx,
                                              const std::string& str) const {
    std::ostringstream ss;
    ss << std::quoted(str);
    return ss.str();
  }

  adt::Result<std::string> AtomicToStringImpl(
      LambdaBodySerializeCtx* ctx, const tVar<std::string>& var) const {
    return var.value();
  }

  adt::Result<std::string> AtomicToStringImpl(
      LambdaBodySerializeCtx* ctx, const Lambda<AnfExpr>& lambda) const {
    const auto& func_name =
        std::string("tmp_func_") + std::to_string(ctx->GetAutoIdInBody());
    ADT_RETURN_IF_ERR(SerializeFunction(ctx->Generate, func_name, lambda));
    return func_name;
  }
};

}  // namespace ap::axpr

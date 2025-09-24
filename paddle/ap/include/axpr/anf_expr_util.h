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

#include <atomic>
#include "glog/logging.h"
#include "nlohmann/json.hpp"
#include "paddle/ap/include/axpr/anf_expr.h"
#include "paddle/ap/include/axpr/anf_expr_builder.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/core_expr_builder.h"
#include "paddle/common/enforce.h"

namespace ap::axpr {

AnfExpr ConvertCoreExprToAnfExpr(const CoreExpr& core_expr);

namespace detail {

struct CoreExprToAnfExprConverter {
  AnfExpr ConvertCoreExprToAnfExpr(const CoreExpr& core_expr) {
    return core_expr.Match(
        [&](const Atomic<CoreExpr>& atomic) -> AnfExpr {
          return ConvertAtomic(atomic);
        },
        [&](const ComposedCall<Atomic<CoreExpr>>& composed_call) -> AnfExpr {
          return ConvertComposedCall(composed_call);
        });
  }

 private:
  Atomic<AnfExpr> ConvertAtomic(const Atomic<CoreExpr>& atomic) {
    return atomic.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Atomic<AnfExpr> {
          return ConvertLambda(lambda);
        },
        [&](const Symbol& symbol) -> Atomic<AnfExpr> {
          return symbol.Match(
              [&](const tVar<std::string>& var) -> Atomic<AnfExpr> {
                return Atomic<AnfExpr>{var};
              },
              [&](const builtin_symbol::Symbol& symbol) -> Atomic<AnfExpr> {
                tVar<std::string> var{symbol.Name()};
                return Atomic<AnfExpr>{var};
              });
        },
        [&](adt::Nothing) -> Atomic<AnfExpr> {
          return Atomic<AnfExpr>{adt::Nothing{}};
        },
        [&](bool c) -> Atomic<AnfExpr> { return Atomic<AnfExpr>{c}; },
        [&](int64_t c) -> Atomic<AnfExpr> { return Atomic<AnfExpr>{c}; },
        [&](double c) -> Atomic<AnfExpr> { return Atomic<AnfExpr>{c}; },
        [&](const std::string& val) -> Atomic<AnfExpr> {
          return Atomic<AnfExpr>{val};
        });
  }

  Atomic<AnfExpr> ConvertLambda(const Lambda<CoreExpr>& lambda) {
    return Lambda<AnfExpr>{lambda->args,
                           ConvertCoreExprToAnfExpr(lambda->body)};
  }

  AnfExpr ConvertComposedCall(
      const ComposedCall<Atomic<CoreExpr>>& composed_call) {
    const auto& outer_func = composed_call->outer_func;
    return outer_func.Match(
        [&](const Lambda<CoreExpr>& lambda) -> AnfExpr {
          std::vector<Bind<AnfExpr>> bindings;
          return ConvertComposedCallToLet(composed_call, &bindings);
        },
        [&](const Symbol& symbol) -> AnfExpr {
          return symbol.Match(
              [&](const tVar<std::string>& var) -> AnfExpr {
                CHECK_EQ(var.value(), kBuiltinReturn());
                return ConvertComposedCallToCombined(composed_call);
              },
              [&](const builtin_symbol::Symbol& symbol) -> AnfExpr {
                LOG(FATAL) << "outer_func should be a lambda or "
                           << kBuiltinReturn();
                return Atomic<AnfExpr>{""};
              });
        },
        [&](const auto& c) -> AnfExpr {
          LOG(FATAL) << "outer_func should be a lambda or " << kBuiltinReturn();
          return Atomic<AnfExpr>(c);
        });
  }

  Combined<AnfExpr> ConvertComposedCallToCombined(
      const ComposedCall<Atomic<CoreExpr>>& composed_call) {
    const auto& f = ConvertAtomic(composed_call->inner_func);
    std::vector<Atomic<AnfExpr>> args;
    args.reserve(composed_call->args.size());
    for (const auto& arg : composed_call->args) {
      args.push_back(ConvertAtomic(arg));
    }
    return Combined<AnfExpr>{Call<AnfExpr>{f, std::move(args)}};
  }

  AnfExpr ConvertComposedCallToLet(
      const ComposedCall<Atomic<CoreExpr>>& composed_call,
      std::vector<Bind<AnfExpr>>* bindings) {
    const auto& outer_func = composed_call->outer_func;
    return outer_func.Match(
        [&](const Lambda<CoreExpr>& lambda) -> AnfExpr {
          CHECK_EQ(lambda->args.size(), 1U);
          const auto& val = ConvertComposedCallToCombined(composed_call);
          Bind<AnfExpr> binding{lambda->args.at(0), val};
          bindings->emplace_back(std::move(binding));
          const auto& body = lambda->body;
          return body.Match(
              [&](const Atomic<CoreExpr>& atomic_body) -> AnfExpr {
                return Let<AnfExpr>{*bindings, ConvertAtomic(atomic_body)};
              },
              [&](const ComposedCall<Atomic<CoreExpr>>& composed_call_body)
                  -> AnfExpr {
                return ConvertComposedCallToLet(composed_call_body, bindings);
              });
        },
        [&](const Symbol& symbol) -> AnfExpr {
          return symbol.Match(
              [&](const tVar<std::string>& var) -> AnfExpr {
                CHECK_EQ(var.value(), kBuiltinReturn());
                const auto& body = ConvertComposedCallToCombined(composed_call);
                return Let<AnfExpr>{*bindings, body};
              },
              [&](const builtin_symbol::Symbol& symbol) -> AnfExpr {
                LOG(FATAL) << "outer_func should be a lambda or "
                           << kBuiltinReturn();
                return Atomic<AnfExpr>{""};
              });
        },
        [&](const auto& c) -> AnfExpr {
          LOG(FATAL) << "outer_func should be a lambda or " << kBuiltinReturn();
          return Atomic<AnfExpr>(c);
        });
  }
};

}  // namespace detail

inline AnfExpr ConvertCoreExprToAnfExpr(const CoreExpr& core_expr) {
  return detail::CoreExprToAnfExprConverter().ConvertCoreExprToAnfExpr(
      core_expr);
}

namespace detail {

// Convert anf expr to core expr without duplicate var name.
struct AnfExprToCoreExprConverter {
  AnfExprToCoreExprConverter() : core_() {}

  using LazyCoreExpr = std::function<ComposedCallAtomic<CoreExpr>(
      const Atomic<CoreExpr>& continuation)>;

  using MaybeLazyCoreExprBase = std::variant<CoreExpr, LazyCoreExpr>;

  struct MaybeLazyCoreExpr : public MaybeLazyCoreExprBase {
    using MaybeLazyCoreExprBase::MaybeLazyCoreExprBase;

    DEFINE_MATCH_METHOD();

    const MaybeLazyCoreExprBase& variant() const {
      return reinterpret_cast<const MaybeLazyCoreExprBase&>(*this);
    }

    template <typename T>
    bool Has() const {
      return std::holds_alternative<T>(variant());
    }

    template <typename T>
    const T& Get() const {
      return std::get<T>(variant());
    }
  };

  template <typename T>
  MaybeLazyCoreExpr CoreVal(const T& val) {
    return MaybeLazyCoreExpr{CoreExpr{val}};
  }

  MaybeLazyCoreExpr LazyCoreVal(const LazyCoreExpr& lazy) {
    return MaybeLazyCoreExpr{lazy};
  }

  using value_type = MaybeLazyCoreExpr;

  CoreExpr ConvertAnfExprToCoreExpr(const AnfExpr& anf_expr) {
    MaybeLazyCoreExpr ret_val = Convert(anf_expr);
    const auto& lazy_core_expr = TryWrapperToLazyCoreExpr(ret_val);
    CoreExpr ret = lazy_core_expr(CoreExprBuilder().Var(kBuiltinReturn()));
    return ret.Match(
        [&](const Atomic<CoreExpr>&) -> CoreExpr { return ret; },
        [&](const ComposedCallAtomic<CoreExpr>& composed_call) -> CoreExpr {
          Atomic<CoreExpr> return_id{tVar<std::string>{kBuiltinReturn()}};
          Atomic<CoreExpr> identity{Symbol{builtin_symbol::Id{}}};
          if (composed_call->outer_func != return_id) {
            return composed_call;
          }
          if (composed_call->inner_func != identity) {
            return composed_call;
          }
          if (composed_call->args.size() != 1) {
            return composed_call;
          }
          return composed_call->args.at(0);
        });
  }

  value_type Convert(const AnfExpr& anf_expr) {
    return anf_expr.Match(
        [&](const Atomic<AnfExpr>& atomic_expr) {
          return ConvertAtomic(atomic_expr);
        },
        [&](const Combined<AnfExpr>& combined_expr) {
          return ConvertCombined(combined_expr);
        },
        [&](const Let<AnfExpr>& let_expr) { return ConvertLet(let_expr); });
  }

  LazyCoreExpr TryWrapperToLazyCoreExpr(
      const MaybeLazyCoreExpr& maybe_lazy_core_expr) {
    return maybe_lazy_core_expr.Match(
        [&](const LazyCoreExpr& lazy) { return lazy; },
        [&](const CoreExpr& core_expr) {
          PADDLE_ENFORCE_EQ(
              core_expr.Has<Atomic<CoreExpr>>(),
              true,
              common::errors::InvalidArgument(
                  "core_expr should return a Atomic<CoreExpr> instance"));
          const Atomic<CoreExpr> val = core_expr.Get<Atomic<CoreExpr>>();
          return LazyCoreExpr([val](const Atomic<CoreExpr>& continuation) {
            CoreExprBuilder core{};
            return core.ComposedCallAtomic(
                continuation, Symbol{builtin_symbol::Id{}}, {val});
          });
        });
  }

  value_type ConvertAtomic(const Atomic<AnfExpr>& atomic_expr) {
    return atomic_expr.Match(
        [&](const tVar<std::string>& var) { return ConvertVar(var); },
        [&](const adt::Nothing) { return ConvertNothing(); },
        [&](bool c) { return ConvertBool(c); },
        [&](int64_t c) { return ConvertInt64(c); },
        [&](double c) { return ConvertDouble(c); },
        [&](const std::string& c) { return ConvertString(c); },
        [&](const Lambda<AnfExpr>& lambda) { return ConvertLambda(lambda); });
  }

  value_type ConvertCombined(const Combined<AnfExpr>& combined_expr) {
    return combined_expr.Match(
        [&](const Call<AnfExpr>& call_expr) { return ConvertCall(call_expr); },
        [&](const If<AnfExpr>& if_expr) { return ConvertIf(if_expr); });
  }

  value_type ConvertVar(const tVar<std::string>& var) {
    const auto& opt_symbol = builtin_symbol::GetSymbolFromString(var.value());
    return CoreVal(opt_symbol.Match(
        [&](const builtin_symbol::Symbol& symbol) -> Symbol { return symbol; },
        [&](const adt::Nothing&) -> Symbol { return var; }));
  }
  value_type ConvertNothing() { return CoreVal(core_.None()); }
  value_type ConvertBool(const bool c) { return CoreVal(core_.Bool(c)); }
  value_type ConvertInt64(const int64_t c) { return CoreVal(core_.Int64(c)); }
  value_type ConvertDouble(const double c) { return CoreVal(core_.Double(c)); }
  value_type ConvertString(const std::string& c) {
    return CoreVal(core_.String(c));
  }
  value_type ConvertLambda(const Lambda<AnfExpr>& anf_expr) {
    const auto& core_body_val = Convert(anf_expr->body);
    LazyCoreExpr lazy_core_expr = TryWrapperToLazyCoreExpr(core_body_val);
    CoreExpr core_body = lazy_core_expr(core_.Var(kBuiltinReturn()));
    return CoreVal(core_.Lambda(anf_expr->args, core_body));
  }

  value_type ConvertCall(const Call<AnfExpr>& anf_expr) {
    const auto& inner_func = ConvertAtomicToAtomic(anf_expr->func);
    std::vector<Atomic<CoreExpr>> core_args{};
    core_args.reserve(anf_expr->args.size());
    for (const auto& arg : anf_expr->args) {
      core_args.push_back(ConvertAtomicToAtomic(arg));
    }
    return LazyCoreVal(
        [inner_func, core_args](const Atomic<CoreExpr>& continuation) {
          CoreExprBuilder core{};
          return core.ComposedCallAtomic(continuation, inner_func, core_args);
        });
  }
  value_type ConvertIf(const If<AnfExpr>& anf_expr) {
    const Atomic<CoreExpr>& core_cond = ConvertAtomicToAtomic(anf_expr->cond);
    const auto& MakeZeroArgLambda = [](const auto& expr_ptr) {
      return AnfExprBuilder().Lambda({}, expr_ptr);
    };
    const Atomic<CoreExpr>& core_true_expr =
        ConvertAtomicToAtomic(MakeZeroArgLambda(anf_expr->true_expr));
    const Atomic<CoreExpr>& core_false_expr =
        ConvertAtomicToAtomic(MakeZeroArgLambda(anf_expr->false_expr));
    return LazyCoreVal([=](const Atomic<CoreExpr>& continuation) {
      CoreExprBuilder core{};
      return core.ComposedCallAtomic(
          continuation,
          core.Var("if"),
          {core_cond, core_true_expr, core_false_expr});
    });
  }
  value_type ConvertLet(const Let<AnfExpr>& anf_expr) {
    std::vector<std::string> symbol_names;
    std::vector<LazyCoreExpr> lazy_core_exprs;
    lazy_core_exprs.reserve(anf_expr->bindings.size());
    for (const auto& binding : anf_expr->bindings) {
      symbol_names.push_back(binding.var.value());
      lazy_core_exprs.push_back(ConvertCombinedToLazyCoreExpr(binding.val));
    }
    value_type body_val = Convert(anf_expr->body);
    LazyCoreExpr body_lazy_core_expr = TryWrapperToLazyCoreExpr(body_val);
    lazy_core_exprs.push_back(body_lazy_core_expr);
    PADDLE_ENFORCE_EQ(
        lazy_core_exprs.size(),
        symbol_names.size() + 1,
        common::errors::InvalidArgument(
            "lazy_core_exprs.size() should equal to symbol_names.size() + 1"));
    return LazyCoreVal(
        [symbol_names, lazy_core_exprs](Atomic<CoreExpr> continuation) {
          CoreExprBuilder core{};
          LazyCoreExpr first_body_lazy_core_expr = lazy_core_exprs.at(0);
          for (int i = lazy_core_exprs.size() - 1; i > 0; i--) {
            const auto& var = symbol_names.at(i - 1);
            LazyCoreExpr lazy_core_expr = lazy_core_exprs.at(i);
            CoreExpr body = lazy_core_expr(continuation);
            continuation = core.Lambda({tVar<std::string>{var}}, body);
          }
          return first_body_lazy_core_expr(continuation);
        });
  }

 private:
  void CheckIsAtomic(const value_type& maybe_lazy_core_expr) {
    PADDLE_ENFORCE_EQ(maybe_lazy_core_expr.Has<CoreExpr>(),
                      true,
                      common::errors::InvalidArgument(
                          "ConvertAtomic should return a CoreExpr instance"));
    const auto& core_expr = maybe_lazy_core_expr.Get<CoreExpr>();
    PADDLE_ENFORCE_EQ(
        core_expr.Has<Atomic<CoreExpr>>(),
        true,
        common::errors::InvalidArgument(
            "ConvertAtomic should return a Atomic<CoreExpr> instance"));
  }

  Atomic<CoreExpr> GetAtomic(const value_type& val) {
    return val.Get<CoreExpr>().Get<Atomic<CoreExpr>>();
  }

  Atomic<CoreExpr> ConvertAtomicToAtomic(const Atomic<AnfExpr>& atomic_anf) {
    value_type val = ConvertAtomic(atomic_anf);
    CheckIsAtomic(val);
    return GetAtomic(val);
  }

  void CheckIsLazyCoreExpr(const value_type& maybe_lazy_core_expr) {
    PADDLE_ENFORCE_EQ(
        maybe_lazy_core_expr.Has<LazyCoreExpr>(),
        true,
        common::errors::InvalidArgument(
            "ConvertCombined should return a LazyCoreExpr instance"));
  }

  LazyCoreExpr GetLazyCoreExpr(const value_type& val) {
    return val.Get<LazyCoreExpr>();
  }

  LazyCoreExpr ConvertCombinedToLazyCoreExpr(
      const Combined<AnfExpr>& combined_anf) {
    value_type val = ConvertCombined(combined_anf);
    CheckIsLazyCoreExpr(val);
    return GetLazyCoreExpr(val);
  }

  CoreExprBuilder core_;
};

}  // namespace detail

inline CoreExpr ConvertAnfExprToCoreExpr(const AnfExpr& anf_expr) {
  return detail::AnfExprToCoreExprConverter().ConvertAnfExprToCoreExpr(
      anf_expr);
}

namespace detail {

using adt::Result;

using Json = nlohmann::json;

inline adt::errors::Error JsonParseFailed(const Json& j_obj,
                                          const std::string& msg) {
  return adt::errors::TypeError{msg + " json: " + j_obj.dump()};
}

inline adt::errors::Error JsonParseMismatch(const Json& j_obj,
                                            const std::string& msg) {
  return adt::errors::MismatchError{msg};
}

typedef Result<AnfExpr> (*JsonParseFuncType)(const Json& j_obj);

Result<AnfExpr> ConvertJsonToAnfExpr(const Json& j_obj);

struct ParseJsonToAnfExprHelperVar {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_string()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<tVar<std::string>>: json "
                               "objects should be strings");
    }
    std::string str = j_obj.get<std::string>();
    return AnfExpr{AnfExprBuilder().Var(str)};
  }
};

struct ParseJsonToAnfExprHelperNull {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_null()) {
      return JsonParseMismatch(
          j_obj, "ParseJsonToAnfExpr<bool>: json object should be null.");
    }
    return AnfExpr{AnfExprBuilder().None()};
  }
};

struct ParseJsonToAnfExprHelperBool {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_boolean()) {
      return JsonParseMismatch(
          j_obj, "ParseJsonToAnfExpr<bool>: json object should be a boolean.");
    }
    bool c = j_obj.get<bool>();
    return AnfExpr{AnfExprBuilder().Bool(c)};
  }
};

struct ParseJsonToAnfExprHelperInt64 {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_number_integer()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<int64_t>: json object "
                               "should be a intergral number.");
    }
    auto c = j_obj.get<Json::number_integer_t>();
    return AnfExpr{AnfExprBuilder().Int64(c)};
  }
};

struct ParseJsonToAnfExprHelperDouble {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_number_float()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<double>: json object should "
                               "be a floating point number.");
    }
    auto c = j_obj.template get<double>();
    return AnfExpr{AnfExprBuilder().Double(c)};
  }
};

struct ParseJsonToAnfExprHelperString {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_object()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<std::string>: an string "
                               "AnfExpr should be a json object.");
    }
    if (!j_obj.contains(AnfExpr::kString())) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<std::string>: an string "
                               "AnfExpr should contain a string.");
    }
    if (j_obj.size() != 1) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<std::string>: length of json "
                             "object should equal to 1.");
    }
    if (!j_obj[AnfExpr::kString()].is_string()) {
      return JsonParseFailed(
          j_obj,
          "ParseJsonToAnfExpr<std::string>: an string AnfExpr "
          "should contain a string.");
    }
    auto c = j_obj[AnfExpr::kString()].get<std::string>();
    return AnfExpr{AnfExprBuilder().String(c)};
  }
};

struct ParseJsonToAnfExprHelperLambdaAnfExpr {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_array()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<Lambda<AnfExpr>>: json "
                               "objects should be arrays.");
    }
    if (j_obj.size() != 3) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<Lambda<AnfExpr>>: length of "
                               "json array should equal to 3.");
    }
    if (j_obj.at(0) != AnfExpr::kLambda()) {
      return JsonParseMismatch(
          j_obj,
          "ParseJsonToAnfExpr<Lambda<AnfExpr>>: the first "
          "element of json array should equal to 'lambda'.");
    }
    if (!j_obj.at(1).is_array()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Lambda<AnfExpr>>: the second "
                             "element of json array should be a list.");
    }
    std::vector<tVar<std::string>> args;
    for (const auto& arg : j_obj.at(1)) {
      if (!arg.is_string()) {
        return JsonParseFailed(j_obj,
                               "ParseJsonToAnfExpr<Lambda<AnfExpr>>: lambda "
                               "arguments should be var names.");
      }
      args.emplace_back(arg.get<std::string>());
    }
    const auto& body = ConvertJsonToAnfExpr(j_obj.at(2));
    if (!body.HasOkValue()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Lambda<AnfExpr>>: the lambda "
                             "body should be a valid AnfExpr.");
    }
    return AnfExpr{AnfExprBuilder().Lambda(args, body.GetOkValue())};
  }
};

struct ParseJsonToAnfExprHelperCallAnfExpr {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_array()) {
      return JsonParseMismatch(
          j_obj,
          "ParseJsonToAnfExpr<Call<AnfExpr>>: json objects should be arrays.");
    }
    if (j_obj.empty()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Call<AnfExpr>>: json arrays "
                             "should be not empty.");
    }
    const auto& func = ConvertJsonToAnfExpr(j_obj.at(0));
    if (!func.HasOkValue()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Call<AnfExpr>>: the function "
                             "should a valid AnfExpr.");
    }
    if (!func.GetOkValue().Has<Atomic<AnfExpr>>()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<Call<AnfExpr>>: the function "
                             "should a valid atomic AnfExpr.");
    }
    std::vector<Atomic<AnfExpr>> args;
    for (size_t i = 1; i < j_obj.size(); ++i) {
      const auto& arg = j_obj.at(i);
      const auto& arg_expr = ConvertJsonToAnfExpr(arg);
      if (!arg_expr.HasOkValue()) {
        return JsonParseFailed(j_obj,
                               "ParseJsonToAnfExpr<Call<AnfExpr>>: the args "
                               "should be valid AnfExprs.");
      }
      if (!arg_expr.GetOkValue().Has<Atomic<AnfExpr>>()) {
        return JsonParseFailed(j_obj,
                               "ParseJsonToAnfExpr<Call<AnfExpr>>: the args "
                               "should be valid atomic AnfExprs.");
      }
      args.push_back(arg_expr.GetOkValue().Get<Atomic<AnfExpr>>());
    }
    return AnfExpr{
        AnfExprBuilder().Call(func.GetOkValue().Get<Atomic<AnfExpr>>(), args)};
  }
};

struct ParseJsonToAnfExprHelperIfAnfExpr {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_array()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<If<AnfExpr>>: json objects "
                               "should be valid atomic AnfExprs.");
    }
    if (j_obj.size() != 4) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<If<AnfExpr>>: the length of "
                               "json array should equal to 4.");
    }
    if (j_obj.at(0) != AnfExpr::kIf()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<If<AnfExpr>>: the first "
                               "argument of json array should equal to 'if'.");
    }
    const auto& cond = ConvertJsonToAnfExpr(j_obj.at(1));
    if (!cond.HasOkValue()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<If<AnfExpr>>: the second "
                             "argument of json array should a valid AnfExpr.");
    }
    if (!cond.GetOkValue().Has<Atomic<AnfExpr>>()) {
      return JsonParseFailed(
          j_obj,
          "ParseJsonToAnfExpr<If<AnfExpr>>: the second argument of json array "
          "should a valid atomic AnfExpr.");
    }
    const auto& cond_expr = cond.GetOkValue().Get<Atomic<AnfExpr>>();
    const auto& true_expr = ConvertJsonToAnfExpr(j_obj.at(2));
    if (!true_expr.HasOkValue()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<If<AnfExpr>>: the third "
                             "argument of json array should a valid AnfExpr.");
    }
    const auto& false_expr = ConvertJsonToAnfExpr(j_obj.at(3));
    if (!false_expr.HasOkValue()) {
      return JsonParseFailed(j_obj,
                             "ParseJsonToAnfExpr<If<AnfExpr>>: the forth "
                             "argument of json array should a valid AnfExpr.");
    }
    return AnfExpr{AnfExprBuilder().If(
        cond_expr, true_expr.GetOkValue(), false_expr.GetOkValue())};
  }
};

struct ParseJsonToAnfExprHelperLetAnfExpr {
  static Result<AnfExpr> Call(const Json& j_obj) {
    if (!j_obj.is_array()) {
      return JsonParseMismatch(
          j_obj,
          "ParseJsonToAnfExpr<Let<AnfExpr>>: json objects should be arrays.");
    }
    if (j_obj.size() != 3) {
      return JsonParseMismatch(
          j_obj,
          "ParseJsonToAnfExpr<Let<AnfExpr>>: the length of "
          "json array should equal to 3.");
    }
    if (j_obj.at(0) != AnfExpr::kLet()) {
      return JsonParseMismatch(j_obj,
                               "ParseJsonToAnfExpr<Let<AnfExpr>>: the first "
                               "argument of json array should be 'let'.");
    }
    std::vector<Bind<AnfExpr>> bindings;
    const auto& j_bindings = j_obj.at(1);
    for (size_t i = 0; i < j_bindings.size(); ++i) {
      const auto& binding = j_bindings.at(i);
      if (!binding.is_array()) {
        return JsonParseFailed(binding,
                               "ParseJsonToAnfExpr<Let<AnfExpr>>: bindings "
                               "should be json arrays.");
      }
      if (binding.size() != 2) {
        return JsonParseFailed(binding,
                               "ParseJsonToAnfExpr<Let<AnfExpr>>: the size of "
                               "one binding should equal to 2.");
      }
      if (!binding.at(0).is_string()) {
        return JsonParseFailed(binding.at(0),
                               "ParseJsonToAnfExpr<Let<AnfExpr>>: the first "
                               "element of a binding should be var name.");
      }
      std::string var = binding.at(0).get<std::string>();
      const auto& val = ConvertJsonToAnfExpr(binding.at(1));
      if (!val.HasOkValue()) {
        return JsonParseFailed(
            binding.at(1),
            "ParseJsonToAnfExpr<Let<AnfExpr>>: the second "
            "element of a binding should be a valid AnfExpr.");
      }
      if (!val.GetOkValue().Has<Combined<AnfExpr>>()) {
        return JsonParseFailed(
            binding.at(1),
            "ParseJsonToAnfExpr<Let<AnfExpr>>: the second element of a binding "
            "should be a valid combined AnfExpr.");
      }
      bindings.push_back(AnfExprBuilder().Bind(
          var, val.GetOkValue().Get<Combined<AnfExpr>>()));
    }
    const auto& body = ConvertJsonToAnfExpr(j_obj.at(2));
    if (!body.HasOkValue()) {
      return JsonParseFailed(
          j_obj.at(2),
          "ParseJsonToAnfExpr<Let<AnfExpr>>: the body of Let "
          "AnfExpr should be a valid AnfExpr.");
    }
    return AnfExpr{AnfExprBuilder().Let(bindings, body.GetOkValue())};
  }
};

inline const std::vector<JsonParseFuncType>& GetJsonParseFuncs() {
  static const std::vector<JsonParseFuncType> vec{
      &ParseJsonToAnfExprHelperLambdaAnfExpr::Call,
      &ParseJsonToAnfExprHelperIfAnfExpr::Call,
      &ParseJsonToAnfExprHelperLetAnfExpr::Call,
      &ParseJsonToAnfExprHelperCallAnfExpr::Call,
      &ParseJsonToAnfExprHelperVar::Call,
      &ParseJsonToAnfExprHelperNull::Call,
      &ParseJsonToAnfExprHelperBool::Call,
      &ParseJsonToAnfExprHelperInt64::Call,
      &ParseJsonToAnfExprHelperDouble::Call,
      &ParseJsonToAnfExprHelperString::Call,
  };
  return vec;
}

inline Result<AnfExpr> ConvertJsonToAnfExpr(const Json& j_obj) {
  try {
    for (const auto& parse_func : GetJsonParseFuncs()) {
      const auto& ret = parse_func(j_obj);
      if (ret.HasOkValue()) {
        return ret.GetOkValue();
      }
      if (!ret.GetError().Has<adt::errors::MismatchError>()) {
        LOG(ERROR) << "\nTraceback (most recent call last):\n"
                   << ret.GetError().CallStackToString() << "\n"
                   << ret.GetError().class_name()
                   << ": ConvertJsonToAnfExpr: " << ret.GetError().msg();
        return ret.GetError();
      }
    }
  } catch (std::exception& e) {
    return JsonParseFailed(j_obj,
                           "ConvertJsonToAnfExpr: throw error when parsing.");
  }
  return JsonParseFailed(j_obj, "ConvertJsonToAnfExpr: failed to convert.");
}

inline Result<AnfExpr> MakeAnfExprFromJsonString(const std::string& json_str) {
  try {
    return detail::ConvertJsonToAnfExpr(Json::parse(json_str));
  } catch (std::exception& e) {
    return adt::errors::InvalidArgumentError{
        std::string() + "json parse failed. exception::what():" + e.what()};
  }
}

}  // namespace detail

inline adt::Result<AnfExpr> MakeAnfExprFromJsonString(
    const std::string& json_str) {
  return detail::MakeAnfExprFromJsonString(json_str);
}

}  // namespace ap::axpr

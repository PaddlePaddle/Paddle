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

#include <glog/logging.h>
#include <set>
#include <utility>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/bool_helper.h"
#include "paddle/ap/include/axpr/builtin_classes.h"
#include "paddle/ap/include/axpr/builtin_environment.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/builtin_functions.h"
#include "paddle/ap/include/axpr/call_environment.h"
#include "paddle/ap/include/axpr/const_global_environment.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/interpreter_base.h"
#include "paddle/ap/include/axpr/module_mgr_helper.h"
#include "paddle/ap/include/axpr/mutable_global_environment.h"
#include "paddle/ap/include/axpr/to_string.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/axpr/value_method_class.h"

namespace ap::axpr {

class CpsInterpreter : public InterpreterBase<axpr::Value> {
 public:
  using This = CpsInterpreter;
  using Env = Environment<axpr::Value>;
  explicit CpsInterpreter(
      const AttrMap<axpr::Value>& builtin_frame_attr_map,
      const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list)
      : builtin_env_(GetBuiltinEnvironment(builtin_frame_attr_map)),
        circlable_ref_list_(circlable_ref_list) {}
  CpsInterpreter(const CpsInterpreter&) = delete;
  CpsInterpreter(CpsInterpreter&&) = delete;

  using Ok = adt::Result<adt::Ok>;

  const std::shared_ptr<Env>& builtin_env() const { return builtin_env_; }

  Result<axpr::Value> Interpret(const Lambda<CoreExpr>& lambda,
                                const std::vector<axpr::Value>& args) {
    Function<SerializableValue> function{lambda, std::nullopt};
    return Interpret(function, args);
  }

  Result<axpr::Value> Interpret(const axpr::Value& function,
                                const std::vector<axpr::Value>& args) {
    return InterpretCall(function, args);
  }

  Result<axpr::Value> InterpretCall(
      const axpr::Value& func, const std::vector<axpr::Value>& args) override {
    ComposedCallImpl<axpr::Value> composed_call{&BuiltinHalt, func, args};
    ADT_RETURN_IF_ERR(InterpretComposedCallUntilHalt(&composed_call));
    ADT_CHECK(IsHalt(composed_call.inner_func))
        << RuntimeError{"CpsInterpreter does not halt."};
    ADT_CHECK(composed_call.args.size() == 1) << RuntimeError{
        std::string() + "halt function takes 1 argument. but " +
        std::to_string(composed_call.args.size()) + " were given."};
    return composed_call.args.at(0);
  }

  Result<axpr::Value> InterpretModule(
      const Frame<SerializableValue>& const_global_frame,
      const Lambda<CoreExpr>& lambda) override {
    std::optional<std::shared_ptr<Environment<axpr::Value>>> env;
    {
      ADT_LET_CONST_REF(ref_lst, adt::WeakPtrLock(circlable_ref_list_));
      auto tmp_frame_object = std::make_shared<AttrMapImpl<axpr::Value>>();
      auto tmp_frame = Frame<axpr::Value>::Make(ref_lst, tmp_frame_object);
      const auto& mut_global_env = MakeMutableGlobalEnvironment(
          builtin_env(), const_global_frame, tmp_frame);
      env = mut_global_env;
    }
    ADT_CHECK(lambda->args.empty());
    ADT_RETURN_IF_ERR(env.value()->Set(kBuiltinReturn(), &BuiltinHalt));
    Continuation<axpr::Value> continuation{lambda, env.value()};
    const auto& ret = InterpretCall(continuation, {});
    return ret;
  }

 protected:
  Ok InterpretComposedCallUntilHalt(
      ComposedCallImpl<axpr::Value>* composed_call) {
    while (!IsHalt(composed_call->inner_func)) {
      ADT_RETURN_IF_ERR(InterpretComposedCall(composed_call));
    }
    return adt::Ok{};
  }

  Ok InterpretComposedCall(ComposedCallImpl<axpr::Value>* composed_call) {
    using TypeT = typename TypeTrait<axpr::Value>::TypeT;
    return composed_call->inner_func.Match(
        [&](const TypeT& type) -> Ok {
          return InterpretConstruct(type, composed_call);
        },
        [&](const BuiltinFuncType<axpr::Value>& func) -> Ok {
          return InterpretBuiltinFuncCall(func, composed_call);
        },
        [&](const BuiltinHighOrderFuncType<axpr::Value>& func) -> Ok {
          return InterpretBuiltinHighOrderFuncCall(func, composed_call);
        },
        [&](const Method<axpr::Value>& method) -> Ok {
          return method->func.Match(
              [&](const BuiltinFuncType<axpr::Value>& func) {
                return InterpretBuiltinMethodCall(
                    func, method->obj, composed_call);
              },
              [&](const BuiltinHighOrderFuncType<axpr::Value>& func) {
                return InterpretBuiltinHighOrderMethodCall(
                    func, method->obj, composed_call);
              },
              [&](const auto&) {
                return InterpretMethodCall(method, composed_call);
              });
        },
        [&](const Closure<axpr::Value>& closure) -> Ok {
          return InterpretClosureCall(composed_call->outer_func,
                                      closure,
                                      composed_call->args,
                                      composed_call);
        },
        [&](const Continuation<axpr::Value>& continuation) -> Ok {
          return InterpretContinuation(
              &BuiltinHalt, continuation, composed_call);
        },
        [&](const Function<SerializableValue>& function) -> Ok {
          ADT_LET_CONST_REF(closure, ConvertFunctionToClosure(function));
          return InterpretClosureCall(composed_call->outer_func,
                                      closure,
                                      composed_call->args,
                                      composed_call);
        },
        [&](const builtin_symbol::Symbol& symbol) -> Ok {
          return InterpretBuiltinSymbolCall(symbol, composed_call);
        },
        [&](const auto&) -> Ok {
          const auto& call_func =
              MethodClass<axpr::Value>::template GetBuiltinUnaryFunc<
                  builtin_symbol::Call>(composed_call->inner_func);
          ADT_RETURN_IF_ERR(call_func.Match(
              [&](const adt::Nothing&) -> Ok {
                return adt::errors::TypeError{
                    std::string("'") +
                    axpr::GetTypeName(composed_call->inner_func) +
                    "' object is not callable"};
              },
              [&](adt::Result<axpr::Value> (*unary_func)(
                  const axpr::Value&)) -> Ok {
                ADT_LET_CONST_REF(func, unary_func(composed_call->inner_func));
                composed_call->inner_func = func;
                return adt::Ok{};
              },
              [&](adt::Result<axpr::Value> (*unary_func)(
                  InterpreterBase<axpr::Value>*, const axpr::Value&)) -> Ok {
                ADT_LET_CONST_REF(func,
                                  unary_func(this, composed_call->inner_func));
                composed_call->inner_func = func;
                return adt::Ok{};
              }));
          return adt::Ok{};
        });
  }

  bool IsHalt(const axpr::Value& func) {
    return func.Match(
        [&](BuiltinFuncType<axpr::Value> f) { return f == &BuiltinHalt; },
        [&](const auto&) { return false; });
  }

  Result<axpr::Value> InterpretAtomic(const std::shared_ptr<Env>& env,
                                      const Atomic<CoreExpr>& atomic) {
    return atomic.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Result<axpr::Value> {
          if (const auto& const_global_frame = env->GetConstGlobalFrame()) {
            return Function<SerializableValue>{lambda,
                                               const_global_frame.value()};
          } else {
            return Closure<axpr::Value>{lambda, env};
          }
        },
        [&](const Symbol& symbol) -> Result<axpr::Value> {
          return symbol.Match(
              [&](const tVar<std::string>& var) -> Result<axpr::Value> {
                ADT_LET_CONST_REF(val, env->Get(var.value()))
                    << adt::errors::NameError{std::string("var '") +
                                              var.value() +
                                              "' is not defined."};
                return val;
              },
              [&](const builtin_symbol::Symbol& symbol) -> Result<axpr::Value> {
                return symbol;
              });
        },
        [&](adt::Nothing) -> Result<axpr::Value> { return adt::Nothing{}; },
        [&](bool c) -> Result<axpr::Value> { return c; },
        [&](int64_t c) -> Result<axpr::Value> { return c; },
        [&](double c) -> Result<axpr::Value> { return c; },
        [&](const std::string& val) -> Result<axpr::Value> { return val; });
  }

  Result<axpr::Value> InterpretAtomicAsContinuation(
      const std::shared_ptr<Env>& env, const Atomic<CoreExpr>& atomic) {
    return atomic.Match(
        [&](const Lambda<CoreExpr>& lambda) -> Result<axpr::Value> {
          return Continuation<axpr::Value>{lambda, env};
        },
        [&](const Symbol& symbol) -> Result<axpr::Value> {
          return symbol.Match(
              [&](const tVar<std::string>& var) -> Result<axpr::Value> {
                ADT_CHECK(var.value() == kBuiltinReturn());
                ADT_LET_CONST_REF(val, env->Get(var.value()))
                    << adt::errors::NotImplementedError{
                           "no return continuation found."};
                return val;
              },
              [&](const auto&) -> Result<axpr::Value> {
                return adt::errors::NotImplementedError{
                    "Invalid continuation."};
              });
        },
        [&](const auto&) -> Result<axpr::Value> {
          return adt::errors::NotImplementedError{"Invalid continuation."};
        });
  }

  Ok InterpretBuiltinSymbolCall(
      const builtin_symbol::Symbol& symbol,
      ComposedCallImpl<axpr::Value>* ret_composed_call) {
    return symbol.Match(
        [&](const builtin_symbol::If&) -> Ok {
          ADT_RETURN_IF_ERR(InterpretIf(ret_composed_call));
          return adt::Ok{};
        },
        [&](const builtin_symbol::Id&) -> Ok {
          ret_composed_call->inner_func = &BuiltinIdentity;
          return adt::Ok{};
        },
        [&](const builtin_symbol::List&) -> Ok {
          ret_composed_call->inner_func = &BuiltinList;
          return adt::Ok{};
        },
        [&](const builtin_symbol::Op& op) -> Ok {
          return op.Match([&](auto impl) -> Ok {
            using BuiltinSymbol = decltype(impl);
            if constexpr (BuiltinSymbol::num_operands == 1) {
              return this
                  ->template InterpretBuiltinUnarySymbolCall<BuiltinSymbol>(
                      ret_composed_call);
            } else if constexpr (BuiltinSymbol::num_operands == 2) {
              return this
                  ->template InterpretBuiltinBinarySymbolCall<BuiltinSymbol>(
                      ret_composed_call);
            } else {
              static_assert(true, "NotImplemented");
              return NotImplementedError{"NotImplemented."};
            }
          });
        });
  }

  Ok InterpretIf(ComposedCallImpl<axpr::Value>* composed_call) {
    const auto args = composed_call->args;
    ADT_CHECK(args.size() == 3)
        << TypeError{std::string("`if` takes 3 arguments, but ") +
                     std::to_string(args.size()) + "were given."};
    const auto& cond = args.at(0);
    ADT_LET_CONST_REF(select_true_branch, BoolHelper{}.ConvertToBool(cond));
    ADT_LET_CONST_REF(true_closure,
                      args.at(1).template TryGet<Closure<axpr::Value>>());
    ADT_LET_CONST_REF(false_closure,
                      args.at(2).template TryGet<Closure<axpr::Value>>());
    Closure<axpr::Value> closure{select_true_branch ? true_closure
                                                    : false_closure};
    composed_call->inner_func = closure;
    composed_call->args = std::vector<axpr::Value>{};
    return adt::Ok{};
  }

  template <typename BuiltinSymbol>
  Ok InterpretBuiltinUnarySymbolCall(
      ComposedCallImpl<axpr::Value>* ret_composed_call) {
    ADT_CHECK(ret_composed_call->args.size() == 1) << TypeError{
        std::string() + "'" + BuiltinSymbol::Name() +
        "' takes 1 argument. but " +
        std::to_string(ret_composed_call->args.size()) + " were given."};
    const auto& operand = ret_composed_call->args.at(0);
    std::optional<axpr::Value> opt_ret;
    const auto& func =
        MethodClass<axpr::Value>::template GetBuiltinUnaryFunc<BuiltinSymbol>(
            operand);
    ADT_RETURN_IF_ERR(func.Match(
        [&](const adt::Nothing&) -> Ok {
          return TypeError{std::string() + "unsupported operand type for " +
                           GetBuiltinSymbolDebugString<BuiltinSymbol>() +
                           ": '" + axpr::GetTypeName(operand) + "'"};
        },
        [&](adt::Result<axpr::Value> (*unary_func)(const axpr::Value&)) -> Ok {
          ADT_LET_CONST_REF(ret, unary_func(operand));
          opt_ret = ret;
          return adt::Ok{};
        },
        [&](adt::Result<axpr::Value> (*unary_func)(
            InterpreterBase<axpr::Value>*, const axpr::Value&)) -> Ok {
          ADT_LET_CONST_REF(ret, unary_func(this, operand));
          opt_ret = ret;
          return adt::Ok{};
        }));
    ADT_CHECK(opt_ret.has_value());
    ret_composed_call->args = {opt_ret.value()};
    ret_composed_call->inner_func = ret_composed_call->outer_func;
    ret_composed_call->outer_func = &BuiltinHalt;
    return adt::Ok{};
  }

  template <typename TypeT>
  Ok InterpretConstruct(const TypeT& type,
                        ComposedCallImpl<axpr::Value>* ret_composed_call) {
    const auto& func = MethodClass<axpr::Value>::template GetBuiltinUnaryFunc<
        builtin_symbol::Call>(axpr::Value{type});
    ADT_RETURN_IF_ERR(func.Match(
        [&](const adt::Nothing&) -> Ok {
          return adt::errors::TypeError{
              std::string() + "no constructor for type '" + type.Name() + "'"};
        },
        [&](adt::Result<axpr::Value> (*unary_func)(const axpr::Value&)) -> Ok {
          ADT_LET_CONST_REF(constructor, unary_func(axpr::Value{type}));
          ret_composed_call->inner_func = constructor;
          return adt::Ok{};
        },
        [&](adt::Result<axpr::Value> (*unary_func)(
            InterpreterBase<axpr::Value>*, const axpr::Value&)) -> Ok {
          ADT_LET_CONST_REF(constructor, unary_func(this, axpr::Value{type}));
          ret_composed_call->inner_func = constructor;
          return adt::Ok{};
        }));
    return adt::Ok{};
  }

  template <typename BuiltinSymbol>
  Ok InterpretBuiltinBinarySymbolCall(
      ComposedCallImpl<axpr::Value>* ret_composed_call) {
    ADT_CHECK(ret_composed_call->args.size() == 2) << TypeError{
        std::string() + "'" + BuiltinSymbol::Name() +
        "' takes 2 argument. but " +
        std::to_string(ret_composed_call->args.size()) + " were given."};
    const auto& lhs = ret_composed_call->args.at(0);
    const auto& func =
        MethodClass<axpr::Value>::template GetBuiltinBinaryFunc<BuiltinSymbol>(
            lhs);
    std::optional<axpr::Value> opt_ret;
    ADT_RETURN_IF_ERR(func.Match(
        [&](const adt::Nothing&) -> Ok {
          return TypeError{std::string() + "unsupported operand type for " +
                           GetBuiltinSymbolDebugString<BuiltinSymbol>() +
                           ": '" + axpr::GetTypeName(lhs) + "'"};
        },
        [&](adt::Result<axpr::Value> (*binary_func)(const axpr::Value&,
                                                    const axpr::Value&)) -> Ok {
          const auto& rhs = ret_composed_call->args.at(1);
          ADT_LET_CONST_REF(ret, binary_func(lhs, rhs));
          opt_ret = ret;
          return adt::Ok{};
        },
        [&](adt::Result<axpr::Value> (*binary_func)(
            InterpreterBase<axpr::Value>*,
            const axpr::Value&,
            const axpr::Value&)) -> Ok {
          const auto& rhs = ret_composed_call->args.at(1);
          ADT_LET_CONST_REF(ret, binary_func(this, lhs, rhs));
          opt_ret = ret;
          return adt::Ok{};
        }));
    ADT_CHECK(opt_ret.has_value());
    ret_composed_call->args = {opt_ret.value()};
    ret_composed_call->inner_func = ret_composed_call->outer_func;
    ret_composed_call->outer_func = &BuiltinHalt;
    return adt::Ok{};
  }

  Ok InterpretClosureCall(const axpr::Value& continuation,
                          const Closure<axpr::Value>& closure,
                          const std::vector<axpr::Value>& args,
                          ComposedCallImpl<axpr::Value>* ret_composed_call) {
    ADT_LET_CONST_REF(new_env, MakeCallEnvironment(closure->environment));
    ADT_RETURN_IF_ERR(new_env->Set(kBuiltinReturn(), continuation));
    return InterpretLambdaCall(
        new_env, continuation, closure->lambda, args, ret_composed_call);
  }

  Ok InterpretLambdaCall(
      const std::shared_ptr<Env>& env,
      const axpr::Value& outer_func,
      const Lambda<CoreExpr>& lambda,
      const std::vector<axpr::Value>& args,
      ComposedCallImpl<axpr::Value>* ret_composed_call) override {
    auto PassPackedArgs = [&](const std::optional<axpr::Value>& self,
                              const axpr::Value& packed) -> Ok {
      ADT_LET_CONST_REF(packed_args,
                        packed.template TryGet<PackedArgs<axpr::Value>>());
      const auto& [pos_args, kwargs] = *packed_args;
      size_t lambda_arg_idx = (self.has_value() ? 1 : 0);
      ADT_CHECK(lambda_arg_idx + pos_args->size() <= lambda->args.size())
          << TypeError{std::string("<lambda>() takes ") +
                       std::to_string(lambda->args.size()) +
                       "at most positional arguments but " +
                       std::to_string(pos_args->size()) + " was given"};
      std::set<std::string> passed_args;
      if (self.has_value()) {
        const auto& self_name = lambda->args.at(0).value();
        passed_args.insert(self_name);
        ADT_RETURN_IF_ERR(env->Set(self_name, self.value()));
      }
      for (size_t pos_arg_idx = 0; pos_arg_idx < pos_args->size();
           ++pos_arg_idx, ++lambda_arg_idx) {
        const auto& arg_name = lambda->args.at(lambda_arg_idx).value();
        passed_args.insert(arg_name);
        ADT_RETURN_IF_ERR(env->Set(arg_name, pos_args->at(pos_arg_idx)));
      }
      for (; lambda_arg_idx < lambda->args.size(); ++lambda_arg_idx) {
        const auto& arg_name = lambda->args.at(lambda_arg_idx).value();
        if (passed_args.count(arg_name) > 0) {
          return adt::errors::TypeError{
              std::string() + "<lambda>() got multiple values for argument '" +
              arg_name + "'"};
        }
        passed_args.insert(arg_name);
        ADT_LET_CONST_REF(kwarg, kwargs->Get(arg_name))
            << adt::errors::TypeError{
                   std::string() +
                   "<lambda>() missing 1 required positional argument: '" +
                   arg_name + "'"};
        ADT_RETURN_IF_ERR(env->Set(arg_name, kwarg));
      }
      for (const auto& [key, _] : kwargs->storage) {
        ADT_CHECK(passed_args.count(key) > 0) << adt::errors::TypeError{
            std::string() + "<lambda>() got an unexpected keyword argument '" +
            key + "'"};
      }
      return adt::Ok{};
    };
    if (args.size() == 1 &&
        args.at(0).template Has<PackedArgs<axpr::Value>>()) {
      ADT_RETURN_IF_ERR(
          PassPackedArgs(/*self=*/std::nullopt, /*packed=*/args.at(0)));
    } else if (args.size() == 2 &&
               args.at(1).template Has<PackedArgs<axpr::Value>>()) {
      ADT_RETURN_IF_ERR(
          PassPackedArgs(/*self=*/args.at(0), /*packed=*/args.at(1)));
    } else {
      if (args.size() > lambda->args.size()) {
        return adt::errors::TypeError{
            std::string("<lambda>() takes ") +
            std::to_string(lambda->args.size()) + " positional arguments but " +
            std::to_string(args.size()) + " was given"};
      }
      if (args.size() < lambda->args.size()) {
        if (args.size() + 1 == lambda->args.size()) {
          return adt::errors::TypeError{
              "<lambda>() missing 1 required positional argument: '" +
              lambda->args.at(args.size()).value() + "'"};
        } else {
          std::ostringstream ss;
          ss << "<lambda>() missing " << (lambda->args.size() - args.size())
             << " required positional arguments: ";
          ss << "'" << lambda->args.at(args.size()).value() << "'";
          for (size_t i = args.size() + 1; i < lambda->args.size(); ++i) {
            ss << "and '" << lambda->args.at(i).value() << "'";
          }
          return adt::errors::TypeError{ss.str()};
        }
      }
      for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg_name = lambda->args.at(i).value();
        ADT_RETURN_IF_ERR(env->Set(arg_name, args.at(i)));
      }
    }
    return InterpretLambdaBody(
        env, outer_func, lambda->body, ret_composed_call);
  }

  Ok InterpretContinuation(const axpr::Value& outer_func,
                           const Continuation<axpr::Value>& continuation,
                           ComposedCallImpl<axpr::Value>* composed_call) {
    const auto& env = continuation->environment;
    const auto& lambda = continuation->lambda;
    if (lambda->args.size() > 0) {
      ADT_CHECK(lambda->args.size() == 1);
      ADT_CHECK(composed_call->args.size() == 1);
      ADT_RETURN_IF_ERR(
          env->Set(lambda->args.at(0).value(), composed_call->args.at(0)));
    } else {
      // Do nothing.
    }
    return InterpretLambdaBody(env, outer_func, lambda->body, composed_call);
  }

  Ok InterpretLambdaBody(const std::shared_ptr<Env>& env,
                         const axpr::Value& outer_func,
                         const CoreExpr& lambda_body,
                         ComposedCallImpl<axpr::Value>* ret_composed_call) {
    return lambda_body.Match(
        [&](const Atomic<CoreExpr>& atomic) -> Ok {
          ADT_LET_CONST_REF(val, InterpretAtomic(env, atomic));
          ret_composed_call->inner_func = outer_func;
          ret_composed_call->outer_func = &BuiltinHalt;
          ret_composed_call->args = {val};
          return adt::Ok{};
        },
        [&](const ComposedCallAtomic<CoreExpr>& core_expr) -> Ok {
          return InterpretLambdaBodyComposedCallAtomic(
              env, core_expr, ret_composed_call);
        });
  }

  Ok InterpretLambdaBodyComposedCallAtomic(
      const std::shared_ptr<Env>& env,
      const ComposedCallAtomic<CoreExpr>& core_expr,
      ComposedCallImpl<axpr::Value>* ret_composed_call) {
    ADT_LET_CONST_REF(
        continuation,
        InterpretAtomicAsContinuation(env, core_expr->outer_func));
    ADT_LET_CONST_REF(new_inner_func,
                      InterpretAtomic(env, core_expr->inner_func));
    std::vector<axpr::Value> args;
    args.reserve(core_expr->args.size());
    for (const auto& arg_expr : core_expr->args) {
      ADT_LET_CONST_REF(arg, InterpretAtomic(env, arg_expr));
      args.emplace_back(arg);
    }
    ret_composed_call->outer_func = continuation;
    ret_composed_call->inner_func = new_inner_func;
    ret_composed_call->args = std::move(args);
    return adt::Ok{};
  }

  Ok InterpretBuiltinFuncCall(const BuiltinFuncType<axpr::Value>& func,
                              ComposedCallImpl<axpr::Value>* composed_call) {
    return InterpretBuiltinMethodCall(
        func, axpr::Value{adt::Nothing{}}, composed_call);
  }

  Ok InterpretBuiltinHighOrderFuncCall(
      const BuiltinHighOrderFuncType<axpr::Value>& func,
      ComposedCallImpl<axpr::Value>* composed_call) {
    return InterpretBuiltinHighOrderMethodCall(
        func, axpr::Value{adt::Nothing{}}, composed_call);
  }

  Ok InterpretBuiltinMethodCall(const BuiltinFuncType<axpr::Value>& func,
                                const axpr::Value& obj,
                                ComposedCallImpl<axpr::Value>* composed_call) {
    ADT_LET_CONST_REF(inner_ret, func(obj, composed_call->args));
    composed_call->inner_func = composed_call->outer_func;
    composed_call->outer_func = &BuiltinHalt;
    composed_call->args = {inner_ret};
    return adt::Ok{};
  }

  Ok InterpretBuiltinHighOrderMethodCall(
      const BuiltinHighOrderFuncType<axpr::Value>& func,
      const axpr::Value& obj,
      ComposedCallImpl<axpr::Value>* composed_call) {
    ADT_LET_CONST_REF(inner_ret, func(this, obj, composed_call->args));
    composed_call->inner_func = composed_call->outer_func;
    composed_call->outer_func = &BuiltinHalt;
    composed_call->args = {inner_ret};
    return adt::Ok{};
  }

  Ok InterpretMethodCall(const Method<axpr::Value>& method,
                         ComposedCallImpl<axpr::Value>* composed_call) {
    std::vector<axpr::Value> new_args;
    new_args.reserve(composed_call->args.size() + 1);
    new_args.emplace_back(method->obj);
    for (const auto& arg : composed_call->args) {
      new_args.emplace_back(arg);
    }
    composed_call->inner_func = method->func;
    composed_call->args = std::move(new_args);
    return adt::Ok{};
  }

  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list()
      const override {
    return circlable_ref_list_;
  }

  std::shared_ptr<Env> builtin_env_;
  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list_;

 private:
  Result<Closure<axpr::Value>> ConvertFunctionToClosure(
      const Function<SerializableValue>& function) {
    const auto& global_frame = function->global_frame;
    if (global_frame.has_value()) {
      const auto& const_env =
          MakeConstGlobalEnvironment(builtin_env(), global_frame.value());
      return Closure<axpr::Value>{function->lambda, const_env};
    } else {
      return Closure<axpr::Value>{function->lambda, builtin_env()};
    }
  }

  static std::shared_ptr<Environment<axpr::Value>> GetBuiltinEnvironment(
      const AttrMap<axpr::Value>& builtin_frame_attr_map) {
    return std::make_shared<BuiltinEnvironment<axpr::Value>>(
        builtin_frame_attr_map);
  }

  static std::shared_ptr<Environment<axpr::Value>> MakeConstGlobalEnvironment(
      const std::shared_ptr<Environment<axpr::Value>>& parent,
      const Frame<SerializableValue>& frame) {
    return std::make_shared<ConstGlobalEnvironment<axpr::Value>>(parent, frame);
  }

  static std::shared_ptr<Environment<axpr::Value>> MakeMutableGlobalEnvironment(
      const std::shared_ptr<Environment<axpr::Value>>& parent,
      const Frame<SerializableValue>& const_frame,
      const Frame<axpr::Value>& temp_frame) {
    return std::make_shared<MutableGlobalEnvironment<axpr::Value>>(
        parent, const_frame, temp_frame);
  }

  adt::Result<std::shared_ptr<Environment<axpr::Value>>> MakeCallEnvironment(
      const std::shared_ptr<Environment<axpr::Value>>& parent) {
    auto builtin_obj = std::make_shared<AttrMapImpl<axpr::Value>>();
    ADT_LET_CONST_REF(ref_lst, adt::WeakPtrLock(circlable_ref_list()));
    const auto& frame = Frame<axpr::Value>::Make(ref_lst, builtin_obj);
    return std::make_shared<CallEnvironment<axpr::Value>>(parent, frame);
  }
};

}  // namespace ap::axpr

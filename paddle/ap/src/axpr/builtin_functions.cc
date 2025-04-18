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

#include "paddle/ap/include/axpr/builtin_functions.h"
#include <functional>
#include <sstream>
#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/bool_helper.h"
#include "paddle/ap/include/axpr/bool_int_double_helper.h"
#include "paddle/ap/include/axpr/builtin_high_order_func_type.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/data_value_util.h"
#include "paddle/ap/include/axpr/exception_method_class.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/string_util.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/axpr/value_method_class.h"

namespace ap::axpr {

Result<axpr::Value> BuiltinIdentity(const axpr::Value&,
                                    const std::vector<axpr::Value>& args) {
  if (args.size() != 1) {
    return TypeError{std::string(kBuiltinIdentity()) +
                     "takes 1 argument, but " + std::to_string(args.size()) +
                     "were given."};
  }
  return args.at(0);
}

Result<axpr::Value> BuiltinNot(const axpr::Value&,
                               const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(bool_val, BoolHelper{}.ConvertToBool(args.at(0)));
  return !bool_val;
}

Result<axpr::Value> Raise(const axpr::Value&,
                          const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(exception, args.at(0).template CastTo<Exception>());
  return exception.value();
}

Result<axpr::Value> BuiltinList(const axpr::Value&,
                                const std::vector<axpr::Value>& args) {
  adt::List<axpr::Value> l;
  for (const auto& arg : args) {
    const auto& arg_ret = arg.Match(
        [&](const Starred<axpr::Value>& starred) -> Result<adt::Ok> {
          ADT_LET_CONST_REF(
              sublist, starred->obj.template TryGet<adt::List<axpr::Value>>());
          for (const auto& elt : *sublist) {
            l->emplace_back(elt);
          }
          return adt::Ok{};
        },
        [&](const auto&) -> Result<adt::Ok> {
          l->emplace_back(arg);
          return adt::Ok{};
        });
    ADT_RETURN_IF_ERR(arg_ret);
  }
  return axpr::Value{l};
}

Result<axpr::Value> BuiltinHalt(const axpr::Value&,
                                const std::vector<axpr::Value>& args) {
  return RuntimeError{"Dead code. Halt function should never be touched."};
}

adt::Result<axpr::Value> Print(InterpreterBase<axpr::Value>* interpreter,
                               const axpr::Value&,
                               const std::vector<axpr::Value>& args) {
  std::ostringstream ss;
  int i = 0;
  for (const auto& obj : args) {
    if (i++ > 0) {
      ss << " ";
    }
    const auto& func = MethodClass<axpr::Value>::ToString(obj);
    using Ok = adt::Result<adt::Ok>;
    ADT_RETURN_IF_ERR(func.Match(
        [&](const adt::Nothing&) -> Ok {
          return adt::errors::TypeError{std::string() + GetTypeName(obj) +
                                        " class has no ToString method"};
        },
        [&](adt::Result<axpr::Value> (*unary_func)(const axpr::Value&)) -> Ok {
          ADT_LET_CONST_REF(str_val, unary_func(obj));
          ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>())
              << adt::errors::TypeError{
                     std::string() + "'" + axpr::GetTypeName(obj) +
                     ".__builtin_ToString__ should return a 'str' but '" +
                     axpr::GetTypeName(str_val) + "' were returned."};
          ss << str;
          return adt::Ok{};
        },
        [&](adt::Result<axpr::Value> (*unary_func)(
            InterpreterBase<axpr::Value>*, const axpr::Value&)) -> Ok {
          ADT_LET_CONST_REF(str_val, unary_func(interpreter, obj));
          ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>())
              << adt::errors::TypeError{
                     std::string() + "'" + axpr::GetTypeName(obj) +
                     ".__builtin_ToString__ should return a 'str' but '" +
                     axpr::GetTypeName(str_val) + "' were returned."};
          ss << str;
          return adt::Ok{};
        }));
  }
  LOG(ERROR) << "Print\n" << ss.str();
  return adt::Nothing{};
}

adt::Result<axpr::Value> ReplaceOrTrimLeftComma(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
      std::string() + "'replace_or_trim_left_comma' takes 3 arguments but " +
      std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(self, args.at(0).template TryGet<std::string>())
      << adt::errors::TypeError{
             std::string() +
             "the argument 1 of 'replace_or_trim_left_comma' should be a str "
             "(not '" +
             axpr::GetTypeName(args.at(0)) + "')."};
  ADT_LET_CONST_REF(pattern, args.at(1).template TryGet<std::string>())
      << adt::errors::TypeError{
             std::string() +
             "the argument 2 of 'replace_or_trim_left_comma' should be a str "
             "(not '" +
             axpr::GetTypeName(args.at(1)) + "')."};
  ADT_LET_CONST_REF(replacement, args.at(2).template TryGet<std::string>())
      << adt::errors::TypeError{
             std::string() +
             "the argument 3 of 'replace_or_trim_left_comma' should be a str "
             "(not '" +
             axpr::GetTypeName(args.at(2)) + "')."};
  std::size_t pattern_pos = self.find(pattern);
  if (pattern_pos == std::string::npos) {
    return self;
  }
  auto EquivalentComma =
      [](const std::string& self, std::size_t start, std::size_t end) {
        if (start == std::string::npos) {
          return false;
        }
        if (start >= self.size()) {
          return false;
        }
        if (end == std::string::npos) {
          return false;
        }
        if (end >= self.size()) {
          return false;
        }
        if (start >= end) {
          return false;
        }
        if (self[start] != ',') {
          return false;
        }
        for (size_t i = start + 1; i < end; ++i) {
          char ch = self[i];
          if (ch == ' ') {
            continue;
          }
          if (ch == '\r') {
            continue;
          }
          if (ch == '\n') {
            continue;
          }
          if (ch == '\t') {
            continue;
          }
          return false;
        }
        return true;
      };
  if (replacement.empty()) {
    std::string str = self;
    while (true) {
      std::size_t pattern_pos = self.find(pattern);
      if (pattern_pos == std::string::npos) {
        break;
      }
      std::size_t comma_pos = str.rfind(',', pattern_pos);
      if (!EquivalentComma(str, comma_pos, pattern_pos)) {
        break;
      }
      str = str.replace(comma_pos, pattern_pos + pattern.size(), "");
    }
    return str;
  } else {
    std::string str = self;
    while (true) {
      std::size_t pos = str.find(pattern);
      if (pos == std::string::npos) {
        break;
      }
      str = str.replace(pos, pattern.size(), replacement);
    }
    return str;
  }
}

adt::Result<axpr::Value> MakeRange(const axpr::Value&,
                                   const std::vector<axpr::Value>& args) {
  std::optional<int64_t> start;
  std::optional<int64_t> end;
  if (args.size() == 1) {
    start = 0;
    ADT_LET_CONST_REF(arg0, args.at(0).template TryGet<int64_t>())
        << adt::errors::TypeError{
               std::string() + "'range' takes int argument but " +
               axpr::GetTypeName(args.at(0)) + " were given."};
    end = arg0;
  } else if (args.size() == 2) {
    ADT_LET_CONST_REF(arg0, args.at(0).template TryGet<int64_t>())
        << adt::errors::TypeError{
               std::string() + "'range' takes int argument but " +
               axpr::GetTypeName(args.at(0)) + " were given."};
    ADT_LET_CONST_REF(arg1, args.at(1).template TryGet<int64_t>())
        << adt::errors::TypeError{
               std::string() + "'range' takes int argument but " +
               axpr::GetTypeName(args.at(1)) + " were given."};
    start = arg0;
    end = arg1;
  } else {
    ADT_CHECK(false) << adt::errors::TypeError{
        std::string() + "'range' takes 1 or 2 arguments but " +
        std::to_string(args.size()) + " were given."};
  }
  ADT_CHECK(start.has_value());
  ADT_CHECK(end.has_value());
  adt::List<axpr::Value> ret;
  ret->reserve((start.value() > end.value() ? 0 : end.value() - start.value()));
  for (int64_t i = start.value(); i < end.value(); ++i) {
    ret->emplace_back(i);
  }
  return ret;
}

Result<axpr::Value> Map(axpr::InterpreterBase<axpr::Value>* interpreter,
                        const axpr::Value&,
                        const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2)
      << adt::errors::TypeError{std::string() + "map() takes 2 arguments but " +
                                std::to_string(args.size()) + " were given."};

  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(1)));
  ADT_LET_CONST_REF(lst_size, lst.size());
  adt::List<axpr::Value> ret;
  ret->reserve(lst_size);
  const auto& f = args.at(0);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& elt) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(
            converted_elt,
            interpreter->InterpretCall(f, std::vector<axpr::Value>{elt}));
        ret->emplace_back(converted_elt);
        return adt::Continue{};
      }));
  return ret;
}

Result<axpr::Value> ForEach(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
      std::string() + "foreach() takes 2 arguments but " +
      std::to_string(args.size()) + " were given."};

  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(1)));
  ADT_LET_CONST_REF(lst_size, lst.size());
  const auto& f = args.at(0);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& elt) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(
            converted_elt,
            interpreter->InterpretCall(f, std::vector<axpr::Value>{elt}));
        return adt::Continue{};
      }));
  return adt::Nothing{};
}

Result<axpr::Value> Apply(axpr::InterpreterBase<axpr::Value>* interpreter,
                          const axpr::Value&,
                          const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
      std::string() + "apply() takes 2 arguments but " +
      std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(1)));
  ADT_LET_CONST_REF(lst_size, lst.size());
  std::vector<axpr::Value> func_args;
  func_args.reserve(lst_size);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& elt) -> adt::Result<adt::LoopCtrl> {
        func_args.push_back(elt);
        return adt::Continue{};
      }));
  return interpreter->InterpretCall(args.at(0), func_args);
}

Result<axpr::Value> Length(axpr::InterpreterBase<axpr::Value>* interpreter,
                           const axpr::Value&,
                           const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1)
      << adt::errors::TypeError{std::string() + "len() takes 1 arguments but " +
                                std::to_string(args.size()) + " were given."};
  axpr::Value len_symbol{builtin_symbol::Symbol{builtin_symbol::Length{}}};
  return interpreter->InterpretCall(len_symbol, args);
}

Result<axpr::Value> FlatMap(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
      std::string() + "flat_map() takes 2 arguments but " +
      std::to_string(args.size()) + " were given."};

  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(1)));
  ADT_LET_CONST_REF(lst_size, lst.size());
  adt::List<axpr::Value> ret;
  ret->reserve(lst_size);
  auto Collect = [&](const auto& sub_elt) -> adt::Result<adt::LoopCtrl> {
    ret->emplace_back(sub_elt);
    return adt::Continue{};
  };
  const auto& f = args.at(0);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& elt) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(
            converted_elt,
            interpreter->InterpretCall(f, std::vector<axpr::Value>{elt}));
        ADT_LET_CONST_REF(a_list,
                          AbstractList<axpr::Value>::CastFrom(converted_elt))
            << adt::errors::TypeError{
                   std::string() +
                   "the argument 1 of flat_map() should be a function "
                   "returning a list/SerializableList/MutableList (not a " +
                   axpr::GetTypeName(converted_elt) + ")"};
        ADT_RETURN_IF_ERR(a_list.Visit(Collect));
        return adt::Continue{};
      }));
  return ret;
}

Result<axpr::Value> Filter(axpr::InterpreterBase<axpr::Value>* interpreter,
                           const axpr::Value&,
                           const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
      std::string() + "filter() takes 2 arguments but " +
      std::to_string(args.size()) + " were given."};

  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(1)));
  ADT_LET_CONST_REF(lst_size, lst.size());
  adt::List<axpr::Value> ret;
  ret->reserve(lst_size);
  const auto& f = args.at(0);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& elt) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(
            filter_result,
            interpreter->InterpretCall(f, std::vector<axpr::Value>{elt}));
        ADT_LET_CONST_REF(is_true, BoolHelper{}.ConvertToBool(filter_result));
        if (is_true) {
          ret->emplace_back(elt);
        }
        return adt::Continue{};
      }));
  return ret;
}

Result<axpr::Value> Zip(const axpr::Value&,
                        const std::vector<axpr::Value>& args) {
  std::optional<std::size_t> size;
  for (const auto& arg : args) {
    ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(arg))
        << adt::errors::TypeError{std::string() +
                                  "the argument of 'zip' should be list."};
    ADT_LET_CONST_REF(lst_size, lst.size());
    if (size.has_value()) {
      ADT_CHECK(size.value() == lst_size) << adt::errors::TypeError{
          std::string() + "the arguments of 'zip' should be the same size."};
    } else {
      size = lst_size;
    }
  }
  adt::List<axpr::Value> ret;
  ret->reserve(size.value());
  for (size_t i = 0; i < size.value(); ++i) {
    adt::List<axpr::Value> tuple;
    tuple->reserve(args.size());
    for (const auto& arg : args) {
      ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(arg));
      ADT_LET_CONST_REF(elt, lst.at(i));
      tuple->emplace_back(elt);
    }
    ret->emplace_back(tuple);
  }
  return ret;
}

Result<axpr::Value> Reduce(axpr::InterpreterBase<axpr::Value>* interpreter,
                           const axpr::Value&,
                           const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2 || args.size() == 3) << adt::errors::TypeError{
      std::string() + "'reduce' takes 2 or 3 arguments but " +
      std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(1)));
  std::optional<axpr::Value> init;
  std::optional<int64_t> start;
  ADT_LET_CONST_REF(lst_size, lst.size());
  if (lst_size > 0) {
    ADT_LET_CONST_REF(init_val, lst.at(0));
    init = init_val;
    start = 1;
  } else {
    ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
        std::string() + "reduce() of empty sequence with no initial value"};
    init = args.at(2);
    start = 0;
  }
  ADT_CHECK(init.has_value());
  ADT_CHECK(start.has_value());
  axpr::Value ret{init.value()};
  const auto& f = args.at(0);
  for (size_t i = start.value(); i < lst_size; ++i) {
    ADT_LET_CONST_REF(elt, lst.at(i));
    ADT_LET_CONST_REF(
        cur_reduced,
        interpreter->InterpretCall(f, std::vector<axpr::Value>{elt, ret}));
    ret = cur_reduced;
  }
  return ret;
}

Result<axpr::Value> Max(const axpr::Value&,
                        const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2)
      << adt::errors::TypeError{std::string() + "max() takes 2 arguments but " +
                                std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(lhs, BoolIntDouble::CastFrom(args.at(0)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of max() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  ADT_LET_CONST_REF(rhs, BoolIntDouble::CastFrom(args.at(1)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of max() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  BoolIntDoubleHelper<axpr::Value> helper{};
  ADT_LET_CONST_REF(cmp_ret,
                    helper.template BinaryFunc<ArithmeticGE>(lhs, rhs));
  ADT_LET_CONST_REF(cmp, cmp_ret.template TryGet<bool>());
  return cmp ? args.at(0) : args.at(1);
}

Result<axpr::Value> Min(const axpr::Value&,
                        const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2)
      << adt::errors::TypeError{std::string() + "min() takes 2 arguments but " +
                                std::to_string(args.size()) + " were given."};
  ADT_LET_CONST_REF(lhs, BoolIntDouble::CastFrom(args.at(0)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of min() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  ADT_LET_CONST_REF(rhs, BoolIntDouble::CastFrom(args.at(1)))
      << adt::errors::TypeError{std::string() +
                                "the argument 1 of min() should be 'bool', "
                                "'int' or 'float' (not '" +
                                axpr::GetTypeName(args.at(0)) + "')."};
  BoolIntDoubleHelper<axpr::Value> helper{};
  ADT_LET_CONST_REF(cmp_ret,
                    helper.template BinaryFunc<ArithmeticLE>(lhs, rhs));
  ADT_LET_CONST_REF(cmp, cmp_ret.template TryGet<bool>());
  return cmp ? args.at(0) : args.at(1);
}

Result<axpr::Value> GetAttr(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
      std::string() + "getattr() takes 2 arguments, but " +
      std::to_string(args.size()) + " were given"};
  ADT_LET_CONST_REF(
      ret, interpreter->InterpretCall(builtin_symbol::GetAttr{}, args));
  return ret;
}

Result<axpr::Value> SetAttr(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
      std::string() + "setattr() takes 3 arguments, but " +
      std::to_string(args.size()) + " were given"};
  ADT_LET_CONST_REF(func,
                    interpreter->InterpretCall(builtin_symbol::SetAttr{},
                                               {args.at(0), args.at(1)}));
  ADT_LET_CONST_REF(ret,
                    interpreter->InterpretCall(func, {args.at(1), args.at(2)}));
  return ret;
}

}  // namespace ap::axpr

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

#include <sstream>
#include <string>
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/string_util.h"

namespace ap::axpr {

template <typename ValueT>
struct StringMethodClass {
  using This = StringMethodClass;
  using Self = std::string;
  using Val = ValueT;

  adt::Result<Val> ToString(const Self& self) { return self; }

  adt::Result<Val> Hash(const Self& self) {
    return static_cast<int64_t>(std::hash<std::string>()(self));
  }

  adt::Result<Val> GetAttr(const Self& self, const Val& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "replace") {
      return axpr::Method<Val>{self, &This::StaticReplace};
    }

    if (attr_name == "join") {
      return axpr::Method<Val>{self,
                               &axpr::WrapAsBuiltinFuncType<This, &This::Join>};
    }
    return adt::errors::TypeError{};
  }

  adt::Result<Val> Join(const Self& self, const std::vector<Val>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "join() takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(lst, args.at(0).template TryGet<adt::List<Val>>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of join() should be 'list' not '" +
               axpr::GetTypeName(args.at(0)) + "'."};
    std::ostringstream ss;
    int i = 0;
    for (const auto& elt : *lst) {
      ADT_LET_CONST_REF(item, elt.template TryGet<std::string>())
          << adt::errors::TypeError{std::string() + "sequence item " +
                                    std::to_string(i) +
                                    ": expected str instance, " +
                                    axpr::GetTypeName(elt) + " found"};
      if (i++ > 0) {
        ss << self;
      }
      ss << item;
    }
    return ss.str();
  }

  static adt::Result<Val> StaticReplace(const Val& self_val,
                                        const std::vector<Val>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "'str.replace' takes 2 arguments but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(pattern, args.at(0).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of 'str.replace' should be a str"};
    ADT_LET_CONST_REF(replacement, args.at(1).template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of 'str.replace' should be a str"};
    return This{}.Replace(self, pattern, replacement);
  }

  std::string Replace(std::string self,
                      const std::string& pattern,
                      const std::string& replacement) {
    while (true) {
      std::size_t pos = self.find(pattern);
      if (pos == std::string::npos) {
        break;
      }
      self = self.replace(pos, pattern.size(), replacement);
    }
    return self;
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<Val> GetBuiltinBinaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BuiltinBinarySymbol>::convertible) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BuiltinBinarySymbol>::arithmetic_op_type;
      return &This::template BinaryFunc<ArithmeticOp>;
    } else {
      return adt::Nothing{};
    }
  }

  template <typename ArithmeticOp>
  static adt::Result<Val> BinaryFunc(const Val& lhs_val, const Val& rhs_val) {
    const auto& opt_lhs = lhs_val.template TryGet<std::string>();
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    return BuiltinStringBinary<ArithmeticOp>(lhs, rhs_val);
  }
};

template <typename Val>
struct MethodClassImpl<Val, std::string> : public StringMethodClass<Val> {};

template <typename Val>
struct MethodClassImpl<Val, TypeImpl<std::string>>
    : public EmptyMethodClass<Val> {};

}  // namespace ap::axpr

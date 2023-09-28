// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/print_value.h"
#include "paddle/cinn/adt/print_constant.h"
#include "paddle/cinn/adt/print_equations.h"

namespace cinn::adt {

namespace {

std::string ToTxtString(const tPointer<UniqueId>& value) {
  std::size_t value_unique_id = value.value().unique_id();
  return "ptr_" + std::to_string(value_unique_id);
}

struct ToTxtStringStruct {
  std::string operator()(const Undefined& value) { return "undefined"; }

  std::string operator()(const Ok& value) { return "ok"; }

  std::string operator()(const Iterator& value) {
    std::string ret;
    ret += ToTxtString(value);
    return ret;
  }

  std::string operator()(const Constant& value) {
    std::string ret;
    ret += ToTxtString(value);
    return ret;
  }

  std::string operator()(const List<Value>& value_list) {
    std::string ret;
    ret += "List(";

    for (std::size_t idx = 0; idx < value_list->size(); ++idx) {
      if (idx != 0) {
        ret += ", ";
      }
      ret += ToTxtString(value_list.Get(idx));
    }

    ret += ")";
    return ret;
  }

  std::string operator()(const IndexDot<Value, Constant>& value) {
    std::string ret;
    const auto& [_, constant] = value.tuple();
    const Value& value_ = value.GetIteratorsValue();
    ret +=
        "IndexDot(" + ToTxtString(value_) + ", " + ToTxtString(constant) + ")";
    return ret;
  }

  std::string operator()(const IndexUnDot<Value, Constant>& value) {
    std::string ret;
    const auto& [_, constant] = value.tuple();
    const Value& value_ = value.GetIndexValue();
    ret += "IndexUndot(" + ToTxtString(value_) + ", " + ToTxtString(constant) +
           ")";
    return ret;
  }

  std::string operator()(const ConstantAdd<Value>& value) {
    std::string ret;
    const auto& [_, constant] = value.tuple();
    const Value& value_ = value.GetArg0();
    ret += "ConstantAdd(" + ToTxtString(value_) + ", " + ToTxtString(constant) +
           ")";
    return ret;
  }

  std::string operator()(const ConstantDiv<Value>& value) {
    std::string ret;
    const auto& [_, constant] = value.tuple();
    const Value& value_ = value.GetArg0();
    ret += "ConstantDiv(" + ToTxtString(value_) + ", " + ToTxtString(constant) +
           ")";
    return ret;
  }

  std::string operator()(const ConstantMod<Value>& value) {
    std::string ret;
    const auto& [_, constant] = value.tuple();
    const Value& value_ = value.GetArg0();
    ret += "ConstantMod(" + ToTxtString(value_) + ", " + ToTxtString(constant) +
           ")";
    return ret;
  }

  std::string operator()(const ListGetItem<Value, Constant>& value) {
    std::string ret;
    const auto& [_, constant] = value.tuple();
    const Value& value_ = value.GetList();
    ret += "ListGetItem(" + ToTxtString(value_) + ", " + ToTxtString(constant) +
           ")";
    return ret;
  }

  std::string operator()(const PtrGetItem<Value>& value) {
    std::string ret;
    const auto& [ptr_tag, _] = value.tuple();
    const Value& value_ = value.GetArg1();
    ret +=
        "PtrGetItem(" + ToTxtString(ptr_tag) + ", " + ToTxtString(value_) + ")";
    return ret;
  }
};

}  // namespace

std::string ToTxtString(const Value& value) {
  return std::visit(ToTxtStringStruct{}, value.variant());
}

}  // namespace cinn::adt

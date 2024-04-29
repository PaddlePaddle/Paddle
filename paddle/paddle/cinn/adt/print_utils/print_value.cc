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

#include "paddle/cinn/adt/print_utils/print_value.h"

#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/print_utils/print_dim_expr.h"
#include "paddle/cinn/adt/print_utils/print_equations.h"

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

  std::string operator()(const DimExpr& value) {
    std::string ret;
    ret += ToString(value);
    return ret;
  }

  std::string operator()(const List<Value>& value_list) {
    std::string ret;
    ret += "[";

    for (std::size_t idx = 0; idx < value_list->size(); ++idx) {
      if (idx != 0) {
        ret += ", ";
      }
      ret += ToTxtString(value_list.Get(idx));
    }

    ret += "]";
    return ret;
  }

  std::string operator()(const IndexDotValue<Value, List<DimExpr>>& value) {
    std::string ret;
    const auto& [iters, constant] = value.tuple();
    ret +=
        "IndexDot(" + ToTxtString(iters) + ", " + ToTxtString(constant) + ")";
    return ret;
  }

  std::string operator()(const IndexUnDotValue<Value, List<DimExpr>>& value) {
    std::string ret;
    const auto& [_, constant] = value.tuple();
    const Value& value_ = value.GetIndexValue();
    ret += "IndexUnDot(" + ToTxtString(value_) + ", " + ToTxtString(constant) +
           ")";
    return ret;
  }

  std::string operator()(const ListGetItem<Value, DimExpr>& list_get_item) {
    std::string ret;
    const auto& [value, constant] = list_get_item.tuple();
    ret +=
        "ListGetItem(" + ToTxtString(value) + ", " + ToString(constant) + ")";
    return ret;
  }

  std::string operator()(const BroadcastedIterator<Value, DimExpr>& broadcast) {
    std::string ret;
    const auto& [value, constant] = broadcast.tuple();
    ret += "BI(" + ToTxtString(value) + ", " + ToString(constant) + ")";
    return ret;
  }

  std::string operator()(const PtrGetItem<Value>& ptr_get_item) {
    std::string ret;
    const auto& [ptr_tag, value] = ptr_get_item.tuple();
    ret +=
        "PtrGetItem(" + ToTxtString(ptr_tag) + ", " + ToTxtString(value) + ")";
    return ret;
  }
};

}  // namespace

std::string ToTxtString(const Value& value) {
  return std::visit(ToTxtStringStruct{}, value.variant());
}

std::string ToTxtString(const std::optional<Value>& opt_value) {
  if (opt_value.has_value()) {
    return ToTxtString(opt_value.value());
  } else {
    return "";
  }
}
}  // namespace cinn::adt

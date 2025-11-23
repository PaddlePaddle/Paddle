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

#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/starred.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, adt::List<ValueT>> {
  using This = MethodClassImpl<ValueT, adt::List<ValueT>>;
  using Self = adt::List<ValueT>;

  adt::Result<ValueT> Length(const Self& self) {
    return static_cast<int64_t>(self->size());
  }

  adt::Result<ValueT> ToString(axpr::InterpreterBase<ValueT>* interpreter,
                               const Self& self) {
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    using Ok = adt::Result<adt::Ok>;
    for (const auto& elt : *self) {
      if (i++ > 0) {
        ss << ", ";
      }
      const auto& func = MethodClass<ValueT>::ToString(elt);
      ADT_RETURN_IF_ERR(func.Match(
          [&](const adt::Nothing&) -> Ok {
            return adt::errors::TypeError{GetTypeName(elt) +
                                          " class has no __str__ method"};
          },
          [&](adt::Result<ValueT> (*unary_func)(const ValueT&)) -> Ok {
            ADT_LET_CONST_REF(str_val, unary_func(elt));
            ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>())
                << adt::errors::TypeError{
                       std::string() + "'" + axpr::GetTypeName(elt) +
                       ".__str__ should return a 'str' but '" +
                       axpr::GetTypeName(str_val) + "' were returned."};
            ss << str;
            return adt::Ok{};
          },
          [&](adt::Result<ValueT> (*unary_func)(axpr::InterpreterBase<ValueT>*,
                                                const ValueT&)) -> Ok {
            ADT_LET_CONST_REF(str_val, unary_func(interpreter, elt));
            ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>())
                << adt::errors::TypeError{
                       std::string() + "'" + axpr::GetTypeName(elt) +
                       ".__str__ should return a 'str' but '" +
                       axpr::GetTypeName(str_val) + "' were returned."};
            ss << str;
            return adt::Ok{};
          }));
    }
    ss << "]";
    return ss.str();
  }

  static Result<ValueT> EQ(const ValueT& lhs_val, const ValueT& rhs_val) {
    ADT_LET_CONST_REF(lhs, lhs_val.template TryGet<Self>());
    ADT_LET_CONST_REF(rhs, rhs_val.template TryGet<Self>());
    return lhs == rhs;
  }

  adt::Result<ValueT> Hash(axpr::InterpreterBase<ValueT>* interpreter,
                           const Self& self) {
    int64_t hash_value = 0;
    using Ok = adt::Result<adt::Ok>;
    for (const auto& elt : *self) {
      const auto& func = MethodClass<ValueT>::Hash(elt);
      ADT_RETURN_IF_ERR(func.Match(
          [&](const adt::Nothing&) -> Ok {
            return adt::errors::TypeError{std::string() + GetTypeName(elt) +
                                          " class has no __hash__ method"};
          },
          [&](adt::Result<ValueT> (*unary_func)(const ValueT&)) -> Ok {
            ADT_LET_CONST_REF(elt_hash_val, unary_func(elt));
            ADT_LET_CONST_REF(elt_hash, elt_hash_val.template TryGet<int64_t>())
                << adt::errors::TypeError{
                       std::string() + "'" + axpr::GetTypeName(elt) +
                       ".__hash__ should return a 'int' but '" +
                       axpr::GetTypeName(elt_hash_val) + "' were returned."};
            hash_value = adt::hash_combine(hash_value, elt_hash);
            return adt::Ok{};
          },
          [&](adt::Result<ValueT> (*unary_func)(axpr::InterpreterBase<ValueT>*,
                                                const ValueT&)) -> Ok {
            ADT_LET_CONST_REF(elt_hash_val, unary_func(interpreter, elt));
            ADT_LET_CONST_REF(elt_hash, elt_hash_val.template TryGet<int64_t>())
                << adt::errors::TypeError{
                       std::string() + "'" + axpr::GetTypeName(elt) +
                       ".__hash__ should return a 'int' but '" +
                       axpr::GetTypeName(elt_hash_val) + "' were returned."};
            hash_value = adt::hash_combine(hash_value, elt_hash);
            return adt::Ok{};
          }));
    }
    return hash_value;
  }

  adt::Result<ValueT> GetItem(const Self& self, const ValueT& idx) {
    return idx.Match(
        [&](int64_t index) -> Result<ValueT> {
          if (index < 0) {
            index += self->size();
          }
          if (index >= 0 && index < static_cast<int64_t>(self->size())) {
            return self->at(index);
          }
          return adt::errors::IndexError{"list index out of range"};
        },
        [&](const auto&) -> Result<ValueT> {
          return adt::errors::TypeError{std::string() +
                                        "list indices must be integers, not " +
                                        axpr::GetTypeName(idx)};
        });
  }

  adt::Result<ValueT> Starred(const Self& self) {
    return ap::axpr::Starred<ValueT>{self};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<adt::List<ValueT>>> {
  using Self = TypeImpl<adt::List<ValueT>>;

  using This = MethodClassImpl<ValueT, Self>;
};

}  // namespace ap::axpr

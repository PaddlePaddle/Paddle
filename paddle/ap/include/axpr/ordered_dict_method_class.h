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

#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/ordered_dict.h"
#include "paddle/ap/include/axpr/to_string.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, OrderedDict<ValueT>> {
  using Val = ValueT;
  using This = MethodClassImpl<ValueT, OrderedDict<ValueT>>;
  using Self = OrderedDict<ValueT>;

  adt::Result<ValueT> Length(const Self& self) {
    return static_cast<int64_t>(self->items().size());
  }

  adt::Result<ValueT> ToString(axpr::InterpreterBase<ValueT>* interpreter,
                               const Self& self) {
    std::ostringstream ss;
    ss << "OrderedDict([";
    int i = 0;
    for (const auto& [k, v] : self->items()) {
      if (i++ > 0) {
        ss << ", ";
      }
      ADT_LET_CONST_REF(key_str, axpr::ToString(interpreter, k));
      ADT_LET_CONST_REF(value_str, axpr::ToString(interpreter, v));
      ss << "[" << key_str << ", " << value_str << "]";
    }
    ss << "])";
    return ss.str();
  }

  adt::Result<ValueT> Hash(axpr::InterpreterBase<ValueT>* interpreter,
                           const Self& self) {
    int64_t hash_value = 0;
    for (const auto& [k, v] : self->items()) {
      ADT_LET_CONST_REF(key_hash_value, axpr::Hash<ValueT>{}(interpreter, k));
      ADT_LET_CONST_REF(value_hash_value, axpr::Hash<ValueT>{}(interpreter, v));
      hash_value = adt::hash_combine(hash_value, key_hash_value);
      hash_value = adt::hash_combine(hash_value, value_hash_value);
    }
    return hash_value;
  }

  adt::Result<ValueT> GetItem(axpr::InterpreterBase<ValueT>* interpreter,
                              const Self& self,
                              const ValueT& key) {
    ADT_LET_CONST_REF(val, self->At(interpreter, key))
        << adt::errors::KeyError{axpr::ToDebugString(interpreter, key)};
    return val;
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "items") {
      return axpr::Method<ValueT>{
          self, &axpr::WrapAsBuiltinFuncType<This, &This::Items>};
    }
    if (attr_name == "contains") {
      return axpr::Method<ValueT>{self, &This::Contains};
    }
    return adt::errors::TypeError{std::string() +
                                  "OrderedDict object has no attribute '" +
                                  attr_name + "'"};
  }

  static adt::Result<ValueT> Contains(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "OrderedDict.contains() takes 1 argument, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(has_elt, self->Has(interpreter, args.at(0)));
    return has_elt;
  }

  adt::Result<ValueT> Items(const Self& self, const std::vector<ValueT>&) {
    adt::List<ValueT> lst;
    lst->reserve(self->items().size());
    for (const auto& [first, second] : self->items()) {
      lst->emplace_back(adt::List<ValueT>{first, second});
    }
    return lst;
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<OrderedDict<ValueT>>> {
  using Val = ValueT;
  using Self = TypeImpl<OrderedDict<ValueT>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> Call(const Self& self) {
    return axpr::Method<ValueT>{self, &This::Construct};
  }

  static adt::Result<ValueT> Construct(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT&,
      const std::vector<ValueT>& args) {
    if (args.size() == 0) {
      return OrderedDict<ValueT>{};
    }
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "OrderedDict() takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(lst, args.at(0).template TryGet<adt::List<ValueT>>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of OrderedDict() should be list, " +
               axpr::GetTypeName(args.at(0)) + " found."};
    OrderedDict<ValueT> ordered_dict{};
    int i = 0;
    for (const auto& elt : *lst) {
      ADT_LET_CONST_REF(pair, elt.template TryGet<adt::List<ValueT>>())
          << adt::errors::TypeError{std::string() + "sequence item " +
                                    std::to_string(i) +
                                    " : expected list instance, " +
                                    axpr::GetTypeName(elt) + " found."};
      ADT_CHECK(pair->size() == 2) << adt::errors::TypeError{
          std::string() + "sequence item " + std::to_string(i) +
          " : expected 2-item list, " + std::to_string(pair->size()) +
          "-item list found."};
      ADT_RETURN_IF_ERR(
          ordered_dict->Insert(interpreter, pair->at(0), pair->at(1)));
      ++i;
    }
    return ordered_dict;
  }
};

}  // namespace ap::axpr

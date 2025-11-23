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
#include "paddle/ap/include/axpr/method.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/mutable_list.h"
#include "paddle/ap/include/axpr/starred.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, MutableList<ValueT>> {
  using This = MethodClassImpl<ValueT, MutableList<ValueT>>;
  using Self = MutableList<ValueT>;

  adt::Result<ValueT> Length(const Self& self) {
    ADT_LET_CONST_REF(vec, self.Get());
    return static_cast<int64_t>(vec->size());
  }

  adt::Result<ValueT> ToString(axpr::InterpreterBase<ValueT>* interpreter,
                               const Self& self) {
    ADT_LET_CONST_REF(vec, self.Get());
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    using Ok = adt::Result<adt::Ok>;
    for (const auto& elt : *vec) {
      if (i++ > 0) {
        ss << ", ";
      }
      const auto& func = MethodClass<ValueT>::ToString(elt);
      ADT_RETURN_IF_ERR(func.Match(
          [&](const adt::Nothing&) -> Ok {
            return adt::errors::TypeError{GetTypeName(elt) +
                                          " class has no __str__ function"};
          },
          [&](adt::Result<ValueT> (*unary_func)(const ValueT&)) -> Ok {
            ADT_LET_CONST_REF(str_val, unary_func(elt));
            ADT_LET_CONST_REF(str, str_val.template TryGet<std::string>())
                << adt::errors::TypeError{
                       std::string() + "'" + axpr::GetTypeName(elt) +
                       ".__builtin_ToString__ should return a 'str' but '" +
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
                       ".__builtin_ToString__ should return a 'str' but '" +
                       axpr::GetTypeName(str_val) + "' were returned."};
            ss << str;
            return adt::Ok{};
          }));
    }
    ss << "]";
    return ss.str();
  }

  adt::Result<ValueT> Hash(const Self& self) {
    return adt::errors::TypeError{"MutableList objects are not hashable"};
  }

  adt::Result<ValueT> GetItem(const Self& self, const ValueT& idx) {
    ADT_LET_CONST_REF(vec, self.Get());
    return idx.Match(
        [&](int64_t index) -> Result<ValueT> {
          if (index < 0) {
            index += vec->size();
          }
          if (index >= 0 && index < static_cast<int64_t>(vec->size())) {
            return vec->at(index);
          }
          return adt::errors::IndexError{"list index out of range"};
        },
        [&](const auto&) -> Result<ValueT> {
          return adt::errors::TypeError{std::string() +
                                        "list indices must be integers, not " +
                                        axpr::GetTypeName(idx)};
        });
  }

  adt::Result<ValueT> SetItem(const Self& self, const ValueT& idx) {
    return Method<ValueT>{self, &This::StaticSetItem};
  }

  static adt::Result<ValueT> StaticSetItem(const ValueT& self_val,
                                           const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(const_idx, args.at(0).template TryGet<int64_t>())
        << adt::errors::TypeError{
               std::string() + "list indices must be integers or slices, not " +
               axpr::GetTypeName(args.at(0))};
    ADT_LET_CONST_REF(self_ptr, self.Mut());
    int64_t idx = const_idx;
    if (idx < 0) {
      idx += self_ptr->size();
    }
    ADT_CHECK(idx < static_cast<int64_t>(self_ptr->size()))
        << adt::errors::IndexError{"list index out of range"};
    self_ptr->at(idx) = args.at(1);
    return adt::Nothing{};
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "append") {
      return Method<ValueT>{self, &This::StaticAppend};
    }
    return adt::errors::AttributeError{
        std::string() + "'MutableList' object has no attribute '" + attr_name +
        "'"};
  }

  static adt::Result<ValueT> StaticAppend(const ValueT& self_val,
                                          const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template TryGet<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "MutableList.append() takes exactly one argument (" +
        std::to_string(args.size()) + " given)"};
    ADT_LET_CONST_REF(self_ptr, self.Mut());
    self_ptr->push_back(args.at(0));
    return adt::Nothing{};
  }

  adt::Result<ValueT> Starred(const Self& self) {
    ADT_LET_CONST_REF(vec, self.Get());
    adt::List<ValueT> ret{};
    ret->reserve(vec->size());
    ret->assign(vec->begin(), vec->end());
    return ap::axpr::Starred<ValueT>{ret};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<MutableList<ValueT>>> {
  using Self = TypeImpl<MutableList<ValueT>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> Call(const Self&) { return &This::StaticConstruct; }

  static adt::Result<ValueT> StaticConstruct(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT&,
      const std::vector<ValueT>& args) {
    return This{}.Construct(interpreter, args);
  }

  adt::Result<ValueT> Construct(axpr::InterpreterBase<ValueT>* interpreter,
                                const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(ref_lst,
                      adt::WeakPtrLock(interpreter->circlable_ref_list()));
    const auto& mut_list = MutableList<ValueT>::Make(
        ref_lst, std::make_shared<std::vector<ValueT>>());
    ADT_LET_CONST_REF(ptr, mut_list.Mut());
    *ptr = args;
    return mut_list;
  }
};

}  // namespace ap::axpr

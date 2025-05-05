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

#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/builtin_high_order_func_type.h"
#include "paddle/ap/include/axpr/class_attrs_helper.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/method.h"
#include "paddle/ap/include/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, BuiltinClassInstance<ValueT>> {
  using Val = ValueT;
  using Self = BuiltinClassInstance<ValueT>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> Hash(InterpreterBase<ValueT>* interpreter,
                           const Self& self) {
    const auto& opt_func = GetClassAttr(self, "__hash__");
    ADT_CHECK(opt_func.has_value()) << adt::errors::TypeError{
        std::string() + self.type.class_attrs()->class_name +
        ".__hash__() not implemented"};
    using RetT = adt::Result<ValueT>;
    static std::vector<ValueT> empty_args{};
    return opt_func.value().Match(
        [&](BuiltinFuncType<ValueT> unary_func) -> RetT {
          return unary_func(self, empty_args);
        },
        [&](BuiltinHighOrderFuncType<ValueT> unary_func) -> RetT {
          return unary_func(interpreter, self, empty_args);
        },
        [&](const axpr::Method<ValueT>& method) -> RetT {
          return interpreter->InterpretCall(method, {});
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() + "casting to builtin function (not " +
              GetTypeName(opt_func.value()) + ") failed."};
        });
  }

  adt::Result<ValueT> ToString(InterpreterBase<ValueT>* interpreter,
                               const Self& self) {
    const auto& opt_func = GetClassAttr(self, "__str__");
    ADT_CHECK(opt_func.has_value()) << adt::errors::TypeError{
        std::string() + self.type.class_attrs()->class_name +
        ".__str__() not implemented"};
    using RetT = adt::Result<ValueT>;
    static std::vector<ValueT> empty_args{};
    return opt_func.value().Match(
        [&](BuiltinFuncType<ValueT> unary_func) -> RetT {
          return unary_func(self, empty_args);
        },
        [&](BuiltinHighOrderFuncType<ValueT> unary_func) -> RetT {
          return unary_func(interpreter, self, empty_args);
        },
        [&](const axpr::Method<ValueT>& method) -> RetT {
          return interpreter->InterpretCall(method, {});
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() + "casting to builtin function (not " +
              GetTypeName(opt_func.value()) + ") failed."};
        });
  }

  adt::Result<ValueT> GetAttr(InterpreterBase<ValueT>* interpreter,
                              const Self& self,
                              const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    const auto& opt_val = GetClassAttr(self, attr_name);
    if (opt_val.has_value()) {
      return opt_val.value();
    }
    const auto& opt_gettattr = GetClassAttr(self, "__getattr__");
    const auto& class_attrs = self.type.class_attrs();
    ADT_CHECK(opt_gettattr.has_value())
        << adt::errors::AttributeError{std::string() + class_attrs->class_name +
                                       " class has no attribute '__getattr__'"};
    std::vector<ValueT> args{attr_name_val};
    ADT_LET_CONST_REF(ret,
                      interpreter->InterpretCall(opt_gettattr.value(), args));
    return ret;
  }

  adt::Result<ValueT> EQ(const Self& self, const ValueT& rhs_val) {
    ADT_LET_CONST_REF(ret, Equals(self, rhs_val));
    return ret;
  }

  adt::Result<ValueT> NE(const Self& self, const ValueT& rhs_val) {
    ADT_LET_CONST_REF(ret, Equals(self, rhs_val));
    return !ret;
  }

  adt::Result<bool> Equals(const Self& self, const ValueT& rhs_val) {
    const auto* class_ops = self.type.class_ops();
    return class_ops->Equals(self, rhs_val);
  }

  adt::Result<ValueT> GetItem(InterpreterBase<ValueT>* interpreter,
                              const Self& self,
                              const ValueT& idx_val) {
    const auto& opt_getitem = GetClassAttr(self, "__getitem__");
    const auto& class_attrs = self.type.class_attrs();
    ADT_CHECK(opt_getitem.has_value())
        << adt::errors::AttributeError{std::string() + class_attrs->class_name +
                                       " class has no attribute '__getitem__'"};
    std::vector<ValueT> args{idx_val};
    ADT_LET_CONST_REF(ret,
                      interpreter->InterpretCall(opt_getitem.value(), args));
    return ret;
  }

  adt::Result<ValueT> Call(const Self& self) {
    const auto& opt_func = GetClassAttr(self, "__call__");
    const auto& class_attrs = self.type.class_attrs();
    ADT_CHECK(opt_func.has_value())
        << adt::errors::AttributeError{std::string() + class_attrs->class_name +
                                       " class has no attribute '__call__'"};
    return opt_func.value();
  }

  adt::Result<ValueT> Starred(const Self& self) {
    const auto& opt_func = GetClassAttr(self, "__starred__");
    const auto& class_attrs = self.type.class_attrs();
    ADT_CHECK(opt_func.has_value())
        << adt::errors::AttributeError{std::string() + class_attrs->class_name +
                                       " class has no attribute '__starred__'"};
    return opt_func.value();
  }

  std::optional<ValueT> GetClassAttr(const Self& self,
                                     const std::string& attr_name) {
    const auto& class_attrs = self.type.class_attrs();
    const auto& opt_func =
        ClassAttrsHelper<ValueT, ValueT>{}.OptGet(class_attrs, attr_name);
    if (!opt_func.has_value()) {
      return std::nullopt;
    }
    using RetT = ValueT;
    return opt_func.value().Match(
        [&](const BuiltinFuncType<ValueT>& f) -> RetT {
          return Method<ValueT>{self, f};
        },
        [&](const BuiltinHighOrderFuncType<ValueT>& f) -> RetT {
          return Method<ValueT>{self, f};
        },
        [&](const auto&) -> RetT { return opt_func.value(); });
  }

  adt::Result<ValueT> SetAttr(const Self& self, const ValueT& attr_name_val) {
    const auto& class_attrs = self.type.class_attrs();
    const auto& opt_func = GetClassAttr(self, "__setattr__");
    ADT_CHECK(opt_func.has_value())
        << adt::errors::AttributeError{std::string() + class_attrs->class_name +
                                       " class has no attribute '__setattr__'"};
    return opt_func.value();
  }

  adt::Result<ValueT> SetItem(const Self& self, const ValueT& idx_val) {
    const auto& class_attrs = self.type.class_attrs();
    const auto& opt_func = GetClassAttr(self, "__setitem__");
    ADT_CHECK(opt_func.has_value())
        << adt::errors::AttributeError{std::string() + class_attrs->class_name +
                                       " class has no attribute '__setitem__'"};
    return opt_func.value();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<BuiltinClassInstance<ValueT>>> {
  using Val = ValueT;
  using Self = TypeImpl<BuiltinClassInstance<ValueT>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    ADT_LET_CONST_REF(attr_val, self.class_attrs()->attrs->Get(attr_name))
        << adt::errors::AttributeError{
               std::string() + "type '" + self.class_attrs()->class_name +
               "' has no attribute '" + attr_name + "'"};
    return attr_val;
  }

  adt::Result<ValueT> Call(const Self& self) {
    ValueT func{&This::StaticConstruct};
    return Method<ValueT>{self, func};
  }

  adt::Result<ValueT> ToString(const Self& self) {
    return std::string() + "<class '" + self.class_attrs()->class_name + "'>";
  }

  adt::Result<ValueT> Hash(const Self& self) {
    return reinterpret_cast<int64_t>(self.class_attrs());
  }

  static adt::Result<ValueT> StaticConstruct(
      axpr::InterpreterBase<ValueT>* interpreter,
      const ValueT& self_val,
      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, axpr::TryGetTypeImpl<Self>(self_val));
    return This{}.Construct(interpreter, self, args);
  }

  adt::Result<ValueT> Construct(axpr::InterpreterBase<ValueT>* interpreter,
                                const Self& self,
                                const std::vector<ValueT>& args) {
    const auto* class_ops = self.class_ops();
    const auto& class_attrs = class_ops->class_attrs();
    TypeImpl<BuiltinClassInstance<ValueT>> type(class_ops);
    BuiltinClassInstance<ValueT> empty_instance{type, std::nullopt};
    const auto& init_func =
        ClassAttrsHelper<ValueT, ValueT>{}.OptGet(class_attrs, "__init__");
    ADT_CHECK(init_func.has_value())
        << adt::errors::TypeError{std::string() + class_attrs->class_name +
                                  " class has no __init__ function"};
    Method<ValueT> f{empty_instance, init_func.value()};
    ADT_LET_CONST_REF(ret_instance, interpreter->InterpretCall(f, args));
    return ret_instance;
  }
};

}  // namespace ap::axpr

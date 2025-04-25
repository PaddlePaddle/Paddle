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

#include "paddle/ap/include/axpr/anf_expr_helper.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/serializable_value.h"

namespace ap::axpr {

struct SerializableValueHelper {
  template <typename ValueT>
  adt::Result<SerializableValue> CastFrom(const ValueT& val) {
    using RetT = adt::Result<SerializableValue>;
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    return val.Match(
        [&](const TypeT& type) -> RetT {
          return type.Match(
              [](const TypeImpl<adt::Nothing>& impl) -> RetT { return impl; },
              [](const TypeImpl<bool>& impl) -> RetT { return impl; },
              [](const TypeImpl<int64_t>& impl) -> RetT { return impl; },
              [](const TypeImpl<double>& impl) -> RetT { return impl; },
              [](const TypeImpl<std::string>& impl) -> RetT { return impl; },
              [](const TypeImpl<ClassInstance<ValueT>>& impl) -> RetT {
                return impl.class_attrs;
              },
              [&](const auto&) -> RetT {
                std::ostringstream ss;
                ss << "Builtin serializable types are: ";
                ss << SerializableValue::SerializableTypeNames();
                ss << " (not include '" << axpr::GetTypeName(val) << "').";
                return adt::errors::ValueError{ss.str()};
              });
        },
        [](const Nothing& impl) -> RetT { return impl; },
        [](bool impl) -> RetT { return impl; },
        [](int64_t impl) -> RetT { return impl; },
        [](double impl) -> RetT { return impl; },
        [](const std::string& impl) -> RetT { return impl; },
        [](const Function<SerializableValue>& impl) -> RetT { return impl; },
        [](const adt::List<SerializableValue>& impl) -> RetT { return impl; },
        [](const AttrMap<SerializableValue>& impl) -> RetT { return impl; },
        [&](const adt::List<ValueT>& list) -> RetT {
          return CastListFrom(list);
        },
        [&](const AttrMap<ValueT>& object) -> RetT {
          return CastObjectFrom(object);
        },
        [&](const BuiltinFuncType<ValueT>& func) -> RetT {
          auto* func_ptr = reinterpret_cast<void*>(func);
          ADT_CHECK(BuiltinFuncNameMgr::Singleton()->Has(func_ptr));
          return BuiltinFuncVoidPtr{func_ptr};
        },
        [&](const BuiltinHighOrderFuncType<ValueT>& func) -> RetT {
          auto* func_ptr = reinterpret_cast<void*>(func);
          ADT_CHECK(BuiltinFuncNameMgr::Singleton()->Has(func_ptr));
          return BuiltinHighOrderFuncVoidPtr{func_ptr};
        },
        [&](const auto&) -> RetT {
          std::ostringstream ss;
          ss << "Builtin serializable types are: ";
          ss << SerializableValue::SerializableTypeNames();
          ss << " (not include '" << axpr::GetTypeName(val) << "').";
          return adt::errors::ValueError{ss.str()};
        });
  }

  adt::Result<int64_t> Hash(const SerializableValue& val) {
    using RetT = adt::Result<int64_t>;
    return val.Match(
        [](const TypeImpl<adt::Nothing>&) -> RetT {
          int64_t hash_value =
              std::hash<const char*>()(typeid(TypeImpl<adt::Nothing>).name());
          return hash_value;
        },
        [](const TypeImpl<bool>&) -> RetT {
          int64_t hash_value =
              std::hash<const char*>()(typeid(TypeImpl<bool>).name());
          return hash_value;
        },
        [](const TypeImpl<int64_t>&) -> RetT {
          int64_t hash_value =
              std::hash<const char*>()(typeid(TypeImpl<int64_t>).name());
          return hash_value;
        },
        [](const TypeImpl<double>&) -> RetT {
          int64_t hash_value =
              std::hash<const char*>()(typeid(TypeImpl<double>).name());
          return hash_value;
        },
        [](const TypeImpl<std::string>&) -> RetT {
          int64_t hash_value =
              std::hash<const char*>()(typeid(TypeImpl<std::string>).name());
          return hash_value;
        },
        [](const ClassAttrs<SerializableValue>& class_attrs) -> RetT {
          return reinterpret_cast<int64_t>(class_attrs.shared_ptr().get());
        },
        [](const adt::Nothing&) -> RetT { return static_cast<int64_t>(0); },
        [](bool c) -> RetT { return static_cast<int64_t>(c); },
        [](int64_t c) -> RetT { return c; },
        [](double c) -> RetT {
          return static_cast<int64_t>(std::hash<double>()(c));
        },
        [](const std::string& c) -> RetT {
          return static_cast<int64_t>(std::hash<std::string>()(c));
        },
        [](const Function<SerializableValue>& impl) -> RetT {
          return impl->GetHashValue();
        },
        [&](const adt::List<SerializableValue>& lst) -> RetT {
          return HashImpl(lst);
        },
        [&](const axpr::AttrMap<SerializableValue>& obj) -> RetT {
          return HashImpl(obj);
        },
        [&](const BuiltinFuncVoidPtr& func) -> RetT {
          return reinterpret_cast<int64_t>(func.func_ptr);
        },
        [&](const BuiltinHighOrderFuncVoidPtr& func) -> RetT {
          return reinterpret_cast<int64_t>(func.func_ptr);
        });
  }

  adt::Result<int64_t> HashImpl(const adt::List<SerializableValue>& lst) {
    int64_t hash_value = 0;
    for (const auto& elt : *lst) {
      ADT_LET_CONST_REF(elt_hash, Hash(elt));
      hash_value = adt::hash_combine(hash_value, elt_hash);
    }
    return hash_value;
  }

  adt::Result<int64_t> HashImpl(
      const axpr::AttrMap<SerializableValue>& object) {
    return reinterpret_cast<int64_t>(object.shared_ptr().get());
  }

  adt::Result<std::string> ToString(const SerializableValue& val) {
    using RetT = adt::Result<std::string>;
    return val.Match(
        [](const TypeImpl<adt::Nothing>&) -> RetT {
          return TypeImpl<adt::Nothing>{}.Name();
        },
        [](const TypeImpl<bool>&) -> RetT { return TypeImpl<bool>{}.Name(); },
        [](const TypeImpl<int64_t>&) -> RetT {
          return TypeImpl<int64_t>{}.Name();
        },
        [](const TypeImpl<double>&) -> RetT {
          return TypeImpl<double>{}.Name();
        },
        [](const TypeImpl<std::string>&) -> RetT {
          return TypeImpl<std::string>{}.Name();
        },
        [](const ClassAttrs<SerializableValue>& class_attrs) -> RetT {
          return std::string() + "<class '" + class_attrs->class_name + "'>";
        },
        [](const adt::Nothing&) -> RetT { return "None"; },
        [](bool c) -> RetT { return std::string(c ? "True" : "False"); },
        [](int64_t c) -> RetT { return std::to_string(c); },
        [](double c) -> RetT { return std::to_string(c); },
        [](const std::string& c) -> RetT {
          std::ostringstream ss;
          ss << std::quoted(c);
          return ss.str();
        },
        [](const Function<SerializableValue>& impl) -> RetT {
          const auto& lambda = impl->lambda;
          const auto& anf_expr = ConvertCoreExprToAnfExpr(lambda);
          ADT_LET_CONST_REF(anf_atomic,
                            anf_expr.template TryGet<Atomic<AnfExpr>>());
          ADT_LET_CONST_REF(anf_lambda,
                            anf_atomic.template TryGet<Lambda<AnfExpr>>());
          AnfExprHelper anf_expr_helper;
          ADT_LET_CONST_REF(anf_expr_str,
                            anf_expr_helper.FunctionToString(anf_lambda));
          return anf_expr_str;
        },
        [&](const adt::List<SerializableValue>& lst) -> RetT {
          return ToStringImpl(lst);
        },
        [&](const axpr::AttrMap<SerializableValue>& obj) -> RetT {
          return ToStringImpl(obj);
        },
        [&](const BuiltinFuncVoidPtr& func) -> RetT {
          const auto& name_info =
              BuiltinFuncNameMgr::Singleton()->OptGet(func.func_ptr);
          ADT_CHECK(name_info.has_value());
          return name_info.value().ToString();
        },
        [&](const BuiltinHighOrderFuncVoidPtr& func) -> RetT {
          const auto& name_info =
              BuiltinFuncNameMgr::Singleton()->OptGet(func.func_ptr);
          ADT_CHECK(name_info.has_value());
          return name_info.value().ToString();
        });
  }

  adt::Result<std::string> ToStringImpl(
      const adt::List<SerializableValue>& lst) {
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    for (const auto& elt : *lst) {
      if (i++ > 0) {
        ss << ", ";
      }
      ADT_LET_CONST_REF(str, ToString(elt));
      ss << str;
    }
    ss << "]";
    return ss.str();
  }

  adt::Result<std::string> ToStringImpl(
      const axpr::AttrMap<SerializableValue>& object) {
    std::ostringstream ss;
    ss << "{";
    int i = 0;
    for (const auto& [k, v] : object->storage) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << std::quoted(k);
      ss << ":";
      ADT_LET_CONST_REF(str, ToString(v));
      ss << str;
    }
    ss << "}";
    return ss.str();
  }

  template <typename ValueT>
  adt::Result<SerializableValue> CastListFrom(const adt::List<ValueT>& lst) {
    adt::List<SerializableValue> ret;
    ret->reserve(lst->size());
    for (const auto& elt : *lst) {
      ADT_LET_CONST_REF(converted, CastFrom(elt));
      ret->emplace_back(converted);
    }
    return ret;
  }

  template <typename ValueT>
  adt::Result<SerializableValue> CastObjectFrom(const AttrMap<ValueT>& obj) {
    AttrMap<SerializableValue> ret_object{};
    for (const auto& [k, v] : obj->storage) {
      ADT_LET_CONST_REF(converted, CastFrom(v));
      ret_object->Set(k, converted);
    }
    return AttrMap<SerializableValue>{ret_object};
  }
};

}  // namespace ap::axpr

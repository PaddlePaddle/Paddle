// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/serializable_value_helper.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/axpr/value_method_class.h"

namespace ap::axpr {

struct SerializableValueHelperImpl {
  adt::Result<SerializableValue> CastFrom(const axpr::Value& val) {
    using RetT = adt::Result<SerializableValue>;
    using TypeT = typename TypeTrait<axpr::Value>::TypeT;
    return val.Match(
        [&](const TypeT& type) -> RetT {
          return type.Match(
              [](const TypeImpl<adt::Nothing>& impl) -> RetT {
                return TypeImplNothing::CastFrom<axpr::Value,
                                                 SerializableValue>(impl);
              },
              [](const TypeImpl<bool>& impl) -> RetT {
                return TypeImplbool::CastFrom<axpr::Value, SerializableValue>(
                    impl);
              },
              [](const TypeImpl<int64_t>& impl) -> RetT {
                return TypeImplint64_t::CastFrom<axpr::Value,
                                                 SerializableValue>(impl);
              },
              [](const TypeImpl<double>& impl) -> RetT {
                return TypeImpldouble::CastFrom<axpr::Value, SerializableValue>(
                    impl);
              },
              [](const TypeImpl<std::string>& impl) -> RetT {
                return TypeImplstring::CastFrom<axpr::Value, SerializableValue>(
                    impl);
              },
              [](const TypeImpl<DataType>& impl) -> RetT {
                return TypeImplDataType::CastFrom<axpr::Value,
                                                  SerializableValue>(impl);
              },
              [](const TypeImpl<DataValue>& impl) -> RetT {
                return TypeImplDataValue::CastFrom<axpr::Value,
                                                   SerializableValue>(impl);
              },
              [](const TypeImpl<PointerType>& impl) -> RetT {
                return TypeImplPointerType::CastFrom<axpr::Value,
                                                     SerializableValue>(impl);
              },
              [](const TypeImpl<PointerValue>& impl) -> RetT {
                return TypeImplPointerValue::CastFrom<axpr::Value,
                                                      SerializableValue>(impl);
              },
              [](const TypeImpl<adt::List<axpr::Value>>& impl) -> RetT {
                return TypeImplList::CastFrom<axpr::Value, SerializableValue>(
                    impl);
              },
              [](const TypeImpl<adt::List<SerializableValue>>& impl) -> RetT {
                return TypeImplListSerializable::CastFrom<axpr::Value,
                                                          SerializableValue>(
                    impl);
              },
              [](const TypeImpl<MutableList<axpr::Value>>& impl) -> RetT {
                return TypeImplMutableList::CastFrom<axpr::Value,
                                                     SerializableValue>(impl);
              },
              [](const TypeImpl<AttrMap<axpr::Value>>& impl) -> RetT {
                return TypeImplAttrMap::CastFrom<axpr::Value,
                                                 SerializableValue>(impl);
              },
              [](const TypeImpl<AttrMap<SerializableValue>>& impl) -> RetT {
                return TypeImplAttrMapSerializable::CastFrom<axpr::Value,
                                                             SerializableValue>(
                    impl);
              },
              [](const TypeImpl<OrderedDict<axpr::Value>>& impl) -> RetT {
                return TypeImplOrderedDict::CastFrom<axpr::Value,
                                                     SerializableValue>(impl);
              },
              [](const TypeImpl<MutableOrderedDict<axpr::Value>>& impl)
                  -> RetT {
                return TypeImplMutableOrderedDict::CastFrom<axpr::Value,
                                                            SerializableValue>(
                    impl);
              },
              [](const TypeImpl<BuiltinClassInstance<axpr::Value>>& impl)
                  -> RetT {
                return TypeImplBuiltinClassInstance{impl.class_ops()};
              },
              [](const TypeImpl<ClassInstance<axpr::Value>>& impl) -> RetT {
                return TypeImplClassInstance<SerializableValue>{
                    impl.class_attrs};
              },
              [](const TypeImpl<PackedArgs<axpr::Value>>& impl) -> RetT {
                return TypeImplPackedArgs::CastFrom<axpr::Value,
                                                    SerializableValue>(impl);
              },
              [](const TypeImpl<Starred<axpr::Value>>& impl) -> RetT {
                return TypeImplStarred::CastFrom<axpr::Value,
                                                 SerializableValue>(impl);
              },
              [](const TypeImpl<Function<SerializableValue>>& impl) -> RetT {
                return TypeImplFunction::CastFrom<axpr::Value,
                                                  SerializableValue>(impl);
              },
              [](const TypeImpl<Closure<axpr::Value>>& impl) -> RetT {
                return TypeImplClosure::CastFrom<axpr::Value,
                                                 SerializableValue>(impl);
              },
              [](const TypeImpl<Continuation<axpr::Value>>& impl) -> RetT {
                return TypeImplContinuation::CastFrom<axpr::Value,
                                                      SerializableValue>(impl);
              },
              [](const TypeImpl<Method<axpr::Value>>& impl) -> RetT {
                return TypeImplMethod::CastFrom<axpr::Value, SerializableValue>(
                    impl);
              },
              [](const TypeImpl<builtin_symbol::Symbol>& impl) -> RetT {
                return TypeImplSymbol::CastFrom<axpr::Value, SerializableValue>(
                    impl);
              },
              [](const TypeImpl<BuiltinFuncType<axpr::Value>>& impl) -> RetT {
                return TypeImplBuiltinFuncType::CastFrom<axpr::Value,
                                                         SerializableValue>(
                    impl);
              },
              [](const TypeImpl<BuiltinHighOrderFuncType<axpr::Value>>& impl)
                  -> RetT {
                return TypeImplBuiltinHighOrderFuncType::
                    CastFrom<axpr::Value, SerializableValue>(impl);
              },
              [](const auto& impl) -> RetT {
                return adt::errors::NotImplementedError{
                    std::string() +
                    "cannot cast to SerializableValue from type " +
                    impl.Name()};
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
        [&](const adt::List<axpr::Value>& list) -> RetT {
          return CastListFrom(list);
        },
        [&](const AttrMap<axpr::Value>& object) -> RetT {
          return CastObjectFrom(object);
        },
        [&](const BuiltinFuncType<axpr::Value>& func) -> RetT {
          auto* func_ptr = reinterpret_cast<void*>(func);
          return BuiltinFuncVoidPtr{func_ptr};
        },
        [&](const BuiltinHighOrderFuncType<axpr::Value>& func) -> RetT {
          auto* func_ptr = reinterpret_cast<void*>(func);
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
        },
        [](const TypeImplBuiltinClassInstance& impl) -> RetT {
          return reinterpret_cast<int64_t>(impl.class_ops);
        },
        [&](const TypeImplClassInstance<SerializableValue>& impl) -> RetT {
          int64_t hash_value = std::hash<std::string>()(impl.Name());
          return hash_value;
        },
        [](const auto& impl) -> RetT {
          const auto& type_impl =
              impl.template CastToAxprType<axpr::Value, SerializableValue>();
          int64_t hash_value = std::hash<std::string>()(type_impl.Name());
          return hash_value;
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
        },
        [](const TypeImplBuiltinClassInstance& impl) -> RetT {
          auto* class_ops =
              reinterpret_cast<const ClassOps<axpr::Value>*>(impl.class_ops);
          return class_ops->class_attrs()->Name();
        },
        [&](const TypeImplClassInstance<SerializableValue>& impl) -> RetT {
          return impl.Name();
        },
        [](const auto& impl) -> RetT {
          const auto& type_impl =
              impl.template CastToAxprType<axpr::Value, SerializableValue>();
          return type_impl.Name();
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

  adt::Result<SerializableValue> CastListFrom(
      const adt::List<axpr::Value>& lst) {
    adt::List<SerializableValue> ret;
    ret->reserve(lst->size());
    for (const auto& elt : *lst) {
      ADT_LET_CONST_REF(converted, CastFrom(elt));
      ret->emplace_back(converted);
    }
    return ret;
  }

  adt::Result<SerializableValue> CastObjectFrom(
      const AttrMap<axpr::Value>& obj) {
    AttrMap<SerializableValue> ret_object{};
    for (const auto& [k, v] : obj->storage) {
      ADT_LET_CONST_REF(converted, CastFrom(v));
      ret_object->Set(k, converted);
    }
    return AttrMap<SerializableValue>{ret_object};
  }
};

adt::Result<SerializableValue> SerializableValueHelper::CastFrom(
    const axpr::Value& val) {
  return SerializableValueHelperImpl{}.CastFrom(val);
}

adt::Result<int64_t> SerializableValueHelper::Hash(
    const SerializableValue& val) {
  return SerializableValueHelperImpl{}.Hash(val);
}

adt::Result<std::string> SerializableValueHelper::ToString(
    const SerializableValue& val) {
  return SerializableValueHelperImpl{}.ToString(val);
}

adt::Result<SerializableValue> SerializableValueHelper::CastObjectFrom(
    const AttrMap<axpr::Value>& val) {
  return SerializableValueHelperImpl{}.CastObjectFrom(val);
}

}  // namespace ap::axpr

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
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/bool.h"
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/builtin_func_name_mgr.h"
#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/builtin_high_order_func_type.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map.h"
#include "paddle/ap/include/axpr/builtin_symbol.h"
#include "paddle/ap/include/axpr/class_attrs.h"
#include "paddle/ap/include/axpr/closure.h"
#include "paddle/ap/include/axpr/continuation.h"
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/data_value.h"
#include "paddle/ap/include/axpr/environment.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/float.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/int.h"
#include "paddle/ap/include/axpr/list.h"
#include "paddle/ap/include/axpr/method.h"
#include "paddle/ap/include/axpr/mutable_list.h"
#include "paddle/ap/include/axpr/mutable_ordered_dict.h"
#include "paddle/ap/include/axpr/nothing.h"
#include "paddle/ap/include/axpr/ordered_dict.h"
#include "paddle/ap/include/axpr/packed_args.h"
#include "paddle/ap/include/axpr/pointer_type.h"
#include "paddle/ap/include/axpr/pointer_value.h"
#include "paddle/ap/include/axpr/starred.h"
#include "paddle/ap/include/axpr/string.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

struct BuiltinFuncVoidPtr {
  void* func_ptr;

  bool operator==(const BuiltinFuncVoidPtr& other) const {
    return this->func_ptr == other.func_ptr;
  }
};

struct BuiltinHighOrderFuncVoidPtr {
  void* func_ptr;

  bool operator==(const BuiltinHighOrderFuncVoidPtr& other) const {
    return this->func_ptr == other.func_ptr;
  }
};

#define DEFINE_TYPEIMPL(name, full_name)                    \
  struct TypeImpl##name : public std::monostate {           \
    using std::monostate::monostate;                        \
    using Self = TypeImpl##name;                            \
                                                            \
    template <typename ValueT, typename SerializableValueT> \
    static Self CastFrom(const TypeImpl<full_name>&) {      \
      return Self{};                                        \
    }                                                       \
                                                            \
    template <typename ValueT, typename SerializableValueT> \
    TypeImpl<full_name> CastToAxprType() const {            \
      return TypeImpl<full_name>{};                         \
    }                                                       \
  };

DEFINE_TYPEIMPL(Nothing, adt::Nothing);
DEFINE_TYPEIMPL(bool, bool);
DEFINE_TYPEIMPL(int64_t, int64_t);
DEFINE_TYPEIMPL(double, double);
DEFINE_TYPEIMPL(string, std::string);
DEFINE_TYPEIMPL(DataType, DataType);
DEFINE_TYPEIMPL(DataValue, DataValue);
DEFINE_TYPEIMPL(PointerType, PointerType);
DEFINE_TYPEIMPL(PointerValue, PointerValue);
DEFINE_TYPEIMPL(List, adt::List<ValueT>);
DEFINE_TYPEIMPL(ListSerializable, adt::List<SerializableValueT>);
DEFINE_TYPEIMPL(MutableList, MutableList<ValueT>);
DEFINE_TYPEIMPL(AttrMap, AttrMap<ValueT>);
DEFINE_TYPEIMPL(AttrMapSerializable, AttrMap<SerializableValueT>);
DEFINE_TYPEIMPL(OrderedDict, OrderedDict<ValueT>);
DEFINE_TYPEIMPL(MutableOrderedDict, MutableOrderedDict<ValueT>);

struct TypeImplBuiltinClassInstance {
  using Self = TypeImplBuiltinClassInstance;

  const void* class_ops;
  bool operator==(const Self& other) const {
    return this->class_ops == other.class_ops;
  }
};

template <typename SerializableValueT>
struct TypeImplClassInstance {
  using Self = TypeImplClassInstance<SerializableValueT>;

  const std::string& Name() const { return this->class_attrs->Name(); }

  ClassAttrs<SerializableValueT> class_attrs;
  bool operator==(const Self& other) const {
    return this->class_attrs == other.class_attrs;
  }
};

DEFINE_TYPEIMPL(PackedArgs, PackedArgs<ValueT>);
DEFINE_TYPEIMPL(Starred, Starred<ValueT>);
DEFINE_TYPEIMPL(Function, Function<SerializableValueT>);
DEFINE_TYPEIMPL(Closure, Closure<ValueT>);
DEFINE_TYPEIMPL(Continuation, Continuation<ValueT>);
DEFINE_TYPEIMPL(Method, Method<ValueT>);
DEFINE_TYPEIMPL(Symbol, builtin_symbol::Symbol);
DEFINE_TYPEIMPL(BuiltinFuncType, BuiltinFuncType<ValueT>);
DEFINE_TYPEIMPL(BuiltinHighOrderFuncType, BuiltinHighOrderFuncType<ValueT>);

template <typename SerializableValueT>
using SerializableValueImpl =
    std::variant<TypeImplNothing,
                 TypeImplbool,
                 TypeImplint64_t,
                 TypeImpldouble,
                 TypeImplstring,
                 TypeImplDataType,
                 TypeImplDataValue,
                 TypeImplPointerType,
                 TypeImplPointerValue,
                 TypeImplList,
                 TypeImplListSerializable,
                 TypeImplMutableList,
                 TypeImplAttrMap,
                 TypeImplAttrMapSerializable,
                 TypeImplOrderedDict,
                 TypeImplMutableOrderedDict,
                 TypeImplBuiltinClassInstance,
                 TypeImplClassInstance<SerializableValue>,
                 TypeImplPackedArgs,
                 TypeImplStarred,
                 TypeImplFunction,
                 TypeImplClosure,
                 TypeImplContinuation,
                 TypeImplMethod,
                 TypeImplSymbol,
                 TypeImplBuiltinFuncType,
                 TypeImplBuiltinHighOrderFuncType,
                 adt::Nothing,
                 bool,
                 int64_t,
                 double,
                 std::string,
                 Function<SerializableValueT>,
                 adt::List<SerializableValueT>,
                 AttrMap<SerializableValueT>,
                 BuiltinFuncVoidPtr,
                 BuiltinHighOrderFuncVoidPtr>;

template <typename ValueT>
struct ClassInstance;

struct SerializableValue : public SerializableValueImpl<SerializableValue> {
  using SerializableValueImpl<SerializableValue>::SerializableValueImpl;

  ADT_DEFINE_VARIANT_METHODS(SerializableValueImpl<SerializableValue>);

  template <typename ValueT>
  ValueT CastTo() const {
    return Match(
        [](const adt::Nothing& impl) -> ValueT { return impl; },
        [](const bool& impl) -> ValueT { return impl; },
        [](const int64_t& impl) -> ValueT { return impl; },
        [](const double& impl) -> ValueT { return impl; },
        [](const std::string& impl) -> ValueT { return impl; },
        [](const Function<SerializableValue>& impl) -> ValueT { return impl; },
        [](const adt::List<SerializableValue>& impl) -> ValueT { return impl; },
        [](const AttrMap<SerializableValue>& impl) -> ValueT { return impl; },
        [](const BuiltinFuncVoidPtr& func) -> ValueT {
          return reinterpret_cast<BuiltinFuncType<ValueT>>(func.func_ptr);
        },
        [](const BuiltinHighOrderFuncVoidPtr& func) -> ValueT {
          return reinterpret_cast<BuiltinHighOrderFuncType<ValueT>>(
              func.func_ptr);
        },
        [&](const TypeImplBuiltinClassInstance& impl) -> ValueT {
          auto* ptr =
              reinterpret_cast<const axpr::ClassOps<ValueT>*>(impl.class_ops);
          return TypeImpl<axpr::BuiltinClassInstance<ValueT>>{ptr};
        },
        [&](const TypeImplClassInstance<SerializableValue>& impl) -> ValueT {
          return TypeImpl<ClassInstance<ValueT>>(impl.class_attrs);
        },
        [&](const auto& impl) -> ValueT {
          return impl.template CastToAxprType<ValueT, SerializableValue>();
        });
  }

  template <typename ValueT>
  static bool IsSerializable(const ValueT& val) {
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    return val.Match(
        [&](const TypeT& type) -> bool { return true; },
        [](const Nothing&) -> bool { return true; },
        [](bool) -> bool { return true; },
        [](int64_t) -> bool { return true; },
        [](double) -> bool { return true; },
        [](const std::string&) -> bool { return true; },
        [](const Function<SerializableValue>&) -> bool { return true; },
        [](const adt::List<SerializableValue>&) -> bool { return true; },
        [](const AttrMap<SerializableValue>&) -> bool { return true; },
        [&](const adt::List<ValueT>& list) -> bool {
          for (const auto& elt : *list) {
            if (!IsSerializable(elt)) {
              return false;
            }
          }
          return true;
        },
        [&](const AttrMap<ValueT>& object) -> bool {
          for (const auto& [k, v] : object->object->storage) {
            if (!IsSerializable(v)) {
              return false;
            }
          }
          return true;
        },
        [&](const BuiltinFuncType<ValueT>& func) -> bool {
          void* func_ptr = reinterpret_cast<void*>(func);
          return BuiltinFuncNameMgr::Singleton()->Has(func_ptr);
        },
        [&](const BuiltinHighOrderFuncType<ValueT>& func) -> bool {
          void* func_ptr = reinterpret_cast<void*>(func);
          return BuiltinFuncNameMgr::Singleton()->Has(func_ptr);
        },
        [&](const auto&) -> bool { return false; });
  }

  static std::string SerializableTypeNames() {
    return "NoneType, bool, int, float, str, class, function, "
           "BuiltinSerializableList, BuiltinSerializableAttrMap";
  }
};

}  // namespace ap::axpr

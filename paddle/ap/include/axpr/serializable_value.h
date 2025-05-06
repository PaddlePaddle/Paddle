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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/bool.h"
#include "paddle/ap/include/axpr/builtin_func_name_mgr.h"
#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/class_attrs.h"
#include "paddle/ap/include/axpr/float.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/int.h"
#include "paddle/ap/include/axpr/nothing.h"
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

template <typename SerializableValueT>
using SerializableValueImpl = std::variant<TypeImpl<adt::Nothing>,
                                           TypeImpl<bool>,
                                           TypeImpl<int64_t>,
                                           TypeImpl<double>,
                                           TypeImpl<std::string>,
                                           ClassAttrs<SerializableValueT>,
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
        [&](const BuiltinFuncVoidPtr& func) -> ValueT {
          return reinterpret_cast<BuiltinFuncType<ValueT>>(func.func_ptr);
        },
        [&](const BuiltinHighOrderFuncVoidPtr& func) -> ValueT {
          return reinterpret_cast<BuiltinHighOrderFuncType<ValueT>>(
              func.func_ptr);
        },
        [&](const ClassAttrs<SerializableValue>& class_attrs) -> ValueT {
          return TypeImpl<ClassInstance<ValueT>>(class_attrs);
        },
        [&](const auto& impl) -> ValueT { return impl; });
  }

  template <typename ValueT>
  static bool IsSerializable(const ValueT& val) {
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    return val.Match(
        [&](const TypeT& type) -> bool {
          return type.Match(
              [](const TypeImpl<adt::Nothing>&) -> bool { return true; },
              [](const TypeImpl<bool>&) -> bool { return true; },
              [](const TypeImpl<int64_t>&) -> bool { return true; },
              [](const TypeImpl<double>&) -> bool { return true; },
              [](const TypeImpl<std::string>&) -> bool { return true; },
              [](const TypeImpl<ClassInstance<ValueT>>&) -> bool {
                return true;
              },
              [&](const auto&) -> bool { return false; });
        },
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

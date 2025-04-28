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

#include <any>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/class_attrs.h"
#include "paddle/ap/include/axpr/class_ops.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinClassInstance;

template <typename ValueT>
struct TypeImpl<BuiltinClassInstance<ValueT>> {
  TypeImpl<BuiltinClassInstance<ValueT>>(
      const ClassOps<ValueT>* class_ops)  // NOLINT
      : class_ops_(class_ops) {}

  const ClassOps<ValueT>* class_ops_;

  const ClassOps<ValueT>* class_ops() const { return class_ops_; }

  const ClassAttrsImpl<ValueT>* class_attrs() const {
    return class_ops_->class_attrs();
  }

  ValueT New(const std::any& any) const;

  const std::string& Name() const { return class_attrs()->Name(); }

  bool operator==(const TypeImpl<BuiltinClassInstance<ValueT>>& other) const {
    return this->class_ops_ == other.class_ops_;
  }
};

template <typename ValueT>
struct BuiltinClassInstance {
  TypeImpl<BuiltinClassInstance<ValueT>> type;
  std::any instance;

  template <typename T>
  bool Has() const {
    return this->instance.type() == typeid(T);
  }

  template <typename T>
  adt::Result<T> TryGet() const {
    if (this->template Has<T>()) {
      return std::any_cast<T>(this->instance);
    } else {
      return adt::errors::TypeError{
          std::string() + "casting from " + type.Name() +
          " class (cpp class name: " + instance.type().name() + ") to " +
          typeid(T).name() + " failed."};
    }
  }

  bool operator==(const BuiltinClassInstance& other) const {
    const auto* class_ops = this->type.class_ops();
    const auto& ret = class_ops->Equals(*this, other);
    CHECK(ret.HasOkValue())
        << "\nTraceback (most recent call last):\n"
        << ret.GetError().CallStackToString() << "\n"
        << ret.GetError().class_name()
        << ": BuiltinClassInstance::operator()(): " << ret.GetError().msg();
    return ret.GetOkValue();
  }
};

template <typename ValueT>
ValueT TypeImpl<BuiltinClassInstance<ValueT>>::New(const std::any& any) const {
  return BuiltinClassInstance<ValueT>{*this, any};
}

template <typename ValueT>
using BuiltinFrameValImpl = std::variant<BuiltinFuncType<ValueT>,
                                         BuiltinHighOrderFuncType<ValueT>,
                                         typename TypeTrait<ValueT>::TypeT>;

template <typename ValueT, typename VisitorT>
ClassAttrs<ValueT> MakeBuiltinClass(const std::string& class_name,
                                    const VisitorT& Visitor) {
  AttrMap<ValueT> attr_map;
  Visitor([&](const auto& name, const auto& val) { attr_map->Set(name, val); });
  adt::List<std::shared_ptr<ClassAttrsImpl<ValueT>>> empty_superclasses{};
  return ClassAttrs<ValueT>{class_name, empty_superclasses, attr_map};
}

}  // namespace ap::axpr

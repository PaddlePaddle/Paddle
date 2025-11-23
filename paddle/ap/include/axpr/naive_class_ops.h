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
#include "paddle/ap/include/axpr/class_ops.h"
#include "paddle/ap/include/axpr/value.h"

namespace ap::axpr {

template <typename T>
class NaiveClassOps : public ClassOps<axpr::Value> {
 public:
  explicit NaiveClassOps(const ClassAttrs<axpr::Value>& class_attrs)
      : class_attrs_(class_attrs) {}

  using This = NaiveClassOps;

  const ClassAttrsImpl<axpr::Value>* class_attrs() const override {
    return class_attrs_.shared_ptr().get();
  }

  adt::Result<bool> Equals(const axpr::Value& lhs_val,
                           const axpr::Value& rhs_val) const override {
    return EqualsImpl(lhs_val, rhs_val);
  }

 private:
  static adt::Result<bool> EqualsImpl(const axpr::Value& lhs_val,
                                      const axpr::Value& rhs_val) {
    ADT_LET_CONST_REF(lhs, lhs_val.template CastTo<T>());
    if (!rhs_val.template CastableTo<T>()) {
      return false;
    }
    ADT_LET_CONST_REF(rhs, rhs_val.template CastTo<T>());
    return lhs == rhs;
  }

  const ClassAttrs<axpr::Value> class_attrs_;
};

template <typename T>
class ClassOps<axpr::Value>* MakeGlobalNaiveClassOps(
    const ClassAttrs<axpr::Value>& class_attrs) {
  static NaiveClassOps<T> naive_class_ops(class_attrs);
  return &naive_class_ops;
}

}  // namespace ap::axpr

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

#include "paddle/ap/include/axpr/closure.h"
#include "paddle/ap/include/axpr/const_global_environment.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/mutable_global_environment.h"
#include "paddle/ap/include/axpr/serializable_value.h"

namespace ap::axpr {

template <typename ValueT>
struct ClosureMethodClass {
  using This = ClosureMethodClass;
  using Self = Closure<ValueT>;

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "__function__") {
      return ToFunction(self);
    }
    return adt::errors::AttributeError{std::string() +
                                       "closure object has not attribute '" +
                                       attr_name + "'."};
  }

  adt::Result<ValueT> ToFunction(const Self& self) {
    const auto& global_frame =
        self->environment->RecursivelyGetConstGlobalFrame();
    return Function<SerializableValue>{self->lambda, global_frame};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, Closure<ValueT>>
    : public ClosureMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<Closure<ValueT>>> {};

}  // namespace ap::axpr

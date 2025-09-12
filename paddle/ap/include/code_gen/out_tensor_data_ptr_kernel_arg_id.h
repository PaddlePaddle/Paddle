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
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::code_gen {

template <typename BirNode>
struct OutTensorDataPtrKernelArgIdImpl {
  BirNode ir_value;
  std::optional<axpr::Function<axpr::SerializableValue>> runtime_getter{};

  bool operator==(const OutTensorDataPtrKernelArgIdImpl& other) const {
    return this->ir_value == other.ir_value;
  }

  template <typename ValueT>
  adt::Result<ValueT> CastData() const {
    using RetT = adt::Result<ValueT>;
    return this->ir_value.Match(
        [&](const typename BirNode::native_value_type& impl) -> RetT {
          return impl;
        },
        [&](const typename BirNode::ref_value_type& impl) -> RetT {
          return impl;
        },
        [&](const auto& impl) -> RetT {
          using T = std::decay_t<decltype(impl)>;
          return adt::errors::NotImplementedError{
              std::string() +
              "CastData() failed, only NativeIrValue and RefIrValue supported, "
              "but '" +
              typeid(T).name() + "' found."};
        });
  }

  std::size_t GetHashValue() const {
    return std::hash<BirNode>()(this->ir_value);
  }
};

template <typename BirNode>
ADT_DEFINE_RC(OutTensorDataPtrKernelArgId,
              OutTensorDataPtrKernelArgIdImpl<BirNode>);

template <typename ValueT, typename BirNode>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>
GetOutTensorDataPtrKernelArgIdClass();

}  // namespace ap::code_gen

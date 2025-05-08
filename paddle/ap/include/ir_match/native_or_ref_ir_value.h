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

namespace ap::ir_match {

template <typename BirNode>
using NativeOrRefIrValueImpl = std::variant<typename BirNode::native_value_type,
                                            typename BirNode::ref_value_type>;

template <typename BirNode>
struct NativeOrRefIrValue : public NativeOrRefIrValueImpl<BirNode> {
  using NativeOrRefIrValueImpl<BirNode>::NativeOrRefIrValueImpl;
  ADT_DEFINE_VARIANT_METHODS(NativeOrRefIrValueImpl<BirNode>);

  template <typename ValueT>
  static adt::Result<NativeOrRefIrValue> CastFrom(const ValueT& val) {
    using RetT = adt::Result<NativeOrRefIrValue>;
    return val.Match(
        [](const typename BirNode::native_value_type& impl) -> RetT {
          return impl;
        },
        [](const typename BirNode::ref_value_type& impl) -> RetT {
          return impl;
        },
        [](const auto& impl) -> RetT {
          using T = std::decay_t<decltype(impl)>;
          const char* type_name = typeid(T).name();
          return adt::errors::ValueError{
              std::string() +
              "NativeOrRefIrValue::CastFrom failed. type(val): " + type_name};
        });
  }
};

}  // namespace ap::ir_match

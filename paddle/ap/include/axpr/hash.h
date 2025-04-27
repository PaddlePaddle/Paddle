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
#include "paddle/ap/include/axpr/method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct Hash {
  adt::Result<int64_t> operator()(InterpreterBase<ValueT>* interpreter,
                                  const ValueT& val) const {
    const auto& func = MethodClass<ValueT>::Hash(val);
    using RetT = adt::Result<int64_t>;
    return func.Match(
        [&](const adt::Nothing&) -> RetT {
          return adt::errors::TypeError{GetTypeName(val) +
                                        " class has no __hash__ function."};
        },
        [&](adt::Result<ValueT> (*unary_func)(const ValueT&)) -> RetT {
          ADT_LET_CONST_REF(hash_val, unary_func(val));
          ADT_LET_CONST_REF(hash, hash_val.template TryGet<int64_t>());
          return hash;
        },
        [&](adt::Result<ValueT> (*unary_func)(InterpreterBase<ValueT>*,
                                              const ValueT&)) -> RetT {
          ADT_LET_CONST_REF(hash_val, unary_func(interpreter, val));
          ADT_LET_CONST_REF(hash, hash_val.template TryGet<int64_t>());
          return hash;
        });
  }
};

}  // namespace ap::axpr

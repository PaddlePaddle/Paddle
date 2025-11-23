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
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/frame.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename SerializableValueT>
struct FunctionImpl {
  Lambda<CoreExpr> lambda;
  std::optional<Frame<SerializableValueT>> global_frame{};

  bool operator==(const FunctionImpl& other) const { return this == &other; }

  adt::Result<int64_t> GetHashValue() const {
    int64_t hash_value = reinterpret_cast<int64_t>(lambda.shared_ptr().get());
    if (global_frame.has_value()) {
      ADT_LET_CONST_REF(global_frame_ptr, global_frame.value().Get());
      int64_t frame_hash_value = reinterpret_cast<int64_t>(global_frame_ptr);
      hash_value = adt::hash_combine(hash_value, frame_hash_value);
    }
    return hash_value;
  }
};

template <typename SerializableValueT>
ADT_DEFINE_RC(Function, FunctionImpl<SerializableValueT>);

template <typename SerializableValueT>
struct TypeImpl<Function<SerializableValueT>> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "function"; }
};

}  // namespace ap::axpr

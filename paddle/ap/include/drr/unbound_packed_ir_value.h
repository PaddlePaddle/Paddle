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
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/type.h"

namespace ap::drr {

struct TensorPatternCtxImpl;

template <typename NodeT>
struct UnboundPackedIrValueImpl {
  std::string name;
  std::weak_ptr<TensorPatternCtxImpl> tensor_pattern_ctx;
  bool operator==(const UnboundPackedIrValueImpl& other) const {
    return this->name == other.name &&
           this->tensor_pattern_ctx.lock() == other.tensor_pattern_ctx.lock();
  }
};

template <typename NodeT>
ADT_DEFINE_RC(UnboundPackedIrValue, UnboundPackedIrValueImpl<NodeT>);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetUnboundPackedIrValueClass();

template <typename NodeT>
struct Type<drr::UnboundPackedIrValue<NodeT>> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "UnboundPackedIrValue"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetUnboundPackedIrValueClass();
  }
};

}  // namespace ap::drr

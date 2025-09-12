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
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/type.h"

namespace ap::drr {

struct OpPatternCtxImpl;

template <typename NodeT>
struct NativeIrOpDeclareImpl {
  std::string op_name;
  std::weak_ptr<OpPatternCtxImpl> op_pattern_ctx;
  axpr::AttrMap<axpr::Value> attr_map{};

  bool operator==(const NativeIrOpDeclareImpl& other) const {
    return this->op_name == other.op_name &&
           this->op_pattern_ctx.lock() == other.op_pattern_ctx.lock() &&
           this->attr_map == other.attr_map;
  }
};

template <typename NodeT>
ADT_DEFINE_RC(NativeIrOpDeclare, NativeIrOpDeclareImpl<NodeT>);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnNativeIrOpDeclareClass();

template <typename NodeT>
struct Type<drr::tSrcPtn<drr::NativeIrOpDeclare<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "SrcPtnNativeIrOpDeclare"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetSrcPtnNativeIrOpDeclareClass();
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnNativeIrOpDeclareClass();

template <typename NodeT>
struct Type<drr::tResPtn<drr::NativeIrOpDeclare<NodeT>>>
    : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "ResPtnNativeIrOpDeclare"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetResPtnNativeIrOpDeclareClass();
  }
};

}  // namespace ap::drr

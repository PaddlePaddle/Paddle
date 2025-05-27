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
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/ir_value.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/type.h"
#include "paddle/ap/include/graph/node_arena.h"
#include "paddle/ap/include/graph/tags.h"

namespace ap::drr {

struct DrrCtxImpl;

struct TensorPatternCtxImpl {
  std::shared_ptr<graph::NodeArena<drr::Node>> node_arena;
  mutable std::map<std::string, IrValue> uid2ir_value{};
  std::weak_ptr<DrrCtxImpl> drr_ctx;
  mutable std::map<std::string, axpr::Value> uid2type{};

  bool operator==(const TensorPatternCtxImpl& other) const {
    return this == &other;
  }
};

ADT_DEFINE_RC(TensorPatternCtx, TensorPatternCtxImpl);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnTensorPatternCtxClass();

template <>
struct Type<drr::tSrcPtn<drr::TensorPatternCtx>> : public std::monostate {
  using value_type = drr::tSrcPtn<drr::TensorPatternCtx>;

  const char* Name() const { return "SrcPtnTensorPatternCtx"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetSrcPtnTensorPatternCtxClass();
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnTensorPatternCtxClass();

template <>
struct Type<drr::tResPtn<drr::TensorPatternCtx>> : public std::monostate {
  using value_type = drr::tResPtn<drr::TensorPatternCtx>;

  const char* Name() const { return "ResPtnTensorPatternCtx"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetResPtnTensorPatternCtxClass();
  }
};

}  // namespace ap::drr

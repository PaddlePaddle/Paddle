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

#include <map>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/ir_op.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/type.h"
#include "paddle/ap/include/graph/node_arena.h"
#include "paddle/ap/include/graph/tags.h"

namespace ap::drr {

struct DrrCtxImpl;

struct OpPatternCtxImpl {
  std::shared_ptr<graph::NodeArena<drr::Node>> node_arena;
  mutable std::map<std::string, IrOp> uid2ir_op;
  std::weak_ptr<DrrCtxImpl> drr_ctx;

  bool operator==(const OpPatternCtxImpl& other) const {
    return this == &other;
  }
};

ADT_DEFINE_RC(OpPatternCtx, OpPatternCtxImpl);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnOpPatternCtxClass();

template <>
struct Type<drr::tSrcPtn<drr::OpPatternCtx>> : public std::monostate {
  using value_type = drr::tSrcPtn<drr::OpPatternCtx>;

  const char* Name() const { return "SrcPtnOpPatternCtx"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetSrcPtnOpPatternCtxClass();
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnOpPatternCtxClass();

template <>
struct Type<drr::tResPtn<drr::OpPatternCtx>> : public std::monostate {
  using value_type = drr::tResPtn<drr::OpPatternCtx>;

  const char* Name() const { return "ResPtnOpPatternCtx"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetResPtnOpPatternCtxClass();
  }
};

}  // namespace ap::drr

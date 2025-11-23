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
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/op_pattern_ctx.h"
#include "paddle/ap/include/drr/source_pattern_ctx.h"
#include "paddle/ap/include/drr/tensor_pattern_ctx.h"
#include "paddle/ap/include/drr/type.h"
#include "paddle/ap/include/graph/node_arena.h"
#include "paddle/ap/include/graph/tags.h"

namespace ap::drr {

struct ResultPatternCtxImpl {
  std::shared_ptr<graph::NodeArena<drr::Node>> node_arena;
  OpPatternCtx op_pattern_ctx;
  TensorPatternCtx tensor_pattern_ctx;
  SourcePatternCtx source_pattern_ctx;
  std::unordered_set<std::string> internal_native_ir_value_names{};

  bool operator==(const ResultPatternCtxImpl& other) const {
    return this != &other;
  }
};

ADT_DEFINE_RC(ResultPatternCtx, ResultPatternCtxImpl);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResultPatternCtxClass();

template <>
struct Type<drr::ResultPatternCtx> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "ResultPatternCtx"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetResultPatternCtxClass();
  }
};

}  // namespace ap::drr

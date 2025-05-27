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
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/source_pattern_ctx.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/ir_match/graph_match_ctx.h"
#include "paddle/ap/include/ir_match/op_match_ctx.h"
#include "paddle/ap/include/ir_match/tensor_match_ctx.h"

namespace ap::ir_match {

template <typename BirNode>
struct IrMatchCtxImpl {
  using DrrNodeT = drr::Node;
  using SmallGraphNodeT = graph::Node<DrrNodeT>;
  drr::SourcePatternCtx source_pattern_ctx;
  GraphMatchCtx<BirNode> graph_match_ctx;
};

template <typename BirNode>
ADT_DEFINE_RC(IrMatchCtx, IrMatchCtxImpl<BirNode>);

}  // namespace ap::ir_match

namespace ap::axpr {

template <typename BirNode>
struct TypeImpl<ir_match::IrMatchCtx<BirNode>> : public std::monostate {
  using std::monostate::monostate;
  const char* Name() const { return "IrMatchCtx"; }
};

}  // namespace ap::axpr

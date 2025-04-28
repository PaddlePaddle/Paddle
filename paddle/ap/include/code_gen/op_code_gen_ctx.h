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
#include "paddle/ap/include/code_gen/kernel_arg_id.h"
#include "paddle/ap/include/code_gen/loop_anchor_flags.h"
#include "paddle/ap/include/ir_match/native_or_ref_ir_value.h"

namespace ap::code_gen {

template <typename BirNode>
struct CodeGenCtxImpl;

template <typename BirNode>
struct OpCodeGenCtxImpl {
  std::weak_ptr<CodeGenCtxImpl<BirNode>> code_gen_ctx;

  LoopAnchorFlags input_index_loop_anchor_flags;
  LoopAnchorFlags output_index_loop_anchor_flags;

  bool operator==(const OpCodeGenCtxImpl& other) const {
    return this == &other;
  }
};

template <typename BirNode>
ADT_DEFINE_RC(OpCodeGenCtx, OpCodeGenCtxImpl<BirNode>);

}  // namespace ap::code_gen

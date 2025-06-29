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
#include "paddle/ap/include/drr/native_ir_op.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/opt_packed_ir_op.h"
#include "paddle/ap/include/drr/packed_ir_op.h"
#include "paddle/ap/include/drr/unbound_native_ir_op.h"
#include "paddle/ap/include/drr/unbound_opt_packed_ir_op.h"
#include "paddle/ap/include/drr/unbound_packed_ir_op.h"

namespace ap::drr {

using IrOpImpl = std::variant<NativeIrOp<drr::Node>,
                              PackedIrOp<drr::Node>,
                              OptPackedIrOp<drr::Node>,
                              UnboundNativeIrOp<drr::Node>,
                              UnboundPackedIrOp<drr::Node>,
                              UnboundOptPackedIrOp<drr::Node>>;

struct IrOp : public IrOpImpl {
  using IrOpImpl::IrOpImpl;
  ADT_DEFINE_VARIANT_METHODS(IrOpImpl);

  const std::string& op_name() const {
    using RetT = const std::string&;
    return Match(
        [&](const NativeIrOp<drr::Node>& impl) -> RetT {
          return impl->op_declare->op_name;
        },
        [&](const PackedIrOp<drr::Node>& impl) -> RetT {
          return impl->op_declare->op_name;
        },
        [&](const OptPackedIrOp<drr::Node>& impl) -> RetT {
          return impl->op_declare->op_name;
        },
        [&](const UnboundNativeIrOp<drr::Node>& impl) -> RetT {
          return impl->op_declare->op_name;
        },
        [&](const UnboundPackedIrOp<drr::Node>& impl) -> RetT {
          return impl->op_declare->op_name;
        },
        [&](const UnboundOptPackedIrOp<drr::Node>& impl) -> RetT {
          return impl->op_declare->op_name;
        });
  }
};

}  // namespace ap::drr

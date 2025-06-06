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
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/drr/native_ir_op.h"
#include "paddle/ap/include/drr/native_ir_op_operand.h"
#include "paddle/ap/include/drr/native_ir_op_result.h"
#include "paddle/ap/include/drr/native_ir_value.h"
#include "paddle/ap/include/drr/opt_packed_ir_op.h"
#include "paddle/ap/include/drr/opt_packed_ir_op_operand.h"
#include "paddle/ap/include/drr/opt_packed_ir_op_result.h"
#include "paddle/ap/include/drr/packed_ir_op.h"
#include "paddle/ap/include/drr/packed_ir_op_operand.h"
#include "paddle/ap/include/drr/packed_ir_op_result.h"
#include "paddle/ap/include/drr/packed_ir_value.h"
#include "paddle/ap/include/graph/node_topo_cstr.h"

namespace ap::drr {

template <typename NodeT>
using NodeImpl = std::variant<NativeIrValue<NodeT>,
                              NativeIrOp<NodeT>,
                              NativeIrOpOperand<NodeT>,
                              NativeIrOpResult<NodeT>,
                              PackedIrValue<NodeT>,
                              PackedIrOp<NodeT>,
                              PackedIrOpOperand<NodeT>,
                              PackedIrOpResult<NodeT>,
                              OptPackedIrOp<NodeT>,
                              OptPackedIrOpOperand<NodeT>,
                              OptPackedIrOpResult<NodeT>>;

struct Node : public NodeImpl<Node> {
  using NodeImpl<Node>::NodeImpl;
  ADT_DEFINE_VARIANT_METHODS(NodeImpl<Node>);

  const graph::Node<Node>& node() const {
    return Match([](const auto& impl) -> const graph::Node<Node>& {
      return impl->node;
    });
  }

  graph::NodeTopoCstr node_topo_cstr() const {
    return Match([](const auto& impl) -> graph::NodeTopoCstr {
      return impl->node_topo_cstr();
    });
  }
};

}  // namespace ap::drr

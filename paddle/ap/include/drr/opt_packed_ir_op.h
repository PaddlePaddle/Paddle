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
#include "paddle/ap/include/drr/opt_packed_ir_op_declare.h"
#include "paddle/ap/include/drr/type.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/graph/node_topo_cstr.h"

namespace ap::drr {

template <typename NodeT>
struct OptPackedIrOpImpl {
  graph::Node<NodeT> node;
  OptPackedIrOpDeclare<NodeT> op_declare;
  std::string name;

  bool operator==(const OptPackedIrOpImpl& other) const {
    return this->node == other.node && this->op_declare == other.op_declare &&
           this->name == other.name;
  }

  graph::OptPackedIrOpTopoCstr node_topo_cstr() const {
    return graph::OptPackedIrOpTopoCstr{this->op_declare->op_name};
  }
};

template <typename NodeT>
ADT_DEFINE_RC(OptPackedIrOp, OptPackedIrOpImpl<NodeT>);

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetOptPackedIrOpClass();

template <typename NodeT>
struct Type<drr::OptPackedIrOp<NodeT>> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "OptPackedIrOp"; }

  static axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetClass() {
    return GetOptPackedIrOpClass();
  }
};

}  // namespace ap::drr

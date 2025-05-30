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

#include "paddle/ap/include/graph/adt.h"

namespace ap::graph {

struct NativeIrValueTopoCstr : public std::monostate {
  using std::monostate::monostate;
};

struct NativeIrOpTopoCstr {
  std::string op_name;

  bool operator==(const NativeIrOpTopoCstr& other) const {
    return this->op_name == other.op_name;
  }
};

struct NativeIrOpOperandTopoCstr {
  std::size_t index;

  bool operator==(const NativeIrOpOperandTopoCstr& other) const {
    return this->index == other.index;
  }
};

struct NativeIrOpResultTopoCstr {
  std::size_t index;

  bool operator==(const NativeIrOpResultTopoCstr& other) const {
    return this->index == other.index;
  }
};

struct PackedIrValueTopoCstr {
  bool operator==(const PackedIrValueTopoCstr&) const { return false; }
  bool operator!=(const PackedIrValueTopoCstr&) const { return false; }
};

struct PackedIrOpTopoCstr {
  std::string op_name;

  bool operator==(const PackedIrOpTopoCstr& other) const {
    return this->op_name == other.op_name;
  }
};

struct PackedIrOpOperandTopoCstr : public std::monostate {
  using std::monostate::monostate;
};

struct PackedIrOpResultTopoCstr {
  std::size_t index;

  bool operator==(const PackedIrOpResultTopoCstr& other) const {
    return this->index == other.index;
  }
};

struct OptPackedIrOpTopoCstr {
  PackedIrOpTopoCstr packed_ir_op_topo_cstr;

  bool operator==(const OptPackedIrOpTopoCstr& other) const {
    return this->packed_ir_op_topo_cstr == other.packed_ir_op_topo_cstr;
  }
};

struct OptPackedIrOpOperandTopoCstr {
  PackedIrOpOperandTopoCstr packed_ir_op_operand_topo_cstr;

  bool operator==(const OptPackedIrOpOperandTopoCstr& other) const {
    return this->packed_ir_op_operand_topo_cstr ==
           other.packed_ir_op_operand_topo_cstr;
  }
};

struct OptPackedIrOpResultTopoCstr : public std::monostate {
  PackedIrOpResultTopoCstr packed_ir_op_result_topo_cstr;

  bool operator==(const OptPackedIrOpResultTopoCstr& other) const {
    return this->packed_ir_op_result_topo_cstr ==
           other.packed_ir_op_result_topo_cstr;
  }
};

struct RefIrValueTopoCstr : public std::monostate {
  using std::monostate::monostate;
};

struct RefIrOpTopoCstr : public std::monostate {
  using std::monostate::monostate;
};

struct RefIrOpOperandTopoCstr : public std::monostate {
  using std::monostate::monostate;
};

struct RefIrOpResultTopoCstr : public std::monostate {
  using std::monostate::monostate;
};

using NodeTopoCstrImpl = std::variant<NativeIrValueTopoCstr,
                                      NativeIrOpTopoCstr,
                                      NativeIrOpOperandTopoCstr,
                                      NativeIrOpResultTopoCstr,
                                      PackedIrValueTopoCstr,
                                      PackedIrOpTopoCstr,
                                      PackedIrOpOperandTopoCstr,
                                      PackedIrOpResultTopoCstr,
                                      OptPackedIrOpTopoCstr,
                                      OptPackedIrOpOperandTopoCstr,
                                      OptPackedIrOpResultTopoCstr,
                                      RefIrValueTopoCstr,
                                      RefIrOpTopoCstr,
                                      RefIrOpOperandTopoCstr,
                                      RefIrOpResultTopoCstr>;
// node constraint
struct NodeTopoCstr : public NodeTopoCstrImpl {
  using NodeTopoCstrImpl::NodeTopoCstrImpl;
  ADT_DEFINE_VARIANT_METHODS(NodeTopoCstrImpl);

  adt::Result<bool> TopoSatisfy(const NodeTopoCstr& sg_node_topo_cstr) const {
    using RetT = adt::Result<bool>;
    const auto& pattern_match = ::common::Overloaded{
        [&](const PackedIrOpTopoCstr& bg_topo_cstr,
            const OptPackedIrOpTopoCstr& sg_topo_cstr) -> RetT {
          return bg_topo_cstr == sg_topo_cstr.packed_ir_op_topo_cstr;
        },
        [&](const PackedIrOpOperandTopoCstr& bg_topo_cstr,
            const OptPackedIrOpOperandTopoCstr& sg_topo_cstr) -> RetT {
          return bg_topo_cstr == sg_topo_cstr.packed_ir_op_operand_topo_cstr;
        },
        [&](const PackedIrOpResultTopoCstr& bg_topo_cstr,
            const OptPackedIrOpResultTopoCstr& sg_topo_cstr) -> RetT {
          return bg_topo_cstr == sg_topo_cstr.packed_ir_op_result_topo_cstr;
        },
        [&](const RefIrValueTopoCstr& bg_topo_cstr,
            const NativeIrValueTopoCstr& sg_topo_cstr) -> RetT { return true; },
        [&](const RefIrOpTopoCstr& bg_topo_cstr,
            const OptPackedIrOpTopoCstr& sg_topo_cstr) -> RetT { return true; },
        [&](const RefIrOpOperandTopoCstr& bg_topo_cstr,
            const OptPackedIrOpOperandTopoCstr& sg_topo_cstr) -> RetT {
          return true;
        },
        [&](const RefIrOpResultTopoCstr& bg_topo_cstr,
            const OptPackedIrOpResultTopoCstr& sg_topo_cstr) -> RetT {
          return true;
        },
        [&](const auto&, const auto&) -> RetT {
          return *this == sg_node_topo_cstr;
        }};
    return std::visit(
        pattern_match, this->variant(), sg_node_topo_cstr.variant());
  }
};

struct SmallGraphNodeTopoCstr {
  NodeTopoCstr node_topo_cstr;
};

struct BigGraphNodeTopoCstr {
  NodeTopoCstr node_topo_cstr;

  adt::Result<bool> TopoSatisfy(
      const SmallGraphNodeTopoCstr& sg_node_topo_cstr) const {
    return this->node_topo_cstr.TopoSatisfy(sg_node_topo_cstr.node_topo_cstr);
  }
};

}  // namespace ap::graph

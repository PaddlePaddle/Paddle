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

#include <sstream>
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/op_tensor_pattern_ctx_helper.h"
#include "paddle/ap/include/drr/src_ptn_packed_ir_op_declare_data.h"
#include "paddle/ap/include/graph/node_descriptor.h"
#include "paddle/ap/include/ir_match/ref_match_ctx.h"
#include "paddle/ap/include/paddle/pir/packed_ir_op_inner_source_pattern_helper.h"
#include "paddle/ap/include/paddle/pir_node.h"

namespace ap::paddle {

struct PirNodeDescriptor {
  using RefNodeInfo = ir_match::RefNodeInfo<NativeIrValue, NativeIrOpOperand>;

  std::string DebugId(const PirNode& node) const {
    return node.Match(
        [&](const NativeIrValue& ir_value) -> std::string {
          if (ir_value.value.defining_op() == nullptr) {
            return std::to_string(
                reinterpret_cast<int64_t>(ir_value.value.impl()));
          } else {
            const auto* op = ir_value.value.defining_op();
            const auto& op_debug_id = GetOpDebugId(op);
            for (int i = 0; i < op->num_results(); ++i) {
              if (op->result(i) == ir_value.value) {
                return op_debug_id + "_out_" + std::to_string(i);
              }
            }
            return op_debug_id + "_error_output";
          }
        },
        [&](const PackedIrValue& ir_value) -> std::string {
          pir::Operation* op = ir_value.fusion_op;
          const auto& op_debug_id = GetOpDebugId(op);
          if (ir_value.is_output) {
            return op_debug_id + "_packed_out";
          } else {
            return op_debug_id + "_packed_in";
          }
        },
        [&](const NativeIrOpOperand& ir_op_operand) -> std::string {
          const auto& operand = ir_op_operand.op_operand;
          const auto& op_debug_id = GetOpDebugId(operand.owner());
          return op_debug_id + "_operand_" + std::to_string(operand.index());
        },
        [&](const PackedIrOpOperand& ir_op_operand) -> std::string {
          pir::Operation* op = ir_op_operand.fusion_op;
          const auto& op_debug_id = GetOpDebugId(op);
          std::size_t index = ir_op_operand.free_tensor_index;
          return op_debug_id + "_packed_operand_" + std::to_string(index);
        },
        [&](const NativeIrOp& ir_op) -> std::string {
          return GetOpDebugId(ir_op.op);
        },
        [&](const PackedIrOp& ir_op) -> std::string {
          pir::Operation* op = ir_op.fusion_op;
          return GetOpDebugId(op);
        },
        [&](const NativeIrOpResult& ir_op_result) -> std::string {
          pir::Operation* op = ir_op_result.op_result.owner();
          const auto& op_debug_id = GetOpDebugId(op);
          std::size_t index = ir_op_result.op_result.index();
          return op_debug_id + "_result_" + std::to_string(index);
        },
        [&](const PackedIrOpResult& ir_op_result) -> std::string {
          pir::Operation* op = ir_op_result.op_result.owner();
          const auto& op_debug_id = GetOpDebugId(op);
          std::size_t index = ir_op_result.op_result.index();
          return op_debug_id + "_packed_result_" + std::to_string(index);
        },
        [&](const RefIrValue& impl) -> std::string {
          return std::string() + "RefIrValue(" +
                 GetRefNodeInfoDebugString(impl.ref_node_info) + ")";
        },
        [&](const RefIrOpOperand& impl) -> std::string {
          return std::string() + "RefIrOpOperand(" +
                 GetRefNodeInfoDebugString(impl.ref_node_info) + ")";
        },
        [&](const RefIrOp& impl) -> std::string {
          return std::string() + "RefIrOp(" +
                 GetRefNodeInfoDebugString(impl.ref_node_info) + ")";
        },
        [&](const RefIrOpResult& impl) -> std::string {
          return std::string() + "RefIrOpResult(" +
                 GetRefNodeInfoDebugString(impl.ref_node_info) + ")";
        });
  }

  std::string GetRefNodeInfoDebugString(
      const RefNodeInfo& ref_node_info) const {
    std::ostringstream ss;
    ss << DebugId(ref_node_info->ir_value);
    ss << "=>[";
    int i = 0;
    for (const auto& op_operand : *ref_node_info->op_operands_subset) {
      if (i++ > 0) {
        ss << ",";
      }
      ss << "(" << DebugId(op_operand) << ")";
    }
    ss << "]";
    return ss.str();
  }

  std::string GetOpDebugId(const pir::Operation* op) const {
    return op->name() + "_" + std::to_string(op->id());
  }

  adt::Result<bool> AttrsSatisfyIfBothAreOpsOrValues(
      const PirNode& node, const graph::Node<drr::Node>& drr_graph_node) {
    ADT_LET_CONST_REF(drr_node, drr_graph_node.Get());
    using RetT = adt::Result<bool>;
    auto pattern_match = ::common::Overloaded{
        [&](const NativeIrValue& pir_value,
            const drr::NativeIrValue<drr::Node>& drr_value) -> RetT {
          return ValueAttrsSatisfy(pir_value, drr_value);
        },
        [&](const NativeIrOp& pir_op, const drr::NativeIrOp<drr::Node>& drr_op)
            -> RetT { return NativeOpAttrsSatisfy(pir_op, drr_op); },
        [&](const PackedIrOp& pir_op, const drr::PackedIrOp<drr::Node>& drr_op)
            -> RetT { return PackedOpAttrsSatisfy(pir_op, drr_op); },
        [&](const auto& lhs, const auto& rhs) -> RetT { return true; }};
    return std::visit(pattern_match, node.variant(), drr_node.variant());
  }

  adt::Result<bool> ValueAttrsSatisfy(
      const NativeIrValue& pir_value,
      const drr::NativeIrValue<drr::Node>& drr_value) {
    ADT_LET_CONST_REF(opt_type,
                      drr::OpTensorPatternCtxHelper{}.GetOptType(drr_value));
    if (!opt_type.has_value()) {
      return true;
    }
    ADT_LET_CONST_REF(type, opt_type.value().template CastTo<pir::Type>());
    return type == pir_value.value.type();
  }

  adt::Result<bool> NativeOpAttrsSatisfy(
      const NativeIrOp& pir_op, const drr::NativeIrOp<drr::Node>& drr_op) {
    if (drr_op->op_declare->attr_map->storage.empty()) {
      return true;
    }
    for (const auto& [attr_name, attr_val] :
         drr_op->op_declare->attr_map->storage) {
      const auto& iter = pir_op.op->attributes().find(attr_name);
      if (iter == pir_op.op->attributes().end()) {
        continue;
      }
      const auto& pir_attr_val = iter->second;
      ADT_LET_CONST_REF(drr_attr_val,
                        attr_val.template CastTo<pir::Attribute>());
      if (pir_attr_val != drr_attr_val) {
        return false;
      }
    }
    return true;
  }

  adt::Result<std::size_t> GetDrrNumInputs(
      const drr::PackedIrOp<drr::Node>& drr_op) {
    return ap::drr::DefaultDrrGraphDescriptor{}.GetNumInputs(drr_op);
  }

  adt::Result<std::size_t> GetDrrNumOutputs(
      const drr::PackedIrOp<drr::Node>& drr_op) {
    return ap::drr::DefaultDrrGraphDescriptor{}.GetNumOutputs(drr_op);
  }

  adt::Result<std::size_t> GetPirNumInputs(const PackedIrOp& pir_op) {
    std::size_t size =
        ap::paddle::GetUsedExternalValue(*pir_op.fusion_op).size();
    return size;
  }

  adt::Result<std::size_t> GetPirNumOutputs(const PackedIrOp& pir_op) {
    for (auto& op : *pir_op.fusion_op.block()) {
      if (op.isa<pir::YieldOp>()) {
        std::size_t size = op.num_operands();
        return size;
      }
    }
    return adt::errors::TypeError{"no cf.yield found in FusionOp"};
  }

  adt::Result<bool> PackedOpAttrsSatisfy(
      const PackedIrOp& pir_op, const drr::PackedIrOp<drr::Node>& drr_op) {
    ADT_LET_CONST_REF(drr_num_inputs, GetDrrNumInputs(drr_op));
    ADT_LET_CONST_REF(pir_num_inputs, GetPirNumInputs(pir_op));
    if (drr_num_inputs != pir_num_inputs) {
      return false;
    }
    ADT_LET_CONST_REF(drr_num_outputs, GetDrrNumOutputs(drr_op));
    ADT_LET_CONST_REF(pir_num_outputs, GetPirNumOutputs(pir_op));
    if (drr_num_outputs != pir_num_outputs) {
      return false;
    }
    ADT_LET_CONST_REF(inner_source_pattern_satisfy,
                      PackedOpInnerSourcePatternSatisfy(pir_op, drr_op));
    if (!inner_source_pattern_satisfy) {
      return false;
    }
    return true;
  }

  adt::Result<bool> PackedOpInnerSourcePatternSatisfy(
      const PackedIrOp& pir_op, const drr::PackedIrOp<drr::Node>& drr_op) {
    ADT_CHECK(drr_op->op_declare->data.has_value());
    auto* raw_data_ptr = drr_op->op_declare->data.value().get();
    auto* data_ptr =
        dynamic_cast<drr::SrcPtnPackedIrOpDeclareData*>(raw_data_ptr);
    ADT_CHECK(data_ptr != nullptr);
    if (!data_ptr->inner_source_pattern_func.has_value()) {
      ADT_CHECK(!data_ptr->inner_source_pattern_ctx.has_value());
      return true;
    }
    ADT_CHECK(data_ptr->inner_source_pattern_ctx.has_value());
    PackedIrOpInnerSourcePatternHelper helper{};
    ADT_LET_CONST_REF(
        opt_match_ctx,
        helper.Match(pir_op, data_ptr->inner_source_pattern_ctx.value()));
    return opt_match_ctx.has_value();
  }
};

}  // namespace ap::paddle

namespace ap::graph {

template <>
struct NodeDescriptor<ap::paddle::PirNode>
    : public ap::paddle::PirNodeDescriptor {};

}  // namespace ap::graph

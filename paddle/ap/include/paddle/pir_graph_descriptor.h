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
#include "paddle/ap/include/drr/topo_kind.h"
#include "paddle/ap/include/graph/graph_descriptor.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/ir_match/ref_match_ctx.h"
#include "paddle/ap/include/paddle/pir_node.h"
#include "paddle/ap/include/paddle/pir_util.h"

namespace ap::paddle {

struct DefaultPirGraphDescriptor {
  using NodeT = PirNode;

  NodeT CastToIrOpResult(const pir::OpResult& op_result) const {
    if (op_result.owner()->isa<cinn::dialect::FusionOp>()) {
      return PackedIrOpResult{op_result};
    } else {
      return NativeIrOpResult{op_result};
    }
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    return node.Match(
        [&](const NativeIrValue& impl) -> adt::Result<adt::Ok> {
          if (pir::OpResult::classof(impl.value)) {
            return DoEach(
                CastToIrOpResult(pir::OpResult::dyn_cast_from(impl.value)));
          }
          return adt::Ok{};
        },
        [&](const PackedIrValue& impl) -> adt::Result<adt::Ok> {
          // TODO(tianchao): support the following case:
          // o.trivial_op0([*t.inputs], [t.op0_output, *t.op0_output1])
          // o.trivial_op1([*.t.op0_output1], [t.op1_output])
          return adt::errors::NotImplementedError{
              "DefaultPirGraphDescriptor::VisitUpstreamNodes does not support "
              "PackedIrValue"};
        },
        [&](const RefIrValue& impl) -> adt::Result<adt::Ok> {
          RefIrOpResult ir_op_result{impl.ref_node_info};
          return DoEach(ir_op_result);
        },
        [&](const NativeIrOpOperand& impl) -> adt::Result<adt::Ok> {
          NativeIrValue ir_value{impl.op_operand.source()};
          return DoEach(ir_value);
        },
        [&](const PackedIrOpOperand& impl) -> adt::Result<adt::Ok> {
          const auto& inputs = GetFusionOpInputValues(impl.fusion_op);
          ADT_CHECK(impl.free_tensor_index >= 0);
          ADT_CHECK(impl.free_tensor_index < inputs.size());
          NativeIrValue ir_value{inputs.at(impl.free_tensor_index)};
          return DoEach(ir_value);
        },
        [&](const RefIrOpOperand& impl) -> adt::Result<adt::Ok> {
          return DoEach(impl.ref_node_info->ir_value);
        },
        [&](const NativeIrOp& impl) -> adt::Result<adt::Ok> {
          for (int i = 0; i < impl.op->num_operands(); ++i) {
            NativeIrOpOperand ir_op_operand{impl.op->operand(i)};
            ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
          }
          return adt::Ok{};
        },
        [&](const PackedIrOp& impl) -> adt::Result<adt::Ok> {
          const auto& inputs = GetFusionOpInputValues(impl.fusion_op);
          for (int i = 0; i < inputs.size(); ++i) {
            PackedIrOpOperand ir_op_operand{impl.fusion_op, i};
            ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
          }
          return adt::Ok{};
        },
        [&](const RefIrOp& impl) -> adt::Result<adt::Ok> {
          RefIrOpOperand ir_op_operand{impl.ref_node_info};
          return DoEach(ir_op_operand);
        },
        [&](const NativeIrOpResult& impl) -> adt::Result<adt::Ok> {
          NativeIrOp ir_op{impl.op_result.defining_op()};
          return DoEach(ir_op);
        },
        [&](const PackedIrOpResult& impl) -> adt::Result<adt::Ok> {
          auto* op = impl.op_result.defining_op();
          ADT_CHECK(op->isa<cinn::dialect::FusionOp>());
          PackedIrOp ir_op{op->dyn_cast<cinn::dialect::FusionOp>()};
          return DoEach(ir_op);
        },
        [&](const RefIrOpResult& impl) -> adt::Result<adt::Ok> {
          RefIrOp ir_op{impl.ref_node_info};
          return DoEach(ir_op);
        });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    return node.Match(
        [&](const NativeIrValue& impl) -> adt::Result<adt::Ok> {
          for (auto iter = impl.value.use_begin(); iter != impl.value.use_end();
               ++iter) {
            auto* user_parent_block = iter->owner()->GetParent();
            ADT_CHECK(user_parent_block != nullptr);
            auto* user_parent_op = user_parent_block->GetParentOp();
            if (user_parent_op->isa<cinn::dialect::FusionOp>()) {
              auto fusion_op =
                  user_parent_op->dyn_cast<cinn::dialect::FusionOp>();
              const auto& user_op_inputs = GetFusionOpInputValues(fusion_op);
              for (int i = 0; i < user_op_inputs.size(); ++i) {
                if (user_op_inputs.at(i) == impl.value) {
                  PackedIrOpOperand ir_op_operand{fusion_op, i};
                  ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
                }
              }
            } else {
              pir::OpOperand op_operand = *iter;
              NativeIrOpOperand ir_op_operand{op_operand};
              ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
            }
          }
          return adt::Ok{};
        },
        [&](const PackedIrValue& impl) -> adt::Result<adt::Ok> {
          // TODO(tianchao): support the following case:
          // o.trivial_op0([*t.inputs], [t.op0_output, *t.op0_output1])
          // o.trivial_op1([*.t.op0_output1], [t.op1_output])
          return adt::Ok{};
        },
        [&](const RefIrValue& impl) -> adt::Result<adt::Ok> {
          for (const auto& ir_op_operand :
               *impl.ref_node_info->op_operands_subset) {
            ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
          }
          return adt::Ok{};
        },
        [&](const NativeIrOpOperand& impl) -> adt::Result<adt::Ok> {
          NativeIrOp ir_op{impl.op_operand.owner()};
          return DoEach(ir_op);
        },
        [&](const PackedIrOpOperand& impl) -> adt::Result<adt::Ok> {
          PackedIrOp ir_op{impl.fusion_op};
          return DoEach(ir_op);
        },
        [&](const RefIrOpOperand& impl) -> adt::Result<adt::Ok> {
          RefIrOp ir_op{impl.ref_node_info};
          return DoEach(ir_op);
        },
        [&](const NativeIrOp& impl) -> adt::Result<adt::Ok> {
          for (int i = 0; i < impl.op->num_results(); ++i) {
            const auto& value = impl.op->result(i);
            ADT_CHECK(pir::OpResult::classof(value));
            NativeIrOpResult ir_op_result{pir::OpResult::dyn_cast_from(value)};
            ADT_RETURN_IF_ERR(DoEach(ir_op_result));
          }
          return adt::Ok{};
        },
        [&](const PackedIrOp& impl) -> adt::Result<adt::Ok> {
          for (int i = 0; i < impl.fusion_op->num_results(); ++i) {
            const auto& value = impl.fusion_op->result(i);
            ADT_CHECK(pir::OpResult::classof(value));
            PackedIrOpResult ir_op_result{pir::OpResult::dyn_cast_from(value)};
            ADT_RETURN_IF_ERR(DoEach(ir_op_result));
          }
          return adt::Ok{};
        },
        [&](const RefIrOp& impl) -> adt::Result<adt::Ok> {
          RefIrOpResult ir_op_result{impl.ref_node_info};
          return DoEach(ir_op_result);
        },
        [&](const NativeIrOpResult& impl) -> adt::Result<adt::Ok> {
          pir::Value value = impl.op_result;
          NativeIrValue ir_value{value};
          return DoEach(ir_value);
        },
        [&](const PackedIrOpResult& impl) -> adt::Result<adt::Ok> {
          pir::Value value = impl.op_result;
          NativeIrValue ir_value{value};
          return DoEach(ir_value);
        },
        [&](const RefIrOpResult& impl) -> adt::Result<adt::Ok> {
          RefIrValue ir_value{impl.ref_node_info};
          return DoEach(ir_value);
        });
  }

  adt::Result<graph::SmallGraphNodeTopoCstr> GetSmallGraphNodeTopoCstr(
      const NodeT& node) const {
    return graph::SmallGraphNodeTopoCstr{node.node_topo_cstr()};
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    return node.Match(
        [](const PackedIrValue&) -> adt::Result<bool> { return true; },
        [](const auto&) -> adt::Result<bool> { return false; });
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return node.Match([&](const NativeIrOp&) -> bool { return true; },
                      [&](const PackedIrOp&) -> bool { return true; },
                      [&](const RefIrOp&) -> bool { return true; },
                      [&](const auto&) -> bool { return false; });
  }

  adt::Result<bool> IsValueNode(const NodeT& node) const {
    return node.Match([&](const NativeIrValue&) -> bool { return true; },
                      [&](const PackedIrValue&) -> bool { return true; },
                      [&](const RefIrValue&) -> bool { return true; },
                      [&](const auto&) -> bool { return false; });
  }

  adt::Result<bool> TopoSatisfy(
      const NodeT& node,
      const graph::SmallGraphNodeTopoCstr& node_topo_cstr) const {
    graph::BigGraphNodeTopoCstr bg_node_topo_cstr{node.node_topo_cstr()};
    return bg_node_topo_cstr.TopoSatisfy(node_topo_cstr);
  }

  const std::vector<pir::Value>& GetFusionOpInputValues(
      cinn::dialect::FusionOp fusion_op) const {
    auto iter = fusion_op2input_values_.find(fusion_op);
    if (iter == fusion_op2input_values_.end()) {
      iter =
          fusion_op2input_values_
              .emplace(fusion_op, ap::paddle::GetUsedExternalValue(*fusion_op))
              .first;
    }
    return iter->second;
  }

 private:
  mutable std::unordered_map<pir::Operation*, std::vector<pir::Value>>
      fusion_op2input_values_;
};

struct RefAugmentedPirGraphDescriptor {
  using NodeT = PirNode;
  using RefNodeInfo = ir_match::RefNodeInfo<NativeIrValue, NativeIrOpOperand>;
  using RefMatchCtx = ir_match::RefMatchCtx<NativeIrValue, NativeIrOpOperand>;
  RefMatchCtx ref_match_ctx;
  DefaultPirGraphDescriptor backend_graph;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    using Ok = adt::Result<adt::Ok>;
    return node.Match(
        [&](const NativeIrOpOperand& impl) -> Ok {
          const auto iter = ref_match_ctx->operand2node_info.find(impl);
          if (iter == ref_match_ctx->operand2node_info.end()) {
            return backend_graph.VisitUpstreamNodes(node, DoEach);
          }
          RefIrValue ref_ir_value{iter->second};
          return DoEach(ref_ir_value);
        },
        [&](const auto&) -> Ok {
          return backend_graph.VisitUpstreamNodes(node, DoEach);
        });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    using Ok = adt::Result<adt::Ok>;
    return node.Match(
        [&](const NativeIrValue& impl) -> Ok {
          const auto iter = ref_match_ctx->value2ref_node_info.find(impl);
          if (iter == ref_match_ctx->value2ref_node_info.end()) {
            return backend_graph.VisitDownstreamNodes(node, DoEach);
          }
          for (const auto& ref_node_info : iter->second) {
            RefIrOpOperand ir_op_operand{ref_node_info};
            ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
          }
          return adt::Ok{};
        },
        [&](const auto&) -> Ok {
          return backend_graph.VisitDownstreamNodes(node, DoEach);
        });
  }

  adt::Result<graph::SmallGraphNodeTopoCstr> GetSmallGraphNodeTopoCstr(
      const NodeT& node) const {
    return backend_graph.GetSmallGraphNodeTopoCstr(node);
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    return backend_graph.IgnoredNode(node);
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return backend_graph.IsOpNode(node);
  }

  adt::Result<bool> TopoSatisfy(
      const NodeT& node,
      const graph::SmallGraphNodeTopoCstr& node_topo_cstr) const {
    return backend_graph.TopoSatisfy(node, node_topo_cstr);
  }
};

struct AllOperandAndResultPirGraphDescriptor {
  using NodeT = PirNode;

  DefaultPirGraphDescriptor backend_graph;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    auto DoEachOpOrValue = [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_op_node, backend_graph.IsOpNode(upstream));
      ADT_LET_CONST_REF(is_value_node, backend_graph.IsValueNode(upstream));
      ADT_CHECK(is_op_node || is_value_node);
      return backend_graph.VisitUpstreamNodes(upstream, DoEach);
    };
    return backend_graph.VisitUpstreamNodes(node, DoEachOpOrValue);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    auto DoEachOpOrValue =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_op_node, backend_graph.IsOpNode(downstream));
      ADT_LET_CONST_REF(is_value_node, backend_graph.IsValueNode(downstream));
      ADT_CHECK(is_op_node || is_value_node);
      return backend_graph.VisitDownstreamNodes(downstream, DoEach);
    };
    return backend_graph.VisitDownstreamNodes(node, DoEachOpOrValue);
  }

  adt::Result<graph::SmallGraphNodeTopoCstr> GetSmallGraphNodeTopoCstr(
      const NodeT& node) const {
    return backend_graph.GetSmallGraphNodeTopoCstr(node);
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    ADT_LET_CONST_REF(is_op_node, backend_graph.IsOpNode(node));
    ADT_LET_CONST_REF(is_value_node, backend_graph.IsValueNode(node));
    if (is_op_node || is_value_node) {
      return true;
    }
    return backend_graph.IgnoredNode(node);
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return backend_graph.IsOpNode(node);
  }

  adt::Result<bool> TopoSatisfy(
      const NodeT& node,
      const graph::SmallGraphNodeTopoCstr& node_topo_cstr) const {
    return backend_graph.TopoSatisfy(node, node_topo_cstr);
  }
};

struct NativeOperandAndResultPirGraphDescriptor {
  using NodeT = PirNode;

  AllOperandAndResultPirGraphDescriptor backend_graph;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(is_node_native, IsNative(node));
    ADT_CHECK(is_node_native);
    auto VisitEachNative = [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_upstream_native, IsNative(upstream));
      ADT_CHECK(!is_upstream_native);
      return backend_graph.VisitUpstreamNodes(upstream, DoEach);
    };
    auto VisitEachPacked = [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_upstream_native, IsNative(upstream));
      ADT_CHECK(!is_upstream_native);
      return backend_graph.VisitUpstreamNodes(upstream, VisitEachNative);
    };
    auto DoEachOperandOrResult =
        [&](const NodeT& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_native, IsNative(upstream));
      if (is_native) {
        return DoEach(upstream);
      } else {
        return backend_graph.VisitUpstreamNodes(upstream, VisitEachPacked);
      }
    };
    return backend_graph.VisitUpstreamNodes(node, DoEachOperandOrResult);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(is_node_native, IsNative(node));
    ADT_CHECK(is_node_native);
    auto VisitEachNative =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_downstream_native, IsNative(downstream));
      ADT_CHECK(!is_downstream_native);
      return backend_graph.VisitDownstreamNodes(downstream, DoEach);
    };
    auto VisitEachPacked =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_downstream_native, IsNative(downstream));
      ADT_CHECK(!is_downstream_native);
      return backend_graph.VisitDownstreamNodes(downstream, VisitEachNative);
    };
    auto DoEachOperandOrResult =
        [&](const NodeT& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(is_native, IsNative(downstream));
      if (is_native) {
        return DoEach(downstream);
      } else {
        return backend_graph.VisitDownstreamNodes(downstream, VisitEachPacked);
      }
    };
    return backend_graph.VisitDownstreamNodes(node, DoEachOperandOrResult);
  }

  adt::Result<graph::SmallGraphNodeTopoCstr> GetSmallGraphNodeTopoCstr(
      const NodeT& node) const {
    return backend_graph.GetSmallGraphNodeTopoCstr(node);
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    ADT_LET_CONST_REF(is_native, IsNative(node));
    if (!is_native) {
      return true;
    }
    return backend_graph.IgnoredNode(node);
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return backend_graph.IsOpNode(node);
  }

  adt::Result<bool> TopoSatisfy(
      const NodeT& node,
      const graph::SmallGraphNodeTopoCstr& node_topo_cstr) const {
    return backend_graph.TopoSatisfy(node, node_topo_cstr);
  }

  adt::Result<bool> IsNative(const NodeT& node) const {
    return node.Match([&](const NativeIrOpOperand&) -> bool { return true; },
                      [&](const NativeIrOpResult&) -> bool { return true; },
                      [&](const auto&) -> bool { return false; });
  }
};

struct BlockBoundPirGraphDescriptor {
  using NodeT = PirNode;

 private:
  std::function<adt::Result<bool>(const NodeT&)> BelongToThisBlockOrNotOp_;
  DefaultPirGraphDescriptor backend_graph_;

 public:
  explicit BlockBoundPirGraphDescriptor(
      const std::function<adt::Result<bool>(const NodeT&)>&
          BelongToThisBlockOrNotOp)
      : BelongToThisBlockOrNotOp_(BelongToThisBlockOrNotOp), backend_graph_{} {}

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    using Ok = adt::Result<adt::Ok>;
    return backend_graph_.VisitUpstreamNodes(
        node, [&](const NodeT& upstream) -> Ok {
          ADT_LET_CONST_REF(belong_to_this_block,
                            BelongToThisBlockOrNotOp_(upstream));
          if (belong_to_this_block) {
            return DoEach(upstream);
          }
          return adt::Ok{};
        });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    using Ok = adt::Result<adt::Ok>;
    return node.Match(
        [&](const NativeIrValue& impl) -> adt::Result<adt::Ok> {
          for (auto iter = impl.value.use_begin(); iter != impl.value.use_end();
               ++iter) {
            ADT_LET_CONST_REF(
                belong_to_this_block,
                BelongToThisBlockOrNotOp_(NativeIrOp{iter->owner()}));
            if (belong_to_this_block) {
              pir::OpOperand op_operand = *iter;
              NativeIrOpOperand ir_op_operand{op_operand};
              ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
              continue;
            }
            auto* user_parent_block = iter->owner()->GetParent();
            ADT_CHECK(user_parent_block != nullptr);
            auto* user_parent_op = user_parent_block->GetParentOp();
            if (!user_parent_op->isa<cinn::dialect::FusionOp>()) {
              continue;
            }
            auto fusion_op =
                user_parent_op->dyn_cast<cinn::dialect::FusionOp>();
            ADT_LET_CONST_REF(parent_belong_to_this_block,
                              BelongToThisBlockOrNotOp_(PackedIrOp{fusion_op}));
            if (!parent_belong_to_this_block) {
              continue;
            }
            const auto& user_op_inputs =
                backend_graph_.GetFusionOpInputValues(fusion_op);
            for (int i = 0; i < user_op_inputs.size(); ++i) {
              if (user_op_inputs.at(i) == impl.value) {
                PackedIrOpOperand ir_op_operand{fusion_op, i};
                ADT_RETURN_IF_ERR(DoEach(ir_op_operand));
              }
            }
          }
          return adt::Ok{};
        },
        [&](const auto&) -> Ok {
          return backend_graph_.VisitDownstreamNodes(
              node, [&](const NodeT& downstream) -> Ok {
                ADT_LET_CONST_REF(belong_to_this_block,
                                  BelongToThisBlockOrNotOp_(downstream));
                if (belong_to_this_block) {
                  return DoEach(downstream);
                }
                return adt::Ok{};
              });
        });
  }

  adt::Result<graph::SmallGraphNodeTopoCstr> GetSmallGraphNodeTopoCstr(
      const NodeT& node) const {
    return backend_graph_.GetSmallGraphNodeTopoCstr(node);
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    return backend_graph_.IgnoredNode(node);
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    return backend_graph_.IsOpNode(node);
  }

  adt::Result<bool> TopoSatisfy(
      const NodeT& node,
      const graph::SmallGraphNodeTopoCstr& node_topo_cstr) const {
    return backend_graph_.TopoSatisfy(node, node_topo_cstr);
  }
};

}  // namespace ap::paddle

namespace ap::graph {

template <>
struct GraphDescriptor<ap::paddle::PirNode, drr::topo_kind::Default>
    : public ap::paddle::DefaultPirGraphDescriptor {};

template <>
struct GraphDescriptor<ap::paddle::PirNode, drr::topo_kind::RefAugmented>
    : public ap::paddle::RefAugmentedPirGraphDescriptor {};

template <>
struct GraphDescriptor<ap::paddle::PirNode, drr::topo_kind::AllOperandAndResult>
    : public ap::paddle::AllOperandAndResultPirGraphDescriptor {};

template <>
struct GraphDescriptor<ap::paddle::PirNode,
                       drr::topo_kind::NativeOperandAndResult>
    : public ap::paddle::NativeOperandAndResultPirGraphDescriptor {};

template <>
struct GraphDescriptor<ap::paddle::PirNode, drr::topo_kind::BlockBound>
    : public ap::paddle::BlockBoundPirGraphDescriptor {
  using ap::paddle::BlockBoundPirGraphDescriptor::BlockBoundPirGraphDescriptor;
};

}  // namespace ap::graph

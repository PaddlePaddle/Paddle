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

#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/topo_kind.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/graph/graph_descriptor.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/ir_match/topo_match_ctx.h"

namespace ap::ir_match {

template <typename bg_node_t /*big graph node type*/>
struct GraphMatchCtxImpl {
  using DrrNode = drr::Node;
  using DrrNativeIrValue = drr::NativeIrValue<DrrNode>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNode>;
  using sg_node_t = graph::Node<DrrNode>;

  TopoMatchCtx<bg_node_t, sg_node_t> topo_match_ctx;

  bool operator==(const GraphMatchCtxImpl& other) const {
    return this == &other;
  }

  std::size_t num_matched_bg_nodes() const {
    return topo_match_ctx->num_matched_bg_nodes();
  }

  adt::Result<bool> HasBigGraphNode(const sg_node_t& node) const {
    return topo_match_ctx->HasBigGraphNode(node);
  }

  adt::Result<std::size_t> GetNumBigGraphIrValueNodes(
      const sg_node_t& node) const {
    std::size_t num = 0;
    auto Increase = [&](const auto&) -> adt::Result<adt::Ok> {
      ++num;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitBigGraphIrValueNode(node, Increase));
    return num;
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitBigGraphIrValueNode(const sg_node_t& node,
                                                const YieldT& Yield) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    using Ok = adt::Result<adt::Ok>;
    return drr_node.Match(
        [&](const DrrNativeIrValue&) -> Ok {
          ADT_LET_CONST_REF(bir_node, GetSoleBigGraphNode(node));
          return Yield(bir_node);
        },
        [&](const DrrPackedIrValue&) -> Ok {
          return VisitPackedBigGraphIrValueNode(node, Yield);
        },
        [&](const auto& impl) -> Ok {
          using T = std::decay_t<decltype(impl)>;
          return adt::errors::NotImplementedError{
              std::string() +
              "VisitBigGraphIrValueNode() support DrrNativeIrValue and "
              "DrrPackedIrValue only, " +
              typeid(T).name() + " found."};
        });
  }

  adt::Result<bg_node_t> GetSoleBigGraphNode(const sg_node_t& node) const {
    return topo_match_ctx->GetSoleBigGraphNode(node);
  }

  std::optional<sg_node_t> GetOptMatchedSmallGraphNode(
      const bg_node_t& bg_node) const {
    return topo_match_ctx->GetMatchedSmallGraphNode(bg_node);
  }

  using DefaultDrrGraph =
      graph::GraphDescriptor<sg_node_t, drr::topo_kind::Default>;

  adt::Result<adt::List<bg_node_t>> GetPackedBigGraphIrValueNodes(
      const sg_node_t& node) const {
    adt::List<bg_node_t> ret;
    using Ok = adt::Result<adt::Ok>;
    auto CollectInput = [&](const bg_node_t& node) -> Ok {
      ret->emplace_back(node);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitPackedBigGraphIrValueNode(node, CollectInput));
    return ret;
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitPackedBigGraphIrValueNode(
      const sg_node_t& node, const YieldT& Yield) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    ADT_CHECK(drr_node.template Has<DrrPackedIrValue>());
    DefaultDrrGraph drr_graph{};
    ADT_LET_CONST_REF(is_ignored, drr_graph.IgnoredNode(node));
    ADT_CHECK(is_ignored);
    ADT_LET_CONST_REF(num_inputs, drr_graph.GetNumInputs(drr_node));
    ADT_LET_CONST_REF(num_outputs, drr_graph.GetNumOutputs(drr_node));
    if (num_inputs == 0 && num_outputs == 1) {
      return VisitPackedInputBigGraphNode(node, Yield);
    }
    if (num_inputs == 1 && num_outputs == 0) {
      return VisitPackedOutputBigGraphNode(node, Yield);
    }
    return adt::errors::TypeError{
        std::string() +
        "VisitPackedBigGraphIrValueNode() failed. num_inputs: " +
        std::to_string(num_inputs) +
        ", num_outputs: " + std::to_string(num_outputs)};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitPackedInputBigGraphNode(
      const sg_node_t& packed_ir_value_node, const YieldT& Yield) const {
    ADT_LET_CONST_REF(packed_ir_value_drr_node, packed_ir_value_node.Get());
    DefaultDrrGraph drr_graph{};
    ADT_LET_CONST_REF(drr_packed_ir_op_operand_node,
                      drr_graph.GetSoleOutput(packed_ir_value_drr_node));
    ADT_LET_CONST_REF(drr_packed_ir_op_node,
                      drr_graph.GetSoleOutput(drr_packed_ir_op_operand_node));
    ADT_LET_CONST_REF(
        exclude_bir_native_ir_values,
        GetBirNativeIrInputsOfPackedIrOp(drr_packed_ir_op_node.node()));
    using Ok = adt::Result<adt::Ok>;
    auto YieldIgnored = [&](const bg_node_t& node) -> Ok {
      if (exclude_bir_native_ir_values.count(node) == 0) {
        return Yield(node);
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitBirIrInputOfPackedIrOp(drr_packed_ir_op_node.node(),
                                                  YieldIgnored));
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitPackedOutputBigGraphNode(
      const sg_node_t& packed_ir_value_node, const YieldT& Yield) const {
    ADT_LET_CONST_REF(packed_ir_value_drr_node, packed_ir_value_node.Get());
    DefaultDrrGraph drr_graph{};
    ADT_LET_CONST_REF(drr_packed_ir_op_result_node,
                      drr_graph.GetSoleInput(packed_ir_value_drr_node));
    ADT_LET_CONST_REF(drr_packed_ir_op_node,
                      drr_graph.GetSoleInput(drr_packed_ir_op_result_node));
    ADT_LET_CONST_REF(
        exclude_bir_native_ir_values,
        GetBirNativeIrOutputsOfPackedIrOp(drr_packed_ir_op_node.node()));
    using Ok = adt::Result<adt::Ok>;
    auto YieldIgnored = [&](const bg_node_t& node) -> Ok {
      if (exclude_bir_native_ir_values.count(node) == 0) {
        return Yield(node);
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitBirIrOutputOfPackedIrOp(drr_packed_ir_op_node.node(),
                                                   YieldIgnored));
    return adt::Ok{};
  }

  using DefaultBirGraph =
      graph::GraphDescriptor<bg_node_t, drr::topo_kind::Default>;

  template <typename YieldT>
  adt::Result<adt::Ok> VisitBirIrInputOfPackedIrOp(
      const sg_node_t& drr_packed_ir_op_node, const YieldT& Yield) const {
    DefaultBirGraph bir_graph{};
    ADT_LET_CONST_REF(bir_packed_or_ref_ir_op_node,
                      GetSoleBigGraphNode(drr_packed_ir_op_node));
    using Ok = adt::Result<adt::Ok>;
    auto VisitIrOpOperand = [&](const bg_node_t& node) -> Ok {
      return bir_graph.VisitUpstreamNodes(node, Yield);
    };
    ADT_RETURN_IF_ERR(bir_graph.VisitUpstreamNodes(bir_packed_or_ref_ir_op_node,
                                                   VisitIrOpOperand));
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitBirIrOutputOfPackedIrOp(
      const sg_node_t& drr_packed_ir_op_node, const YieldT& Yield) const {
    DefaultBirGraph bir_graph{};
    ADT_LET_CONST_REF(bir_packed_or_ref_ir_op_node,
                      GetSoleBigGraphNode(drr_packed_ir_op_node));
    using Ok = adt::Result<adt::Ok>;
    auto VisitIrOpResult = [&](const bg_node_t& node) -> Ok {
      return bir_graph.VisitDownstreamNodes(node, Yield);
    };
    ADT_RETURN_IF_ERR(bir_graph.VisitDownstreamNodes(
        bir_packed_or_ref_ir_op_node, VisitIrOpResult));
    return adt::Ok{};
  }

  adt::Result<std::unordered_set<bg_node_t>> GetBirNativeIrInputsOfPackedIrOp(
      const sg_node_t& packed_ir_op_node) const {
    DefaultDrrGraph drr_graph{};
    std::unordered_set<bg_node_t> set;
    using Ok = adt::Result<adt::Ok>;
    int num_ignored = 0;
    auto VisitIrValue = [&](const sg_node_t& node) -> Ok {
      ADT_LET_CONST_REF(ignored, drr_graph.IgnoredNode(node));
      if (!ignored) {
        ADT_LET_CONST_REF(bir_node, GetSoleBigGraphNode(node));
        set.insert(bir_node);
      } else {
        ++num_ignored;
      }
      return adt::Ok{};
    };
    auto VisitIrOpOperand = [&](const sg_node_t& node) -> Ok {
      return drr_graph.VisitUpstreamNodes(node, VisitIrValue);
    };
    ADT_RETURN_IF_ERR(
        drr_graph.VisitUpstreamNodes(packed_ir_op_node, VisitIrOpOperand));
    ADT_CHECK(num_ignored <= 1) << adt::errors::NotImplementedError{
        std::string() +
        "multiple packed ir value inputs are not supported yet."};
    return set;
  }

  adt::Result<std::unordered_set<bg_node_t>> GetBirNativeIrOutputsOfPackedIrOp(
      const sg_node_t& packed_ir_op_node) const {
    DefaultDrrGraph drr_graph{};
    std::unordered_set<bg_node_t> set;
    using Ok = adt::Result<adt::Ok>;
    int num_ignored = 0;
    auto VisitIrValue = [&](const sg_node_t& node) -> Ok {
      ADT_LET_CONST_REF(ignored, drr_graph.IgnoredNode(node));
      if (!ignored) {
        ADT_LET_CONST_REF(bir_node, GetSoleBigGraphNode(node));
        set.insert(bir_node);
      } else {
        ++num_ignored;
      }
      return adt::Ok{};
    };
    auto VisitIrOpResult = [&](const sg_node_t& node) -> Ok {
      return drr_graph.VisitDownstreamNodes(node, VisitIrValue);
    };
    ADT_RETURN_IF_ERR(
        drr_graph.VisitDownstreamNodes(packed_ir_op_node, VisitIrOpResult));
    ADT_CHECK(num_ignored <= 1) << adt::errors::NotImplementedError{
        std::string() +
        "multiple packed ir value outputs are not supported yet."};
    return set;
  }

  adt::Result<sg_node_t> GetMatchedSmallGraphNode(
      const bg_node_t& bg_node) const {
    const auto& sg_node = topo_match_ctx->GetMatchedSmallGraphNode(bg_node);
    ADT_CHECK(sg_node.has_value());
    return sg_node.value();
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitSmallGraphNode(const YieldT& Yield) const {
    return topo_match_ctx->VisitSmallGraphNode(Yield);
  }
};

template <typename bg_node_t /*big graph node type*/>
ADT_DEFINE_RC(GraphMatchCtx, GraphMatchCtxImpl<bg_node_t>);

}  // namespace ap::ir_match

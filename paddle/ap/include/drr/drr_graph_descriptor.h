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
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/topo_kind.h"
#include "paddle/ap/include/graph/graph_descriptor.h"
#include "paddle/ap/include/graph/node.h"

namespace ap::drr {

struct DefaultDrrGraphDescriptor {
  using DrrNode = drr::Node;
  using DrrGraphNode = graph::Node<DrrNode>;
  using NodeT = DrrGraphNode;

  using DrrNativeIrValue = ap::drr::NativeIrValue<DrrNode>;
  using DrrPackedIrValue = ap::drr::PackedIrValue<DrrNode>;
  using DrrNativeIrOp = ap::drr::NativeIrOp<DrrNode>;
  using DrrPackedIrOp = ap::drr::PackedIrOp<DrrNode>;
  using DrrOptPackedIrOp = ap::drr::OptPackedIrOp<DrrNode>;
  using DrrNativeIrOpOperand = ap::drr::NativeIrOpOperand<DrrNode>;
  using DrrPackedIrOpOperand = ap::drr::PackedIrOpOperand<DrrNode>;
  using DrrOptPackedIrOpOperand = ap::drr::OptPackedIrOpOperand<DrrNode>;
  using DrrNativeIrOpResult = ap::drr::NativeIrOpResult<DrrNode>;
  using DrrPackedIrOpResult = ap::drr::PackedIrOpResult<DrrNode>;
  using DrrOptPackedIrOpResult = ap::drr::OptPackedIrOpResult<DrrNode>;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstreamNodes(const NodeT& node,
                                          const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(upstreams, node.UpstreamNodes());
    return upstreams.VisitNodes(DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstreamNodes(const NodeT& node,
                                            const DoEachT& DoEach) const {
    ADT_LET_CONST_REF(downstreams, node.DownstreamNodes());
    return downstreams.VisitNodes(DoEach);
  }

  template <typename DrrNodeT>
  adt::Result<DrrNodeT> CastSoleUnignoredInput(const DrrNode& node) const {
    std::optional<DrrNodeT> opt_sole_input{};
    auto DoEachUpstream =
        [&](const DrrGraphNode& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ignored, IgnoredNode(upstream));
      if (ignored) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(drr_upstream, upstream.Get());
      ADT_LET_CONST_REF(casted, drr_upstream.template TryGet<DrrNodeT>());
      ADT_CHECK(!opt_sole_input.has_value());
      opt_sole_input = casted;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitUpstreamNodes(node.node(), DoEachUpstream));
    ADT_CHECK(opt_sole_input.has_value());
    return opt_sole_input.value();
  }

  adt::Result<DrrNode> GetSoleInput(const DrrNode& node) const {
    std::optional<DrrNode> opt_sole_input{};
    auto DoEachUpstream =
        [&](const DrrGraphNode& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(drr_upstream, upstream.Get());
      ADT_CHECK(!opt_sole_input.has_value());
      opt_sole_input = drr_upstream;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitUpstreamNodes(node.node(), DoEachUpstream));
    ADT_CHECK(opt_sole_input.has_value());
    return opt_sole_input.value();
  }

  template <typename DrrNodeT>
  adt::Result<DrrNodeT> CastSoleUnignoredOutput(const DrrNode& node) const {
    std::optional<DrrNodeT> opt_sole_output{};
    auto DoEachDownstream =
        [&](const DrrGraphNode& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ignored, IgnoredNode(downstream));
      if (ignored) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(drr_downstream, downstream.Get());
      ADT_LET_CONST_REF(casted, drr_downstream.template TryGet<DrrNodeT>());
      ADT_CHECK(!opt_sole_output.has_value());
      opt_sole_output = casted;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitDownstreamNodes(node.node(), DoEachDownstream));
    ADT_CHECK(opt_sole_output.has_value());
    return opt_sole_output.value();
  }

  adt::Result<DrrNode> GetSoleOutput(const DrrNode& node) const {
    std::optional<DrrNode> opt_sole_output{};
    auto DoEachDownstream =
        [&](const DrrGraphNode& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(drr_downstream, downstream.Get());
      ADT_CHECK(!opt_sole_output.has_value());
      opt_sole_output = drr_downstream;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitDownstreamNodes(node.node(), DoEachDownstream));
    ADT_CHECK(opt_sole_output.has_value());
    return opt_sole_output.value();
  }

  adt::Result<std::size_t> GetNumInputs(const DrrNode& node) const {
    std::size_t num_inputs = 0;
    auto DoEachUpstream =
        [&](const DrrGraphNode& upstream) -> adt::Result<adt::Ok> {
      ++num_inputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitUpstreamNodes(node.node(), DoEachUpstream));
    return num_inputs;
  }

  adt::Result<std::size_t> GetNumOutputs(const DrrNode& node) const {
    std::size_t num_outputs = 0;
    auto DoEachDownstream =
        [&](const DrrGraphNode& downstream) -> adt::Result<adt::Ok> {
      ++num_outputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitDownstreamNodes(node.node(), DoEachDownstream));
    return num_outputs;
  }

  adt::Result<graph::SmallGraphNodeTopoCstr> GetSmallGraphNodeTopoCstr(
      const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return graph::SmallGraphNodeTopoCstr{drr_node.node_topo_cstr()};
  }

  adt::Result<bool> IgnoredNode(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [](const DrrPackedIrValue&) -> adt::Result<bool> { return true; },
        [&](const DrrPackedIrOpOperand& impl) -> adt::Result<bool> {
          ADT_LET_CONST_REF(upstreams, impl->node.UpstreamNodes());
          ADT_CHECK(upstreams.size() == 1);
          ADT_LET_CONST_REF(upstream_node, upstreams.Sole());
          return IgnoredNode(upstream_node);
        },
        [&](const DrrPackedIrOpResult& impl) -> adt::Result<bool> {
          ADT_LET_CONST_REF(downstreams, impl->node.DownstreamNodes());
          ADT_CHECK(downstreams.size() == 1);
          ADT_LET_CONST_REF(downstream_node, downstreams.Sole());
          return IgnoredNode(downstream_node);
        },
        [](const DrrNativeIrValue&) -> adt::Result<bool> { return false; },
        [](const DrrNativeIrOp&) -> adt::Result<bool> { return false; },
        [](const DrrPackedIrOp&) -> adt::Result<bool> { return false; },
        [](const DrrOptPackedIrOp&) -> adt::Result<bool> { return false; },
        [](const DrrNativeIrOpOperand&) -> adt::Result<bool> { return false; },
        [&](const DrrOptPackedIrOpOperand& impl) -> adt::Result<bool> {
          ADT_LET_CONST_REF(upstreams, impl->node.UpstreamNodes());
          ADT_CHECK(upstreams.size() == 1);
          ADT_LET_CONST_REF(upstream_node, upstreams.Sole());
          return IgnoredNode(upstream_node);
        },
        [](const DrrNativeIrOpResult&) -> adt::Result<bool> { return false; },
        [&](const DrrOptPackedIrOpResult& impl) -> adt::Result<bool> {
          ADT_LET_CONST_REF(downstreams, impl->node.DownstreamNodes());
          ADT_CHECK(downstreams.size() == 1);
          ADT_LET_CONST_REF(downstream_node, downstreams.Sole());
          return IgnoredNode(downstream_node);
        });
  }

  adt::Result<bool> IsOpNode(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [](const DrrNativeIrOp&) -> bool { return true; },
        [](const DrrPackedIrOp&) -> bool { return true; },
        [](const DrrOptPackedIrOp&) -> bool { return true; },
        [](const DrrNativeIrValue&) -> bool { return false; },
        [](const DrrPackedIrValue&) -> bool { return false; },
        [](const DrrNativeIrOpOperand&) -> bool { return false; },
        [](const DrrPackedIrOpOperand&) -> bool { return false; },
        [](const DrrOptPackedIrOpOperand&) -> bool { return false; },
        [](const DrrNativeIrOpResult&) -> bool { return false; },
        [](const DrrPackedIrOpResult&) -> bool { return false; },
        [](const DrrOptPackedIrOpResult&) -> bool { return false; });
  }

  adt::Result<bool> IsValueNode(const NodeT& node) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [](const DrrNativeIrOp&) -> bool { return false; },
        [](const DrrPackedIrOp&) -> bool { return false; },
        [](const DrrOptPackedIrOp&) -> bool { return false; },
        [](const DrrNativeIrValue&) -> bool { return true; },
        [](const DrrPackedIrValue&) -> bool { return true; },
        [](const DrrNativeIrOpOperand&) -> bool { return false; },
        [](const DrrPackedIrOpOperand&) -> bool { return false; },
        [](const DrrOptPackedIrOpOperand&) -> bool { return false; },
        [](const DrrNativeIrOpResult&) -> bool { return false; },
        [](const DrrPackedIrOpResult&) -> bool { return false; },
        [](const DrrOptPackedIrOpResult&) -> bool { return false; });
  }

  adt::Result<bool> TopoSatisfy(
      const NodeT& node,
      const graph::SmallGraphNodeTopoCstr& node_topo_cstr) const {
    ADT_LET_CONST_REF(drr_node, node.Get());
    const graph::BigGraphNodeTopoCstr& drr_node_topo_cstr{
        drr_node.node_topo_cstr()};
    return drr_node_topo_cstr.TopoSatisfy(node_topo_cstr);
  }
};

struct AllOperandAndResultDrrGraphDescriptor {
  using DrrNode = drr::Node;
  using DrrGraphNode = graph::Node<DrrNode>;
  using NodeT = DrrGraphNode;

  DefaultDrrGraphDescriptor backend_graph;

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

  template <typename DrrNodeT>
  adt::Result<DrrNodeT> CastSoleUnignoredInput(const DrrNode& node) const {
    std::optional<DrrNodeT> opt_sole_input{};
    auto DoEachUpstream =
        [&](const DrrGraphNode& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ignored, IgnoredNode(upstream));
      if (ignored) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(drr_upstream, upstream.Get());
      ADT_LET_CONST_REF(casted, drr_upstream.template TryGet<DrrNodeT>());
      ADT_CHECK(!opt_sole_input.has_value());
      opt_sole_input = casted;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitUpstreamNodes(node.node(), DoEachUpstream));
    ADT_CHECK(opt_sole_input.has_value());
    return opt_sole_input.value();
  }

  adt::Result<std::size_t> GetNumInputs(const DrrNode& node) const {
    std::size_t num_inputs = 0;
    auto DoEachUpstream =
        [&](const DrrGraphNode& upstream) -> adt::Result<adt::Ok> {
      ++num_inputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitUpstreamNodes(node.node(), DoEachUpstream));
    return num_inputs;
  }

  adt::Result<std::size_t> GetNumOutputs(const DrrNode& node) const {
    std::size_t num_outputs = 0;
    auto DoEachDownstream =
        [&](const DrrGraphNode& downstream) -> adt::Result<adt::Ok> {
      ++num_outputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitDownstreamNodes(node.node(), DoEachDownstream));
    return num_outputs;
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

struct NativeOperandAndResultDrrGraphDescriptor {
  using DrrNode = drr::Node;
  using DrrGraphNode = graph::Node<DrrNode>;
  using NodeT = DrrGraphNode;

  AllOperandAndResultDrrGraphDescriptor backend_graph;

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
    ADT_LET_CONST_REF(drr_node, node.Get());
    return drr_node.Match(
        [&](const NativeIrOpOperand<DrrNode>&) -> bool { return true; },
        [&](const NativeIrOpResult<DrrNode>&) -> bool { return true; },
        [&](const auto&) -> bool { return false; });
  }
};

}  // namespace ap::drr

namespace ap::graph {

template <>
struct GraphDescriptor<graph::Node<drr::Node>, drr::topo_kind::Default>
    : public drr::DefaultDrrGraphDescriptor {};

template <>
struct GraphDescriptor<graph::Node<drr::Node>,
                       drr::topo_kind::AllOperandAndResult>
    : public drr::AllOperandAndResultDrrGraphDescriptor {};

template <>
struct GraphDescriptor<graph::Node<drr::Node>,
                       drr::topo_kind::NativeOperandAndResult>
    : public drr::NativeOperandAndResultDrrGraphDescriptor {};

}  // namespace ap::graph

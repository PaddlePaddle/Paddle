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

#include "paddle/cinn/frontend/cluster_ops/shardable_axes_inferer.h"

#include "paddle/cinn/frontend/cluster_ops/shardable_axes_provider.h"
#include "paddle/cinn/frontend/cluster_ops/shardable_axes_utils.h"

namespace cinn::frontend::cluster_ops {

ShardableAxesSignature ShardableAxesInferer::MakeShardableAxesSignature4Op(
    const pir::Operation* op) {
  return shardable_axes_provider_->MakeShardableAxesSignature4Op(op);
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::InferShardableAxesFromSink(const pir::Operation* sink,
                                                 const OpTopo& op_topo) {
  auto reversed_walker = GetOpsReversedTopoWalker(op_topo);
  CHECK_GT(op_topo.ops->count(sink), 0);
  const int result_idx = GetOutputShardableAxesResultIdx(sink);
  size_t rank = GetRank(sink->result(result_idx));
  const auto& init_sa = MakeFullyShardableAxes(rank);
  return ReversedInferShardableAxes(reversed_walker, sink, init_sa);
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::InferShardableAxes(const OpSetPtr& ops) {
  auto reversed_walker = GetOpsReversedTopoWalker(OpTopo{
      .ops = ops,
  });
  const auto& sinks = GetSinks(*ops);
  const auto& sink_and_init_value =
      GetSinkAndInitValues(reversed_walker, ops, sinks);
  return ReversedInferShardableAxes(
      reversed_walker, sink_and_init_value.begin(), sink_and_init_value.end());
}

template <typename InputIt>
std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::ReversedInferShardableAxes(
    const common::TopoWalker<const pir::Operation*>& reversed_walker,
    InputIt sink_and_init_begin,
    InputIt sink_and_init_end) {
  std::unordered_map<pir::Value, ShardableAxes> value2shardable_axes;
  std::list<const pir::Operation*> sinks;
  for (auto iter = sink_and_init_begin; iter != sink_and_init_end; ++iter) {
    sinks.push_back(iter->first.defining_op());
    value2shardable_axes[iter->first] = iter->second;
  }
  const auto& UpdateValue2ShardableAxes = [&](pir::Value value,
                                              const ShardableAxes& sa) {
    auto iter = value2shardable_axes.find(value);
    if (iter != value2shardable_axes.end()) {
      iter->second = GetCommonShardableAxes(iter->second, sa);
    } else {
      value2shardable_axes[value] = sa;
    }
  };
  reversed_walker(sinks.begin(), sinks.end(), [&](const auto* op) {
    auto shardable_axes_sig = MakeShardableAxesSignature4Op(op);
    const auto& sole_output_sa = shardable_axes_sig.sole_output_sa;
    const int result_idx = GetOutputShardableAxesResultIdx(op);
    const auto& old2new =
        GetOldName2NewName(sole_output_sa.shardable_axes,
                           value2shardable_axes.at(op->result(result_idx)));
    for (auto& pair : shardable_axes_sig.input_shardable_axes) {
      const auto& [my_op, input_idx] = pair.first;
      CHECK_EQ(my_op, op);
      auto* input_shardable_axes = &pair.second;
      UpdateShardableAxes(old2new, input_shardable_axes);
      pir::Value input_value = op->operand_source(input_idx);
      UpdateValue2ShardableAxes(input_value, *input_shardable_axes);
    }
  });
  return value2shardable_axes;
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::ReversedInferShardableAxes(
    const common::TopoWalker<const pir::Operation*>& reversed_walker,
    const pir::Operation* sink,
    const ShardableAxes& init_sa) {
  using OpAndInitValue = std::pair<pir::Value, ShardableAxes>;
  const int result_idx = GetOutputShardableAxesResultIdx(sink);
  std::array<OpAndInitValue, 1> sinks{
      OpAndInitValue{sink->result(result_idx), init_sa}};
  return ReversedInferShardableAxes(
      reversed_walker, sinks.begin(), sinks.end());
}

std::unordered_map<const pir::Operation*, ShardableAxesSignature>
ShardableAxesInferer::GetOp2ShardableAxesSignature(const OpSetPtr& ops) {
  std::unordered_map<const pir::Operation*, ShardableAxesSignature> ret;
  for (const auto* op : *ops) {
    ret[op] = MakeShardableAxesSignature4Op(op);
  }
  return ret;
}

std::map<std::string, std::vector<std::string>>
ShardableAxesInferer::GetAxisName2BoundAxisName(
    const OpSetPtr& ops,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
        op2shardable_axes_signature) {
  const auto GetInputShardableAxes = [&](const OpAndOperandIndex& op_and_idx)
      -> std::optional<const ShardableAxes*> {
    const auto& [op, idx] = op_and_idx;
    const auto* input_op = op->operand_source(idx).defining_op();
    if (ops->count(input_op) == 0) return std::nullopt;
    const auto& iter = op2shardable_axes_signature.find(input_op);
    if (iter == op2shardable_axes_signature.end()) return std::nullopt;
    const auto& output_sa = iter->second.sole_output_sa.shardable_axes;
    return &output_sa;
  };
  std::map<std::string, std::vector<std::string>> axis_name2bound_axis_name;
  const auto UpdateAxisName2BoundAxisName = [&](const ShardableAxes& input_sa,
                                                const ShardableAxes& sa) {
    for (const auto& [input_axis, input_axis_name] : input_sa) {
      for (const auto& [axis, axis_name] : sa) {
        if (input_axis != axis) continue;
        axis_name2bound_axis_name[axis_name].push_back(input_axis_name);
        axis_name2bound_axis_name[input_axis_name].push_back(axis_name);
      }
    }
  };
  for (const auto& [op, signature] : op2shardable_axes_signature) {
    for (const auto& [op_and_idx, sa] : signature.input_shardable_axes) {
      const auto& input_sa = GetInputShardableAxes(op_and_idx);
      if (!input_sa.has_value()) continue;
      UpdateAxisName2BoundAxisName(*input_sa.value(), sa);
    }
  }
  return axis_name2bound_axis_name;
}

std::unordered_map<std::string, std::string>
ShardableAxesInferer::GetAxisName2UnionFindSetRoot(
    const OpSetPtr& ops,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
        op2shardable_axes_signature) {
  const auto axis_name2bound_axis_name =
      GetAxisName2BoundAxisName(ops, op2shardable_axes_signature);
  using NodeVisitor = std::function<void(const std::string&)>;
  const auto VisitNext = [&](const std::string& axis_name,
                             const NodeVisitor& DoEach) {
    const auto& iter = axis_name2bound_axis_name.find(axis_name);
    if (iter == axis_name2bound_axis_name.end()) return;
    for (const auto& input_axis_name : iter->second) {
      DoEach(input_axis_name);
    }
  };
  common::BfsWalker<std::string> walk(VisitNext);
  std::unordered_map<std::string, std::string> axis_name2root;
  for (const auto& [union_find_root, _] : axis_name2bound_axis_name) {
    if (axis_name2root.count(union_find_root) > 0) continue;
    walk(union_find_root, [&](const std::string& axis_name) {
      CHECK(axis_name2root.emplace(axis_name, union_find_root).second);
    });
  }
  return axis_name2root;
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::GetSinkAndInitShardableAxes(
    const std::list<const pir::Operation*>& sinks,
    const std::unordered_map<const pir::Operation*, ShardableAxesSignature>&
        op2shardable_axes_signature,
    const std::unordered_map<std::string, std::string>&
        axis_name2union_find_set_root) {
  const auto& ConvertByBoundAxisName = [&](const ShardableAxes& sa) {
    ShardableAxes ret_sa;
    for (const auto& [axis, axis_name] : sa) {
      const auto& iter = axis_name2union_find_set_root.find(axis_name);
      CHECK(iter != axis_name2union_find_set_root.end());
      ret_sa.emplace_back(ShardableAxis{
          .axis = axis,
          .axis_name = iter->second,
      });
    }
    return ret_sa;
  };
  std::unordered_map<pir::Value, ShardableAxes> sink2sa;
  for (const auto* sink : sinks) {
    const auto& sig_iter = op2shardable_axes_signature.find(sink);
    CHECK(sig_iter != op2shardable_axes_signature.end());
    const auto& sole_output_sa = sig_iter->second.sole_output_sa;
    const auto& output_shardable_axes = sole_output_sa.shardable_axes;
    const int result_idx = GetOutputShardableAxesResultIdx(sink);
    sink2sa[sink->result(result_idx)] =
        ConvertByBoundAxisName(output_shardable_axes);
  }
  return sink2sa;
}

void ShardableAxesInferer::RenameDuplicatedAxisName(
    std::unordered_map<pir::Value, ShardableAxes>* sink2sa) {
  const auto& RenameDuplicated = [&](ShardableAxes* sa) {
    std::set<std::string> existed_axis_name;
    for (auto& [_, axis_name] : *sa) {
      if (!existed_axis_name.emplace(axis_name).second) {
        axis_name =
            axis_name + "_" + std::to_string(ShardableAxis::UnqiueSeqNo());
      } else {
        // do nothing.
      }
    }
  };
  for (auto& [_, sa] : *sink2sa) {
    RenameDuplicated(&sa);
  }
}

std::unordered_map<pir::Value, ShardableAxes>
ShardableAxesInferer::GetSinkAndInitValues(
    const common::TopoWalker<const pir::Operation*>& reverse_walker,
    const OpSetPtr& ops,
    const std::list<const pir::Operation*>& sinks) {
  const auto& op2shardable_axes_signature = GetOp2ShardableAxesSignature(ops);
  const auto& axis_name2union_find_set_root =
      GetAxisName2UnionFindSetRoot(ops, op2shardable_axes_signature);
  std::unordered_map<pir::Value, ShardableAxes> sink_and_inits =
      GetSinkAndInitShardableAxes(
          sinks, op2shardable_axes_signature, axis_name2union_find_set_root);
  RenameDuplicatedAxisName(&sink_and_inits);
  return sink_and_inits;
}

}  // namespace cinn::frontend::cluster_ops

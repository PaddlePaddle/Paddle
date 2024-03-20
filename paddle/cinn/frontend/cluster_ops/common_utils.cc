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

#include "paddle/cinn/frontend/cluster_ops/common_utils.h"

namespace cinn::frontend::cluster_ops {

std::list<const pir::Operation*> GetSinks(const OpSet& ops) {
  const auto IsSink = [&](const pir::Operation* op) {
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      for (auto consumer_it = output.use_begin();
           consumer_it != output.use_end();
           ++consumer_it) {
        const auto* consumer_op = consumer_it->owner();
        if (consumer_op->isa<pir::YieldOp>()) continue;
        if (ops.count(consumer_op) > 0) return false;
      }
    }
    return true;
  };
  std::list<const pir::Operation*> sinks;
  for (const auto* op : ops) {
    if (IsSink(op)) {
      sinks.push_back(op);
    }
  }
  return sinks;
}

const pir::Operation* GetSoleSink(const OpSet& ops) {
  const auto& sinks = GetSinks(ops);
  CHECK_EQ(sinks.size(), 1);
  return *sinks.begin();
}

common::TopoWalker<const pir::Operation*> GetOpsReversedTopoWalker(
    const OpTopo& op_topo) {
  const auto VisitUpStreamInOps = [op_topo](const pir::Operation* op,
                                            const OpVisitor& DoEach) {
    op_topo.VisitInputOp(op, DoEach);
  };
  const auto VisitDownStreamInOps = [op_topo](const pir::Operation* op,
                                              const OpVisitor& DoEach) {
    op_topo.VisitOutputOp(op, DoEach);
  };
  common::TopoWalker<const pir::Operation*> reversed_walker(
      VisitDownStreamInOps, VisitUpStreamInOps);
  return reversed_walker;
}

size_t GetRank(pir::Value value) {
  return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
}

std::vector<int64_t> GetReduceAxes(const pir::Operation* reduce_op) {
  const size_t input_rank = GetRank(reduce_op->operand_source(0));
  const auto& attr_val = reduce_op->attributes().at("dim");
  CHECK(attr_val.isa<::pir::ArrayAttribute>());
  const auto& axis_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  std::vector<int64_t> reduce_axes;
  for (int i = 0; i < axis_attr.size(); ++i) {
    int64_t axis = axis_attr.at(i).dyn_cast<::pir::Int64Attribute>().data();
    if (axis < 0) {
      axis += input_rank;
    }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_rank);
    reduce_axes.push_back(axis);
  }
  return reduce_axes;
}

bool GetReduceOpKeepDims(const pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keep_dim");
  CHECK(attr_val.isa<::pir::BoolAttribute>());
  return attr_val.dyn_cast<::pir::BoolAttribute>();
}

std::function<size_t(const pir::Operation*)> MakeTopoOrderFinderOfOp(
    const std::vector<const pir::Operation*>& ops) {
  std::unordered_map<const pir::Operation*, size_t> op2order_in_block;
  size_t order = 0;
  for (const pir::Operation* op : ops) {
    op2order_in_block[op] = ++order;
  }
  return [map = std::move(op2order_in_block)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}

}

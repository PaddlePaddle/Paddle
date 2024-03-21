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

#include "paddle/cinn/frontend/cluster_ops/group_pattern.h"

namespace cinn::frontend::cluster_ops {

bool IsISPattern(const StmtPattern& pattern);

bool IsPSPattern(const StmtPattern& pattern);

bool IsRPattern(const StmtPattern& pattern);

template <typename DoEachT>
void VisitStmtOpImpl(const IS& injective_source, const DoEachT& DoEach) {
  for (const auto* op : injective_source.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const PS& partial_shardable, const DoEachT& DoEach) {
  for (const auto* op : partial_shardable.ops) {
    DoEach(op);
  }
}

template <typename DoEachT>
void VisitStmtOpImpl(const R& reduce, const DoEachT& DoEach) {
  std::visit(adt::match{
                 [](const std::monostate&) {
                   // do nothing.
                 },
                 [&](const IS& injective_source) {
                   VisitStmtOpImpl(injective_source, DoEach);
                 },
                 [&](const PS& partial_shardable) {
                   VisitStmtOpImpl(partial_shardable, DoEach);
                 },
             },
             reduce.input);
  DoEach(reduce.reduce_op_pattern.reduce_op);
}

template <typename DoEachT>
void VisitStmtOp(const StmtPattern& stmt, const DoEachT& DoEach) {
  std::visit([&](const auto& impl) { VisitStmtOpImpl(impl, DoEach); }, stmt);
}

pir::Value GetStmtBigestShapeValue(const StmtPattern& stmt);

const pir::Operation* GetStmtSoleSinkOp(const StmtPattern& stmt);

void SortStmtPtrs(
    std::vector<const StmtPattern*>* stmt_ptrs,
    const std::function<size_t(const pir::Operation*)>& OrderValue4Op);

common::TopoWalker<const StmtPattern*> MakeTopoWalker(
    const OpTopo& op_topo, const std::vector<StmtPattern>& stmt_patterns);

std::function<bool(const pir::Operation*)> MakePredicatorIsInjectiveSource(
    const OpTopo& op_topo);

std::vector<const pir::Operation*> GetStmtContainedOps(const StmtPattern& stmt);

std::string StmtDebugStr(const StmtPattern& stmt);

struct LoopAlignableStmtPatternVec {
  std::vector<StmtPattern> stmts;
  std::string DebugStr() const;
};

struct ClusteringResult {
  std::vector<LoopAlignableStmtPatternVec> loop_alignable_list;
  std::string DebugStr() const;
};

}  // namespace cinn::frontend::cluster_ops

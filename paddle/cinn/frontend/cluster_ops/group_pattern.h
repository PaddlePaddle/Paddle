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

#include <atomic>
#include <unordered_map>
#include <variant>
#include <vector>
#include "glog/logging.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/tree.h"
#include "paddle/cinn/api/op_topo_pattern.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace cinn::frontend {
  struct FrontendPattern {};
}

namespace cinn::api {

template <>
struct ErrorPattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
  std::string error_string;
};

template <>
struct InjectiveSourcePattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
  const pir::Operation* sole_sink;
};

template <>
struct SingleReductionOpPattern<frontend::FrontendPattern> {
  const pir::Operation* reduce_op;
};
template <>
struct PartialShardablePattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
  const pir::Operation* sole_sink;
  frontend::ShardableAxesSignature shardable_axes_signature;
};

}  // namespace cinn::api

namespace cinn::frontend {

using ErrorGroupPattern = api::ErrorPattern<FrontendPattern>;
using GroupPattern = api::OpTopoPattern<FrontendPattern>;

struct LoopAlignableStmtsPattern {
  std::vector<api::StmtPattern<FrontendPattern>> stmts;
};

struct ClusteringResult {
  std::vector<LoopAlignableStmtsPattern> loop_alignable_list;
};

namespace cluster_ops {
using IS = api::InjectiveSourcePattern<frontend::FrontendPattern>;
using R = api::ReductionPattern<frontend::FrontendPattern>;
using PS = api::PartialShardablePattern<frontend::FrontendPattern>;
using StmtPattern = api::StmtPattern<frontend::FrontendPattern>;
using StmtsPattern = api::StmtsPattern<frontend::FrontendPattern>;
using StmtVisitor = std::function<void(const StmtPattern*)>;
}  // namespace cluster_ops

}  // namespace cinn::frontend

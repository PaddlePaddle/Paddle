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

#include "paddle/cinn/api/op_topo_pattern.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

#include "paddle/cinn/frontend/cluster_ops/shardable_axes_inferer.h"
#include "paddle/cinn/frontend/cluster_ops/shardable_axes_provider.h"

namespace cinn::frontend {
struct FrontendPattern {};
}  // namespace cinn::frontend

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
  frontend::cluster_ops::ShardableAxesSignature shardable_axes_signature;
};

}  // namespace cinn::api

namespace cinn::frontend {

using ErrorGroupPattern = api::ErrorPattern<FrontendPattern>;
using GroupPattern = api::OpTopoPattern<FrontendPattern>;

}  // namespace cinn::frontend

namespace cinn::frontend::cluster_ops {
using IS = api::InjectiveSourcePattern<frontend::FrontendPattern>;
using R = api::ReductionPattern<frontend::FrontendPattern>;
using PS = api::PartialShardablePattern<frontend::FrontendPattern>;
using StmtPattern = api::StmtPattern<frontend::FrontendPattern>;
using StmtPatternVec = api::StmtPatternVec<frontend::FrontendPattern>;
using StmtVisitor = std::function<void(const StmtPattern*)>;

}  // namespace cinn::frontend::cluster_ops

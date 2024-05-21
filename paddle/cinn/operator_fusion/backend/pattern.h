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
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

struct BackendStage {};

template <>
struct PatternContent<BackendStage> {
  explicit PatternContent<BackendStage>(pir::Operation* op,
                                        std::optional<ir::Expr> expr)
      : op(op), expr(expr) {}
  pir::Operation* op;
  std::optional<ir::Expr> expr;
};

using BackendContent = PatternContent<BackendStage>;
using TrivialOp = cinn::hlir::framework::pir::trivial_fusion_detail::TrivialOp;
using ReduceOp = cinn::hlir::framework::pir::trivial_fusion_detail::ReduceOp;
using FusionOp = std::variant<ReduceOp, TrivialOp>;
template <>
struct TrivialPattern<BackendStage> {
  explicit TrivialPattern(const std::vector<pir::Operation*>& ops,
                          pir::Operation* sink_op,
                          const TrivialOp& op)
      : ops_(ops), sink_op(sink_op), trivial_op(op) {}
  std::vector<pir::Operation*> ops_;
  pir::Operation* sink_op;
  TrivialOp trivial_op;
  static std::string name() { return "Trivial"; }
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* sink() const { return sink_op; }
};

template <>
struct ReducePattern<BackendStage> {
  explicit ReducePattern(const std::vector<pir::Operation*>& ops,
                         const ReduceOp& op)
      : ops_(ops), reduce_op(op) {}
  std::vector<pir::Operation*> ops_;
  ReduceOp reduce_op;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* GetReduceOp() const { return ops_.back(); }
  static std::string name() { return "Reduce"; }
};

template <>
struct UnsupportPattern<BackendStage> {
  explicit UnsupportPattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  static std::string name() { return "Unsupport"; }
};

}  // namespace cinn::fusion

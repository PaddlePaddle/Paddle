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
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

struct FrontendStage {};

template <>
struct PatternContent<FrontendStage> {
  explicit PatternContent<FrontendStage>(pir::Operation* op) : op(op) {}
  pir::Operation* op;
  bool operator==(const PatternContent<FrontendStage>& other) const {
    return op == other.op;
  }
};

using FrontendContent = PatternContent<FrontendStage>;

}  // namespace cinn::fusion

namespace std {
template <>
struct hash<cinn::fusion::FrontendContent> {
  size_t operator()(const cinn::fusion::FrontendContent& content) const {
    return std::hash<pir::Operation*>()(content.op);
  }
};

}  // namespace std

namespace cinn::fusion {
template <>
struct TrivialPattern<FrontendStage> {
  explicit TrivialPattern(const std::vector<pir::Operation*>& ops,
                          pir::Operation* sink_op)
      : ops_(ops), sink_op(sink_op) {}
  std::vector<pir::Operation*> ops_;
  pir::Operation* sink_op;
  static std::string name() { return "Trivial"; }
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* sink() const { return sink_op; }
};

template <>
struct ReducePattern<FrontendStage> {
  explicit ReducePattern(const std::vector<pir::Operation*>& ops) : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* GetReduceOp() const { return ops_.back(); }
  static std::string name() { return "Reduce"; }
};

template <>
struct UnsupportPattern<FrontendStage> {
  explicit UnsupportPattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  static std::string name() { return "Unsupport"; }
};

}  // namespace cinn::fusion

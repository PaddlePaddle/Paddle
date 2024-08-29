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

#include <variant>
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {
struct TransformInfo {
  explicit TransformInfo(pir::Operation* op_,
                         size_t input_idx_,
                         size_t output_idx_,
                         bool is_upstream_anchor_)
      : op(op_),
        input_idx(input_idx_),
        output_idx(output_idx_),
        is_upstream_anchor(is_upstream_anchor_) {}
  pir::Value InputValue() { return op->operand_source(input_idx); }
  pir::Value OutputValue() { return op->result(output_idx); }
  pir::Value SrcValue() {
    if (is_upstream_anchor) {
      return OutputValue();
    } else {
      return InputValue();
    }
  }
  pir::Value DstValue() {
    if (is_upstream_anchor) {
      return InputValue();
    } else {
      return OutputValue();
    }
  }

  std::string DebugStr() const {
    std::stringstream ss;
    ss << "op: " << op->name() << "[id:" << op->id()
       << "], input_idx: " << input_idx << ", output_idx: " << output_idx
       << ", is_upstream_anchor: " << is_upstream_anchor;
    return ss.str();
  }

  pir::Operation* op;
  size_t input_idx;
  size_t output_idx;
  bool is_upstream_anchor;
};

struct UnsupportTransform {
  explicit UnsupportTransform(const TransformInfo& info_) : info(info_) {}
  TransformInfo info;
  std::string DebugStr() const {
    std::stringstream ss;
    ss << "UnsupportTransform || " << info.DebugStr();
    return ss.str();
  }
};

struct IdentityTransform {
  explicit IdentityTransform(const TransformInfo& info_) : info(info_) {}
  TransformInfo info;
  std::string DebugStr() const {
    std::stringstream ss;
    ss << "IdentityTransform || " << info.DebugStr();
    return ss.str();
  }
};

struct AppendDimTransform {
  explicit AppendDimTransform(const TransformInfo& info_,
                              const std::vector<size_t>& append_dims_)
      : info(info_), append_dims(append_dims_) {}
  TransformInfo info;
  std::vector<size_t> append_dims;
  std::string DebugStr() const {
    std::stringstream ss;
    ss << "AppendDimTransform || " << info.DebugStr();
    return ss.str();
  }
};

struct DeleteDimTransform {
  explicit DeleteDimTransform(const TransformInfo& info_,
                              const std::vector<size_t>& delete_dims_)
      : info(info_), delete_dims(delete_dims_) {}
  TransformInfo info;
  std::vector<size_t> delete_dims;
  std::string DebugStr() const {
    std::stringstream ss;
    ss << "DeleteDimTransform || " << info.DebugStr();
    return ss.str();
  }
};

using UnsupportTransformPtr = std::shared_ptr<UnsupportTransform>;
using IdentityTransformPtr = std::shared_ptr<IdentityTransform>;
using AppendDimTransformPtr = std::shared_ptr<AppendDimTransform>;
using DeleteDimTransformPtr = std::shared_ptr<DeleteDimTransform>;

using AnchorTransform = std::variant<UnsupportTransformPtr,
                                     IdentityTransformPtr,
                                     AppendDimTransformPtr,
                                     DeleteDimTransformPtr>;
using AnchorTransformRoute = std::vector<AnchorTransform>;

std::string DebugStrOfAnchorTransformRoute(const AnchorTransformRoute& route);

struct ExprPromise {
  explicit ExprPromise(const pir::Value& root, const std::string& id)
      : root_value(root), id_(id) {}
  AnchorTransformRoute transform_route;
  pir::Value root_value;
  std::string id_;

  void update(const AnchorTransformRoute& route) {
    transform_route.insert(transform_route.end(), route.begin(), route.end());
  }

  std::string DebugStr() const {
    std::stringstream ss;
    ss << "[ExprPromise] root_value: " << root_value.impl() << "\n";
    ss << DebugStrOfAnchorTransformRoute(transform_route) << "\n";
    return ss.str();
  }
};

struct AnchorState {
  explicit AnchorState(const std::vector<ExprPromise>& init_promise)
      : promise(init_promise) {}
  std::vector<ExprPromise> promise;

  void update(AnchorState new_state) {
    auto new_exprs = new_state.promise;
    promise.insert(promise.end(), new_exprs.begin(), new_exprs.end());
  }

  std::string DebugStr() const {
    std::stringstream ss;
    ss << "[AnchorState]\n";
    for (const auto& expr_promise : promise) {
      ss << expr_promise.DebugStr() << "\n";
    }
    return ss.str();
  }
};

AnchorTransform CreateAnchorTransform(const TransformInfo& info);
std::vector<AnchorTransform> PossibleTransform(
    pir::Value v, const std::unordered_set<pir::Operation*>& ops);
TransformInfo GetTransformInfo(AnchorTransform trans);
std::string DebugStrOfAnchorTransform(const AnchorTransform& trans);
}  // namespace cinn::fusion

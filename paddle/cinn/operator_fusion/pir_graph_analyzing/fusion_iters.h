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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/shardable_axes_base.h"

namespace cinn::fusion {

using FusionIters = std::vector<std::string>;
struct FusionItersSignature {
  FusionItersSignature() = default;
  std::string DebugStr() const;
  // The following iters are generated by ShardableAxesInfoManager, and related
  // iters have the same id, see ShardableAxesInfoManager::GetModifiedSignature.
  FusionIters loop_iters = {};
  size_t reduce_iter_nums = 0;
  std::set<pir::Value> input_values = {};
  std::set<pir::Value> output_values = {};
};

struct FusionItersManager {
  FusionItersManager(pir::ShapeConstraintIRAnalysis* shape_analysis,
                     ShardableAxesInfoManager* axes_info)
      : shape_analysis_(shape_analysis), axes_info_(axes_info) {
    PADDLE_ENFORCE_NOT_NULL(shape_analysis,
                            "shape_analysis should not be nullptr.");
    PADDLE_ENFORCE_NOT_NULL(axes_info, "axes_info should not be nullptr.");
  }
  FusionItersSignature GetItersSignature(pir::Operation* op);

  std::string PrintItersSignature(const FusionItersSignature& sig);

  FusionItersSignature SingleDownstreamItersFusion(
      const FusionItersSignature& upstream,
      const FusionItersSignature& downstream);
  FusionItersSignature MultiDownstreamItersFusion(
      const FusionItersSignature& upstream,
      const FusionItersSignature& downstream);

  bool IterSymbolEqual(const std::string& lhs, const std::string& rhs);

 private:
  void StoreIter2DimExprForValue(const pir::Value& value);

  std::unordered_map<pir::Value, FusionIters> value2iters_;
  std::unordered_map<pir::Value, size_t> value_remain_usage_;
  std::unordered_map<std::string, symbol::DimExpr> iter2dimexpr_;

  pir::ShapeConstraintIRAnalysis* shape_analysis_;
  ShardableAxesInfoManager* axes_info_;
};

// Fusion Iters Transform
struct IdentityItersTransform {
  IdentityItersTransform() = default;
  std::string DebugStr() const { return "Identity"; }
};
struct TransposeItersTransform {
  TransposeItersTransform() = default;
  explicit TransposeItersTransform(const std::vector<int32_t>& perm)
      : perm_(perm) {}
  std::string DebugStr() const {
    return "Transpose(perm={" + cinn::utils::Join(perm_, ",") + "})";
  }
  std::vector<int32_t> perm_;
};
struct AppendItersTransform {
  AppendItersTransform() = default;
  explicit AppendItersTransform(const std::vector<int32_t>& axis)
      : axis_(axis) {}
  std::string DebugStr() const {
    return "AppendIters(axis={" + cinn::utils::Join(axis_, ",") + "})";
  }
  std::vector<int32_t> axis_;
};
struct ReuseItersTransform {
  using IterMap = std::unordered_map<std::string, std::string>;
  ReuseItersTransform() = default;
  explicit ReuseItersTransform(const IterMap& reuse_target_to_source)
      : reuse_target_to_source_(reuse_target_to_source) {}
  std::string DebugStr() const {
    std::string result = "ReuseIters(";
    for (const auto& [t, s] : reuse_target_to_source_) {
      result += s + "->" + t + ",";
    }
    return result.substr(0, result.size() - 1) + ")";
  }
  IterMap reuse_target_to_source_;
};
using ItersTransform = std::variant<IdentityItersTransform,
                                    TransposeItersTransform,
                                    AppendItersTransform,
                                    ReuseItersTransform>;
using ItersTransformRoute = std::vector<ItersTransform>;

}  // namespace cinn::fusion

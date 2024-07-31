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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/anchor_transform.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"

namespace cinn::fusion {

AnchorTransform CreateDefaultAnchorTransform(const TransformInfo& info) {
  return std::make_shared<UnsupportTransform>(info);
}

AnchorTransform CreateAnchorTransformForElementWise(const TransformInfo& info) {
  return std::make_shared<IdentityTransform>(info);
}

AnchorTransform CreateAnchorTransformForReduce(const TransformInfo& info) {
  if (info.op->num_operands() != 1 || info.input_idx != 0) {
    return CreateDefaultAnchorTransform(info);
  }

  if (info.is_upstream_anchor) {
    // TODO(@wuzhanfei) do not support reduce currently

    // return std::make_shared<AppendDimTransform>(
    //     info, GetReduceAxisIdx(pir::Operation * reduce_op))
    return CreateDefaultAnchorTransform(info);
  } else {
    return CreateDefaultAnchorTransform(info);
  }
}

AnchorTransform CreateAnchorTransformForBroadcast(const TransformInfo& info) {
  if (info.op->num_operands() != 1 || info.input_idx != 0) {
    return CreateDefaultAnchorTransform(info);
  }

  if (info.is_upstream_anchor) {
    // TODO(@wuzhanfei) do not support broadcast currently

    // auto related_dims = GetNonBroadCastDims(info.op);
    // size_t output_rank = GetRank(info.OutputValue());
    // std::vector<size_t> used_dim_mask = std::vector<size_t>(output_rank, 0);
    // std::vector<size_t> delete_dim_idx;
    // for (const auto& [input_dim_idx, output_dim_idx] : related_dims) {
    //   used_dim_mask[output_dim_idx] = 1;
    // }
    // for (size_t used_dim : used_dim_mask) {
    //   if (used_dim == 0) {
    //     delete_dim_idx.emplace_back();
    //   }
    // }
    // return std::make_shared<DeleteDimTransform>(info, delete_dim_idx);
    return CreateDefaultAnchorTransform(info);
  } else {
    return CreateDefaultAnchorTransform(info);
  }
}

std::optional<AnchorTransform> CreateAnchorTransformForSpecialOps(
    const TransformInfo& info) {
  if (info.op->num_results() != 1) {
    VLOG(4) << "Now we do not support op with multi outputs, create default: "
            << info.op->name();
    return CreateDefaultAnchorTransform(info);
  }
  if (info.op->isa<cinn::dialect::ReshapeOp>()) {
    return CreateDefaultAnchorTransform(info);
  }
  if (info.op->name() == "cinn_op.generate_shape") {
    return CreateDefaultAnchorTransform(info);
  }
  if (info.op->name() == "cinn_op.reshape") {
    return CreateDefaultAnchorTransform(info);
  }
  if (info.op->name() == "pd_op.reshape") {
    return CreateDefaultAnchorTransform(info);
  }
  // these op has no OpPatternKind
  if (info.op->name() == "pd_op.if" || info.op->name() == "pd_op.embedding") {
    return CreateDefaultAnchorTransform(info);
  }
  return std::nullopt;
}

AnchorTransform CreateAnchorTransform(const TransformInfo& info) {
  auto special_result = CreateAnchorTransformForSpecialOps(info);
  if (special_result != std::nullopt) {
    return special_result.value();
  }

  const hlir::framework::OpPatternKind kind = GetOpPatternKind(info.op);
  AnchorTransform result;
  if (kind == hlir::framework::kReduction) {
    result = CreateAnchorTransformForReduce(info);
  } else if (kind == hlir::framework::kElementWise) {
    result = CreateAnchorTransformForElementWise(info);
  } else if (kind == hlir::framework::kBroadcast) {
    result = CreateAnchorTransformForBroadcast(info);
  } else {
    result = CreateDefaultAnchorTransform(info);
  }

  return result;
}

std::vector<AnchorTransform> PossibleTransform(
    pir::Value v, const std::unordered_set<pir::Operation*>& ops) {
  std::vector<AnchorTransform> result;

  // Transform to Upstream
  auto defining_op = v.defining_op();
  if (defining_op != nullptr && ops.find(defining_op) != ops.end()) {
    size_t output_idx = GetResultIdx(v, defining_op);
    for (size_t i = 0; i < defining_op->num_operands(); ++i) {
      result.emplace_back(CreateAnchorTransform(
          TransformInfo(defining_op, i, output_idx, true)));
    }
  }

  // Transform to Downstream
  for (auto consumer_it = v.use_begin(); consumer_it != v.use_end();
       ++consumer_it) {
    auto downstream_op = consumer_it->owner();
    if (ops.find(downstream_op) == ops.end()) {
      continue;
    }
    size_t input_idx = GetOperandIdx(v, downstream_op);
    for (size_t i = 0; i < downstream_op->num_results(); i++) {
      result.emplace_back(CreateAnchorTransform(
          TransformInfo(downstream_op, input_idx, i, false)));
    }
  }

  VLOG(4) << "PossibleTransform Result:";
  for (const auto& trans : result) {
    VLOG(4) << DebugStrOfAnchorTransform(trans);
  }

  return result;
}

TransformInfo GetTransformInfo(AnchorTransform trans) {
  return std::visit([](auto&& arg) { return arg->info; }, trans);
}

std::string DebugStrOfAnchorTransform(const AnchorTransform& trans) {
  return std::visit([](auto&& arg) { return arg->DebugStr(); }, trans);
}

std::string DebugStrOfAnchorTransformRoute(const AnchorTransformRoute& route) {
  std::stringstream ss;
  for (const auto& trans : route) {
    ss << DebugStrOfAnchorTransform(trans) << "\n";
  }
  return ss.str();
}

}  // namespace cinn::fusion

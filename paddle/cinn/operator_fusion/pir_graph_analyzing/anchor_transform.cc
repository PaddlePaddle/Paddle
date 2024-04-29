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

namespace cinn::fusion {

AnchorTransform CreateDefaultAnchorTransform(const TransformInfo& info) {
  return UnsupportTransform(info);
}

AnchorTransform CreateAnchorTransformForElementWise(const TransformInfo& info) {
  return IdentityTransform(info);
}

AnchorTransform CreateAnchorTransformForReduce(const TransformInfo& info) {
  if (info.op->num_operands() != 1 || info.input_idx != 0) {
    return CreateDefaultAnchorTransform(info);
  }

  if (info.is_upstream_anchor) {
    return AppendDimTransform(info,
                              GetReduceAxisIdx(pir::Operation * reduce_op))
  } else {
    return CreateDefaultAnchorTransform(info);
  }
}

AnchorTransform CreateAnchorTransformForBroadcast(const TransformInfo& info) {
  if (info.op->num_operands() != 1 || info.input_idx != 0) {
    return CreateDefaultAnchorTransform(info);
  }

  if (info.is_upstream_anchor) {
    auto related_dims = GetNonBroadCastDims(info.op);
    size_t output_rank = GetRank(info.OutputValue());
    std::vector<size_t> used_dim_mask = std::vector<size_t>(output_rank, 0);
    std::vector<size_t> delete_dim_idx;
    for (const auto& [input_dim_idx, output_dim_idx] : related_dims) {
      used_dim_mask[output_dim_idx] = 1;
    }
    for (size_t used_dim : used_dim_mask) {
      if (used_dim == 0) {
        delete_dim_idx.emplace_back();
      }
    }
    return DeleteDimTransform(info, delete_dim_idx);
  } else {
    return CreateDefaultAnchorTransform(info);
  }
}

AnchorTransform CreateAnchorTransformForSpecialOps(const TransformInfo& info) {
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
  if (info.op->name() == "cinn_op.yield_store") {
    return CreateDefaultAnchorTransform(info);
  }
  if (info.op->name() == "cinn_op.reshape") {
    return CreateDefaultAnchorTransform(info);
  }
  if (info.op->name() == "pd_op.reshape") {
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

}  // namespace cinn::fusion

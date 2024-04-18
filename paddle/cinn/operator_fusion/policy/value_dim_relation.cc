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

#include "paddle/cinn/operator_fusion/policy/value_dim_relation.h"

namespace cinn::fusion {

const size_t GetUsageIdx(const pir::Value& v, pir::Operation* op) {
  size_t i = 0;
  for (auto consumer_it = v.use_begin(); consumer_it != v.use_end();
       ++consumer_it, ++i) {
    if (consumer_it->owner() == op) {
      return i;
    }
  }
  PADDLE_THROW(phi::errors::NotFound(
      "Can not find the usage of value %s in op %s", v.impl(), op->name()));
}

std::vector<ValueDim> GetAllValueDimFromValue(const pir::Value& v,
                                              const size_t usage_idx) {
  std::vector<ValueDim> value_dims;
  size_t rank = GetRank(v);
  for (size_t i = 0; i < rank; ++i) {
    value_dims.emplace_back(v, i, usage_idx);
  }
  return value_dims;
}

static std::vector<ValueDim> GetAllInputValueDim(pir::Operation* op) {
  std::vector<ValueDim> value_dims;
  for (const auto& v : op->operands()) {
    value_dims = ConcatVector(
        value_dims,
        GetAllValueDimFromValue(v.source(), GetUsageIdx(v.source(), op)));
  }
  return value_dims;
}

static std::vector<std::vector<ValueDim>> GetAllOutputValueDim(
    pir::Operation* op) {
  std::vector<std::vector<ValueDim>> value_dims;
  for (const auto& v : op->results()) {
    std::vector<ValueDim> one_value_dims;
    for (size_t i = 0; i < v.use_count(); i++) {
      one_value_dims =
          ConcatVector(one_value_dims, GetAllValueDimFromValue(v, i));
    }
    value_dims.emplace_back(one_value_dims);
  }
  return value_dims;
}

static ValueDimRelation CreateOpRelativenessForElementWise(pir::Operation* op) {
  ValueDimRelation res;
  for (const auto& v : op->operands()) {
    const auto& value_dims =
        GetAllValueDimFromValue(v.source(), GetUsageIdx(v.source(), op));
    const auto& out_value_dims = GetAllOutputValueDim(op);
    CHECK_EQ(value_dims.size(), out_value_dims.size());
    for (size_t i = 0; i < value_dims.size(); ++i) {
      for (size_t j = 0; j < out_value_dims[i].size(); ++j) {
        res[value_dims[i]][out_value_dims[i][j]] = true;
      }
    }
  }
  return res;
}

static std::vector<std::pair<size_t, size_t>> GetNonBroadCastDims(
    pir::Operation* op) {
  std::vector<std::pair<size_t, size_t>> res;
  const auto* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

  const auto& broad_cast_value = GetBroadcastOpInputOuputValue(op);
  CHECK(broad_cast_value.has_value());

  const auto& [input_value, output_value] = broad_cast_value.value();
  const int input_rank = GetRank(input_value);
  const int output_rank = GetRank(output_value);
  CHECK_GE(output_rank, input_rank);

  // Compare axis one by one, from back to front.
  // The rule of broadcasting:
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/tensor_cn.html#id7
  for (int i = 1; i <= input_rank; ++i) {
    int input_axis = input_rank - i;
    int output_axis = output_rank - i;
    if (input_axis < 0 || output_axis < 0) break;
    if (shape_analysis->IsProductEqual(
            input_value, {input_axis}, output_value, {output_axis})) {
      res.emplace_back(input_axis, output_axis);
    }
  }

  return res;
}

static ValueDimRelation CreateOpRelativenessForBroadcast(pir::Operation* op) {
  ValueDimRelation res;
  const auto& in_value = op->operand(0).source();
  const auto& out_value = op->result(0);
  for (const auto& t : GetNonBroadCastDims(op)) {
    for (size_t i = 0; i < out_value.use_count(); i++) {
      res[ValueDim(in_value, t.first, GetUsageIdx(in_value, op))]
         [ValueDim(out_value, t.second, i)] = true;
    }
  }
  return res;
}

static ValueDimRelation CreateOpRelativenessForDefault(pir::Operation* op) {
  ValueDimRelation res;

  for (const auto& in_dim : GetAllInputValueDim(op)) {
    for (const auto& out_dims : GetAllOutputValueDim(op)) {
      for (const auto& out_dim : out_dims) {
        res[in_dim][out_dim] = true;
      }
    }
  }
  return res;
}

static ValueDimRelation CreateOpRelativenessForReduce(pir::Operation* op) {
  const auto& reduce_axis_idx = GetReduceAxisIdx(op);
  ValueDimRelation res;
  const size_t input_rank = GetRank(op->operand_source(0));
  int out_idx = 0;
  bool keep_dim = GetReduceOpKeepDims(op);
  for (int i = 0; i < input_rank; i++) {
    if (std::find(reduce_axis_idx.begin(), reduce_axis_idx.end(), i) !=
        reduce_axis_idx.end()) {
      for (size_t j = 0; j < op->result(0).use_count(); ++j) {
        res[ValueDim(
            op->operand_source(0), i, GetUsageIdx(op->operand_source(0), op))]
           [ValueDim(op->result(0), out_idx, j)] = true;
      }
      out_idx += 1;
    } else {
      out_idx += keep_dim;
    }
  }
  return res;
}

static std::optional<ValueDimRelation> CreateOpRelativenessForSpecialOps(
    pir::Operation* op) {
  if (op->num_results() != 1) {
    VLOG(4) << "Now we do not support op with multi outputs, use default: "
            << op->name();
    return CreateOpRelativenessForDefault(op);
  }
  if (op->name() == "cinn_op.reshape") {
    // Special Elementwise.
    return CreateOpRelativenessForDefault(op);
  }
  if (op->name() == "pd_op.reshape") {
    // Special Elementwise.
    return CreateOpRelativenessForDefault(op);
  }
  if (op->name() == "cinn_op.generate_shape") {
    return CreateOpRelativenessForDefault(op);
  }
  if (op->name() == "cinn_op.yield_store") {
    return CreateOpRelativenessForDefault(op);
  }
  return {};
}

static ValueDimRelation GetSingleOpRelation(pir::Operation* op) {
  VLOG(4) << "GetSingleOpRelation for " << op->name();
  const auto& special_result = CreateOpRelativenessForSpecialOps(op);
  if (special_result != std::nullopt) {
    return special_result.value();
  }

  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  ValueDimRelation result;
  if (kind == hlir::framework::kReduction) {
    result = CreateOpRelativenessForReduce(op);
  } else if (kind == hlir::framework::kElementWise) {
    result = CreateOpRelativenessForElementWise(op);
  } else if (kind == hlir::framework::kBroadcast) {
    result = CreateOpRelativenessForBroadcast(op);
  } else {
    result = CreateOpRelativenessForDefault(op);
  }
  return result;
}

static std::vector<std::pair<ValueDim, ValueDim>> FlattenRelation(
    const ValueDimRelation& axes_relation) {
  std::vector<std::pair<ValueDim, ValueDim>> res;
  for (const auto& in_dim_pair : axes_relation) {
    for (const auto& out_dim_pair : in_dim_pair.second) {
      res.emplace_back(in_dim_pair.first, out_dim_pair.first);
    }
  }
  return res;
}

ValueDimRelation AnalysisIndexExprRelation(
    const std::vector<pir::Operation*>& ops) {
  ValueDimRelation res;

  for (size_t i = ops.size(); i >= 1; --i) {
    pir::Operation* op = ops[i - 1];
    if (op->name() == "cf.yield") continue;

    const auto& value_dim_relation = GetSingleOpRelation(op);
    for (const auto& in_out_pair : FlattenRelation(value_dim_relation)) {
      for (const auto& out_relation : res[in_out_pair.second]) {
        res[in_out_pair.first][out_relation.first] = true;
      }
      res[in_out_pair.first][in_out_pair.second] = true;
    }
  }
  return res;
}

}  // namespace cinn::fusion

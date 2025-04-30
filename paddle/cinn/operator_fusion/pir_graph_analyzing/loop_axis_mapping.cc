// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/loop_axis_mapping.h"
#include <deque>
namespace cinn::fusion {

std::ostream& operator<<(std::ostream& os, const AxisTransform& transform) {
  os << std::visit([](auto&& t) { return t->DebugStr(); }, transform);
  return os;
}
std::ostream& operator<<(std::ostream& os, const AxisTransformRoute& route) {
  os << cinn::utils::Join(route, " -> ");
  return os;
}

AxisTransform AppendAxisTransform::reverse() {
  return std::make_shared<DeleteAxisTransform>(axis, shape);
}
AxisTransform DeleteAxisTransform::reverse() {
  return std::make_shared<AppendAxisTransform>(axis, shape);
}

AxisTransform ReverseTransform(const AxisTransform& transform) {
  return std::visit([](auto&& t) { return t->reverse(); }, transform);
}

AxisTransformRoute ReverseTransformRoute(const AxisTransformRoute& route) {
  AxisTransformRoute result;
  for (auto it = route.rbegin(); it != route.rend(); ++it) {
    result.push_back(ReverseTransform(*it));
  }
  return result;
}

std::string LoopAxisMapping::DebugStr() const {
  std::stringstream ss;
  for (size_t i = 0; i < input_values.size(); ++i) {
    ss << "\n input " << i << " :\t["
       << cinn::utils::Join(GetCompatibleValueAllDims(input_values[i]), ", ")
       << "], " << input_values[i].impl();
  }
  ss << "\n  loop   :\t[" << cinn::utils::Join(loop, ", ")
     << "], reduce_axis_num: " << reduce_axis_num;
  for (size_t i = 0; i < output_values.size(); ++i) {
    ss << "\noutput " << i << " :\t["
       << cinn::utils::Join(GetCompatibleValueAllDims(output_values[i]), ", ")
       << "], " << output_values[i].impl()
       << ", use_count: " << outputs_use_count.at(output_values[i]);
  }
  for (size_t i = 0; i < input2loop.size(); ++i) {
    ss << "\ninput2loop  " << i << " : " << input2loop[i];
  }
  for (size_t i = 0; i < loop2output.size(); ++i) {
    ss << "\nloop2output " << i << " : " << loop2output[i];
  }
  return ss.str();
}

void LoopAxisMapping::SetReverseMapping() {
  loop2input.clear();
  output2loop.clear();
  for (const auto& route : input2loop) {
    PADDLE_ENFORCE(
        !route.empty(),
        ::common::errors::InvalidArgument("input2loop must not be empty."));
    loop2input.push_back(ReverseTransformRoute(route));
  }
  for (const auto& route : loop2output) {
    PADDLE_ENFORCE(
        !route.empty(),
        ::common::errors::InvalidArgument("loop2output must not be empty."));
    output2loop.push_back(ReverseTransformRoute(route));
  }
}

void LoopAxisMapping::DisableLoopAxisMapping() {
  for (int i = 0; i < input_values.size(); ++i) {
    input2loop[i].clear();
    input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  for (int i = 0; i < output_values.size(); ++i) {
    loop2output[i].clear();
    loop2output[i].push_back(UnsupportedTransform::InstancePtr());
  }
  loop.clear();
  reduce_axis_num = 0;
  SetReverseMapping();
}

AxisTransformSimulator::AxisTransformSimulator(
    const AxisTransformRoute& route,
    const std::vector<symbol::DimExpr>& inshape)
    : route_(route) {
  for (size_t i = 0; i < inshape.size(); ++i) {
    source_ids_.push_back(UniqueAxisId());
    axis_symbols_[source_ids_[i]] = inshape[i];
  }
  target_ids_ = source_ids_;
  Simulate();
  for (const auto& axis_id : target_ids_) {
    out_shape_.push_back(axis_symbols_.at(axis_id));
  }
}

std::set<std::string> AxisTransformSimulator::GetRelatedAxisIds(
    const std::vector<std::string>& ids) {
  std::deque<std::string> queue(ids.begin(), ids.end());
  std::set<std::string> related_ids;
  while (!queue.empty()) {
    auto cur_id = queue.front();
    queue.pop_front();
    if (related_ids.count(cur_id)) continue;
    related_ids.insert(cur_id);
    if (axis_relation_map_.count(cur_id)) {
      for (const auto& id : axis_relation_map_.at(cur_id)) {
        queue.push_back(id);
      }
    }
  }
  return related_ids;
}

void AxisTransformSimulator::Simulate() {
  auto simulate_transform = adt::match{
      [&](const IdentityTransformPtr&) {},
      [&](const TransposeTransformPtr& transform) {
        target_ids_ = TransposeVector(target_ids_, transform->perm);
      },
      [&](const AppendAxisTransformPtr& transform) {
        for (int i = 0; i < transform->axis.size(); ++i) {
          auto new_id = UniqueAxisId();
          target_ids_.insert(target_ids_.begin() + transform->axis[i], new_id);
          axis_symbols_[new_id] = transform->shape[i];
        }
      },
      [&](const DeleteAxisTransformPtr& transform) {
        for (int i = transform->axis.size() - 1; i >= 0; --i) {
          auto id_to_delete = target_ids_[transform->axis[i]];
          target_ids_.erase(target_ids_.begin() + transform->axis[i]);
          axis_symbols_[id_to_delete] = transform->shape[i];
        }
      },
      [&](const ReshapeTransformPtr& transform) {
        const auto& in_shape = transform->in_shape;
        const auto& out_shape = transform->out_shape;
        const auto& partition_indices =
            PartitionReshapeAxes(in_shape, out_shape);
        std::vector<std::string> new_ids;
        for (int idx = 1; idx < partition_indices.size(); ++idx) {
          const auto& [in_start, out_start] = partition_indices[idx - 1];
          const auto& [in_end, out_end] = partition_indices[idx];
          if (in_end == in_start + 1 && out_end == out_start + 1) {
            new_ids.push_back(target_ids_[in_start]);
          } else {
            for (int i = out_start; i < out_end; ++i) {
              if (out_shape[i] == symbol::DimExpr(1)) {
                new_ids.push_back(UniqueAxisId());
                axis_symbols_[new_ids.back()] = symbol::DimExpr(1);
              } else {
                std::string axis_id;
                for (int j = in_start; j < in_end; ++j) {
                  if (in_shape[j] == symbol::DimExpr(1)) {
                    continue;
                  } else if (in_shape[j] == out_shape[i]) {
                    axis_id = target_ids_[j];
                    break;
                  } else {
                    if (axis_id.empty()) axis_id = UniqueAxisId();
                    axis_relation_map_[target_ids_[j]].insert(axis_id);
                  }
                }
                new_ids.push_back(axis_id);
                if (!axis_symbols_.count(axis_id)) {
                  axis_symbols_[axis_id] = out_shape[i];
                }
              }
            }
          }
        }
        for (int i = in_shape.size(); i < target_ids_.size(); ++i) {
          new_ids.push_back(target_ids_[i]);
        }
        target_ids_ = new_ids;
      },
      [&](const auto& trans) {
        PADDLE_THROW(::common::errors::Unimplemented("Unsupported transform."));
      },
  };
  for (const auto& trans : route_) {
    std::visit(simulate_transform, trans);
  }
}

std::pair<AxisTransformRoute, std::vector<symbol::DimExpr>>
SimplifySimpleTransform(const AxisTransformRoute& route,
                        const std::vector<symbol::DimExpr>& inshape) {
  // 1. Simulate transform route
  AxisTransformSimulator simulator(route, inshape);
  if (route.size() <= 1) return {route, simulator.out_shape_};
  // 2. Get Simlplified transform route
  AxisTransformRoute result;
  auto& source_ids = simulator.source_ids_;
  auto& target_ids = simulator.target_ids_;
  auto& axis_symbols = simulator.axis_symbols_;
  if (source_ids == target_ids) {
    result.push_back(IdentityTransform::InstancePtr());
  } else {
    auto [source_unique_ids, source_unique_pos] =
        GatherFirstNotInSecond(source_ids, target_ids);
    auto [target_unique_ids, target_unique_pos] =
        GatherFirstNotInSecond(target_ids, source_ids);
    auto medium_ids = source_ids;
    if (!source_unique_ids.empty()) {
      auto delete_symbols = GatherMapValue(axis_symbols, source_unique_ids);
      result.push_back(std::make_shared<DeleteAxisTransform>(
          CastVector<int32_t, int64_t>(source_unique_pos), delete_symbols));
      medium_ids = GatherVectorExcept(medium_ids, source_unique_pos);
    }
    if (!target_unique_ids.empty()) {
      auto append_symbols = GatherMapValue(axis_symbols, target_unique_ids);
      result.push_back(std::make_shared<AppendAxisTransform>(
          CastVector<int32_t, int64_t>(target_unique_pos), append_symbols));
      for (const auto& pos : target_unique_pos) {
        medium_ids.insert(medium_ids.begin() + pos, target_ids[pos]);
      }
    }
    if (medium_ids != target_ids) {
      auto perm = GetTransposePerm<int32_t>(medium_ids, target_ids);
      result.push_back(std::make_shared<TransposeTransform>(perm));
    }
  }
  return {result, simulator.out_shape_};
}

AxisTransformRoute SimplifyContinuousReshape(const AxisTransformRoute& route) {
  if (route.size() <= 1) return route;
  const auto simplify_reshape =
      [](const AxisTransformRoute& route) -> AxisTransformRoute {
    if (route.size() <= 1) return route;
    auto in_shape = std::get<ReshapeTransformPtr>(route.front())->in_shape;
    auto out_shape = std::get<ReshapeTransformPtr>(route.back())->out_shape;
    AxisTransformRoute result;
    if (in_shape == out_shape) {
      result.push_back(IdentityTransform::InstancePtr());
    } else {
      result.push_back(std::make_shared<ReshapeTransform>(in_shape, out_shape));
    }
    return result;
  };
  AxisTransformRoute result;
  AxisTransformRoute continuous_reshape;
  for (const auto& trans : route) {
    if (std::holds_alternative<UnsupportedTransformPtr>(trans)) {
      return {trans};
    } else if (std::holds_alternative<IdentityTransformPtr>(trans)) {
      // Do nothing.
    } else if (std::holds_alternative<ReshapeTransformPtr>(trans)) {
      continuous_reshape.push_back(std::get<ReshapeTransformPtr>(trans));
    } else {
      if (!continuous_reshape.empty()) {
        result = ConcatVector(result, simplify_reshape(continuous_reshape));
        continuous_reshape.clear();
      }
      result.push_back(trans);
    }
  }
  if (!continuous_reshape.empty()) {
    result = ConcatVector(result, simplify_reshape(continuous_reshape));
  }
  if (result.empty()) result.push_back(IdentityTransform::InstancePtr());
  return result;
}

AxisTransformRoute SimplifyTransformRoute(
    const AxisTransformRoute& route,
    const std::vector<symbol::DimExpr>& input_shape) {
  AxisTransformRoute reshape_simplified = SimplifyContinuousReshape(route);
  if (reshape_simplified.size() <= 1) return reshape_simplified;
  // Simplify continuous non-reshape route.
  AxisTransformRoute result;
  AxisTransformRoute part;
  auto inshape = input_shape;
  for (const auto& trans : reshape_simplified) {
    if (std::holds_alternative<UnsupportedTransformPtr>(trans)) {
      return {trans};
    } else if (std::holds_alternative<IdentityTransformPtr>(trans)) {
      // Do nothing.
    } else if (auto reshape_trans = std::get_if<ReshapeTransformPtr>(&trans)) {
      if (!part.empty()) {
        const auto& simplified_part = SimplifySimpleTransform(part, inshape);
        result = ConcatVector(result, simplified_part.first);
        inshape = simplified_part.second;
        part.clear();
      }
      result.push_back(trans);
      // Reshape transform only change the first dims in some cases.
      auto next_shape = (*reshape_trans)->out_shape;
      for (int i = (*reshape_trans)->in_shape.size(); i < inshape.size(); ++i) {
        next_shape.push_back(inshape[i]);
      }
      inshape = next_shape;
    } else {
      part.push_back(trans);
    }
  }
  result = ConcatVector(result, SimplifySimpleTransform(part, inshape).first);
  if (result.empty()) result.push_back(IdentityTransform::InstancePtr());
  return result;
}

void LoopAxisMapping::SimplifyForwardMapping() {
  for (int i = 0; i < input_values.size(); ++i) {
    input2loop[i] = SimplifyTransformRoute(
        input2loop[i], GetCompatibleValueAllDims(input_values[i]));
  }
  for (int i = 0; i < output_values.size(); ++i) {
    loop2output[i] = SimplifyTransformRoute(loop2output[i], loop);
  }
}

LoopAxisMapping CreateDefaultLoopAxisMapping(pir::Operation* op) {
  LoopAxisMapping result;
  result.input2loop.resize(op->num_operands());
  result.loop2output.resize(op->num_results());
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
    result.input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.output_values.push_back(op->result(i));
    result.loop2output[i].push_back(UnsupportedTransform::InstancePtr());
  }
  return result;
}

LoopAxisMapping CreateDefaultLoopAxisMappingForTrivialOp(pir::Operation* op) {
  auto result = CreateDefaultLoopAxisMapping(op);
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
  result.loop2output[0].clear();
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopAxisMappingForElementwise(pir::Operation* op) {
  LoopAxisMapping result;
  result.input2loop.resize(op->num_operands());
  result.loop2output.resize(op->num_results());
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
    result.input2loop[i].push_back(IdentityTransform::InstancePtr());
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.output_values.push_back(op->result(i));
    result.loop2output[i].push_back(IdentityTransform::InstancePtr());
  }
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
  return result;
}

LoopAxisMapping CreateLoopAxisMappingForTranspose(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of transpose_op shall be equal 1."));
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);
  std::vector<int32_t> perm =
      GetInt32ArrayAttributeData(op->attributes().at("perm"));
  result.input2loop[0].push_back(std::make_shared<TransposeTransform>(perm));
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopAxisMappingForSlice(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of slice_op shall be equal 1."));
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);

  std::vector<int64_t> axes =
      GetInt64ArrayAttributeData(op->attributes().at("axes"));
  std::vector<int64_t> decrease_axis =
      GetInt64ArrayAttributeData(op->attributes().at("decrease_axis"));
  std::vector<int64_t> starts =
      GetInt64ArrayAttributeData(op->attributes().at("starts"));
  std::vector<int64_t> ends =
      GetInt64ArrayAttributeData(op->attributes().at("ends"));
  auto decrease_axis_set = ToUnorderedSet(decrease_axis);
  auto input_shape = GetValueAllDims(op->operand_source(0));
  for (int i = axes.size() - 1; i >= 0; --i) {
    auto start = starts[i] < 0 ? starts[i] + input_shape.size() : starts[i];
    auto end = ends[i] < 0 ? ends[i] + input_shape.size() : ends[i];
    end = end > input_shape.size() ? input_shape.size() : end;
    if (start > 0) {
      // TODO(huangjiyi): Support slice axis start > 0.
      result.input2loop[0].push_back(UnsupportedTransform::InstancePtr());
      break;
    }
    int64_t slice_size = ends[i] - starts[i];
    if (!decrease_axis_set.count(axes[i]) && slice_size != 1) {
      // TODO(huangjiyi): Support slice size greater than 1.
      result.input2loop[0].push_back(UnsupportedTransform::InstancePtr());
      break;
    }
    std::vector<int64_t> axis = {axes[i]};
    result.input2loop[0].push_back(std::make_shared<DeleteAxisTransform>(
        axis, GatherVector(input_shape, axis)));
    if (!decrease_axis_set.count(axes[i])) {
      result.input2loop[0].push_back(
          std::make_shared<AppendAxisTransform>(axis));
    }
  }
  if (GetRank(result.output_values[0]) == 0) {
    result.input2loop[0].push_back(
        std::make_shared<AppendAxisTransform>(std::vector<int64_t>{0}));
  }
  result.input2loop[0] =
      SimplifyTransformRoute(result.input2loop[0], input_shape);
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopAxisMappingForBroadcast(pir::Operation* op) {
  LoopAxisMapping result;
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
  }
  result.input2loop.resize(op->num_operands());
  for (int i = 1; i < op->num_operands(); ++i) {
    result.input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  result.output_values.push_back(op->result(0));
  result.loop2output.resize(1);
  result.loop = GetValueAllDims(result.output_values[0]);

  const auto& broad_cast_value = GetBroadcastOpInputOutputValue(op);
  PADDLE_ENFORCE(broad_cast_value.has_value(),
                 ::common::errors::InvalidArgument(
                     "Required broad_cast_value is not empty."));
  const auto& [input_value, output_value] = broad_cast_value.value();
  const auto& in_shape = GetCompatibleValueAllDims(input_value);
  const auto& out_shape = GetCompatibleValueAllDims(output_value);
  std::vector<int64_t> broadcast_axes;
  std::vector<int64_t> input_keepdims;
  int i = in_shape.size() - 1, j = out_shape.size() - 1;
  while (i >= 0 && j >= 0) {
    if (in_shape[i] == out_shape[j]) {
      --i;
      --j;
      continue;
    } else if (in_shape[i] == symbol::DimExpr(1)) {
      input_keepdims.insert(input_keepdims.begin(), i--);
      broadcast_axes.insert(broadcast_axes.begin(), j--);
    } else {
      broadcast_axes.insert(broadcast_axes.begin(), j--);
    }
  }
  // each axis in input shape must be 1 or equal to output shape
  if (i >= 0) {
    result.input2loop[0].push_back(UnsupportedTransform::InstancePtr());
  } else {
    while (j >= 0) {
      broadcast_axes.insert(broadcast_axes.begin(), j--);
    }
    if (!input_keepdims.empty()) {
      result.input2loop[0].push_back(std::make_shared<DeleteAxisTransform>(
          input_keepdims, GatherVector(in_shape, input_keepdims)));
    }
    result.input2loop[0].push_back(std::make_shared<AppendAxisTransform>(
        broadcast_axes, GatherVector(out_shape, broadcast_axes)));
  }
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateLoopAxisMappingForReduce(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of reduce_op shall be equal 1."));
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.loop = GetCompatibleValueAllDims(result.input_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);
  const auto& reduce_axis = GetReduceAxisIdx(op);
  result.reduce_axis_num = reduce_axis.size();
  bool keep_dim = GetReduceOpKeepDims(op);
  auto rank = result.loop.size();
  // Input2Loop: Transpose reduce axis to the last dimension if necessary.
  bool need_transpose = false;
  for (int i = reduce_axis.size() - 1, last = rank - 1; i >= 0;) {
    if (reduce_axis[i--] != last--) {
      need_transpose = true;
    }
  }
  if (need_transpose) {
    std::vector<int32_t> perm =
        GatherVectorExcept(ArangeVector<int32_t>(0, rank), reduce_axis);
    for (const auto& axis : reduce_axis) {
      perm.push_back(axis);
    }
    result.input2loop[0].push_back(std::make_shared<TransposeTransform>(perm));
    result.loop = TransposeVector(result.loop, perm);
  }
  // Input2Loop: Insert axis with size 1 for each reduce axis if keep_dim.
  if (keep_dim) {
    result.input2loop[0].push_back(
        std::make_shared<AppendAxisTransform>(reduce_axis));
    for (const auto& axis : reduce_axis) {
      result.loop.insert(result.loop.begin() + axis, symbol::DimExpr(1));
    }
    rank += reduce_axis.size();
  }
  // Input2Loop: Insert a axis with size 1 when reduce all without keep_dim.
  if (result.loop.size() == reduce_axis.size()) {
    result.input2loop[0].push_back(
        std::make_shared<AppendAxisTransform>(std::vector<int64_t>{0}));
    result.loop.insert(result.loop.begin(), symbol::DimExpr(1));
  }
  if (result.input2loop[0].empty()) {
    result.input2loop[0].push_back(IdentityTransform::InstancePtr());
  }
  // Loop2Output: Delete reduce axis
  const auto& delete_axis = ArangeVector<int64_t>(
      result.loop.size() - reduce_axis.size(), result.loop.size());
  result.loop2output[0].push_back(std::make_shared<DeleteAxisTransform>(
      delete_axis, GatherVector(result.loop, delete_axis)));
  return result;
}

LoopAxisMapping CreateLoopAxisMappingForReshape(pir::Operation* op) {
  LoopAxisMapping result;
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
  }
  result.input2loop.resize(op->num_operands());
  for (int i = 1; i < op->num_operands(); ++i) {
    result.input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  result.output_values.push_back(op->result(0));
  result.loop2output.resize(1);
  auto in_shape = GetCompatibleValueAllDims(op->operand_source(0));
  auto out_shape = GetCompatibleValueAllDims(op->result(0));
  result.loop = out_shape;

  if (!ShapeProductEqual(in_shape, out_shape)) {
    return CreateDefaultLoopAxisMappingForTrivialOp(op);
  }

  auto has_dynamic_shape = [](const std::vector<symbol::DimExpr>& shape) {
    return std::any_of(
        shape.begin(), shape.end(), [](const symbol::DimExpr& sym) {
          return !sym.isa<std::int64_t>();
        });
  };
  // TODO(huangjiyi): Support dynamic shape for reshape anchor fusion
  if (has_dynamic_shape(in_shape) || has_dynamic_shape(out_shape)) {
    return CreateDefaultLoopAxisMappingForTrivialOp(op);
  }

  // If Reshape only appends or deletes dims with size 1,
  // we can use DeleteAxisTransform and AppendAxisTransform.
  bool only_append_or_delete_ones = true;
  std::vector<int64_t> input_unique_axis;
  std::vector<int64_t> output_unique_axis;
  for (int i = 0, j = 0; i < in_shape.size() || j < out_shape.size();) {
    if (j >= out_shape.size()) {
      input_unique_axis.push_back(i++);
    } else if (i >= in_shape.size()) {
      output_unique_axis.push_back(j++);
    } else if (in_shape[i] == out_shape[j]) {
      ++i;
      ++j;
    } else if (in_shape[i] == symbol::DimExpr(1)) {
      input_unique_axis.push_back(i++);
    } else if (out_shape[j] == symbol::DimExpr(1)) {
      output_unique_axis.push_back(j++);
    } else {
      only_append_or_delete_ones = false;
      break;
    }
  }
  if (only_append_or_delete_ones) {
    if (!input_unique_axis.empty()) {
      result.input2loop[0].push_back(std::make_shared<DeleteAxisTransform>(
          input_unique_axis, GatherVector(in_shape, input_unique_axis)));
    }
    if (!output_unique_axis.empty()) {
      result.input2loop[0].push_back(std::make_shared<AppendAxisTransform>(
          output_unique_axis, GatherVector(out_shape, output_unique_axis)));
    }
    if (result.input2loop[0].empty()) {
      result.input2loop[0].push_back(IdentityTransform::InstancePtr());
    }
  } else {
    result.input2loop[0].push_back(
        std::make_shared<ReshapeTransform>(in_shape, out_shape));
  }
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopAxisMapping(pir::Operation* op) {
  auto is_special_trivial = [&](const pir::Operation* op) {
    return op->name() == "cinn_op.concat" || op->name() == "pd_op.gather_nd";
  };
  VLOG(4) << "CreateLoopAxisMapping for op: " << OpsDebugStr({op});
  LoopAxisMapping result;
  auto op_kind = GetOpPatternKind(op);
  if (op->name() == "pd_op.transpose") {
    result = CreateLoopAxisMappingForTranspose(op);
  } else if (op->name() == "cinn_op.reshape" || op->name() == "pd_op.reshape") {
    result = CreateLoopAxisMappingForReshape(op);
  } else if (op->name() == "cinn_op.slice") {
    result = CreateLoopAxisMappingForSlice(op);
  } else if (op->name() == "cinn_op.generate_shape") {
    result = CreateDefaultLoopAxisMapping(op);
  } else if (is_special_trivial(op)) {
    result = CreateDefaultLoopAxisMappingForTrivialOp(op);
  } else if (op_kind == hlir::framework::kBroadcast) {
    result = CreateLoopAxisMappingForBroadcast(op);
  } else if (op_kind == hlir::framework::kReduction) {
    result = CreateLoopAxisMappingForReduce(op);
  } else if (op_kind == hlir::framework::kElementWise) {
    result = CreateLoopAxisMappingForElementwise(op);
  } else {
    result = CreateDefaultLoopAxisMapping(op);
  }
  result.SetReverseMapping();
  for (auto value : result.output_values) {
    result.outputs_use_count[value] = value.use_count();
  }
  VLOG(4) << "LoopAxisMapping Result: " << result.DebugStr();
  return result;
}

}  // namespace cinn::fusion

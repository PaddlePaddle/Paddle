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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/loop_transform_analysis.h"

namespace cinn::fusion {

bool HasUnsupportedTransform(const AxisTransformRoute& route) {
  return std::any_of(route.begin(), route.end(), [](const auto& transform) {
    return std::holds_alternative<UnsupportedTransformPtr>(transform);
  });
}

bool HasReshapeTransform(const AxisTransformRoute& route) {
  return std::any_of(route.begin(), route.end(), [](const auto& transform) {
    return std::holds_alternative<ReshapeTransformPtr>(transform);
  });
}

AxisTransformRoute GetLoopSinkRoute(const LoopAxisMapping& upstream,
                                    const LoopAxisMapping& downstream) {
  AxisTransformRoute result;
  for (size_t i = 0; i < upstream.output_values.size(); ++i) {
    auto value = upstream.output_values[i];
    auto indices = FindPosInVector(downstream.input_values, value);
    for (auto idx : indices) {
      AxisTransformRoute route =
          ConcatVector(upstream.loop2output[i], downstream.input2loop[idx]);
      if (HasUnsupportedTransform(route)) continue;
      if (route.size() < result.size() || result.empty()) result = route;
    }
  }
  if (result.empty()) {
    result.push_back(UnsupportedTransform::InstancePtr());
    return result;
  }
  return SimplifyTransformRoute(result, upstream.loop);
}

AxisTransformRoute GetLoopLiftRoute(const LoopAxisMapping& upstream,
                                    const LoopAxisMapping& downstream) {
  AxisTransformRoute result;
  for (size_t i = 0; i < upstream.output_values.size(); ++i) {
    auto value = upstream.output_values[i];
    auto indices = FindPosInVector(downstream.input_values, value);
    for (auto idx : indices) {
      AxisTransformRoute route =
          ConcatVector(downstream.loop2input[idx], upstream.output2loop[i]);
      if (HasUnsupportedTransform(route)) continue;
      if (route.size() < result.size() || result.empty()) result = route;
    }
  }
  if (result.empty()) {
    result.push_back(UnsupportedTransform::InstancePtr());
    return result;
  }
  return SimplifyTransformRoute(result, downstream.loop);
}

bool HasSharedInput(const LoopAxisMapping& lhs, const LoopAxisMapping& rhs) {
  return AnyFirstInSecond(lhs.input_values, rhs.input_values);
}

std::optional<AxisTransformRoute> GetHorizontalLoopTransform(
    const LoopAxisMapping& source, const LoopAxisMapping& target) {
  // 1. Try to find a route that only appends or deletes axes with size 1
  const auto padding_result = [&]() -> std::optional<AxisTransformRoute> {
    AxisTransformRoute result;
    std::vector<int64_t> source_one_dims;
    std::vector<int64_t> target_one_dims;
    for (int i = 0, j = 0; i < source.loop.size() || j < target.loop.size();) {
      if (j >= target.loop.size()) {
        source_one_dims.push_back(i++);
      } else if (i >= source.loop.size()) {
        target_one_dims.push_back(j++);
      } else if (source.loop[i] == target.loop[j]) {
        ++i;
        ++j;
      } else if (source.loop[i] == symbol::DimExpr(1)) {
        source_one_dims.push_back(i++);
      } else if (target.loop[j] == symbol::DimExpr(1)) {
        target_one_dims.push_back(j++);
      } else {
        // TODO(huangjiyi): Decide whether to support reshape transform
        // in horizontal fusion without shared input
        return std::nullopt;
      }
    }
    if (!source_one_dims.empty()) {
      result.push_back(std::make_shared<DeleteAxisTransform>(
          source_one_dims, GatherVector(source.loop, source_one_dims)));
    }
    if (!target_one_dims.empty()) {
      result.push_back(std::make_shared<AppendAxisTransform>(
          target_one_dims, GatherVector(target.loop, target_one_dims)));
    }
    if (result.empty()) {
      result.push_back(IdentityTransform::InstancePtr());
    }
    return result;
  }();
  if (padding_result.has_value()) return padding_result;
  // 2. Try to inference a route via shared input value
  if (!HasSharedInput(source, target)) return std::nullopt;
  AxisTransformRoute result;
  for (size_t i = 0; i < source.input_values.size(); ++i) {
    auto indices = FindPosInVector(target.input_values, source.input_values[i]);
    for (auto idx : indices) {
      AxisTransformRoute route =
          ConcatVector(source.loop2input[i], target.input2loop[idx]);
      if (HasUnsupportedTransform(route)) continue;
      if (route.size() < result.size() || result.empty()) result = route;
    }
  }
  if (result.empty()) return std::nullopt;
  return SimplifyTransformRoute(result, source.loop);
}

LoopAxisMapping LoopAxisMappingMergeImpl(const LoopAxisMapping& upstream,
                                         const LoopAxisMapping& downstream,
                                         bool upstream_is_anchor) {
  const auto& loop_sink_route = GetLoopSinkRoute(upstream, downstream);
  const auto& loop_lift_route = GetLoopLiftRoute(upstream, downstream);

  LoopAxisMapping result;
  result.input_values = upstream.input_values;
  for (const auto& trans : upstream.input2loop) {
    result.input2loop.push_back(
        upstream_is_anchor ? trans : ConcatVector(trans, loop_sink_route));
  }
  result.outputs_use_count = upstream.outputs_use_count;
  for (size_t i = 0; i < downstream.input_values.size(); ++i) {
    auto value = downstream.input_values[i];
    if (upstream.outputs_use_count.count(value)) {
      result.outputs_use_count[value]--;
      continue;
    }
    result.input_values.push_back(value);
    result.input2loop.push_back(
        upstream_is_anchor
            ? ConcatVector(downstream.input2loop[i], loop_lift_route)
            : downstream.input2loop[i]);
  }
  for (size_t i = 0; i < upstream.output_values.size(); ++i) {
    auto value = upstream.output_values[i];
    if (result.outputs_use_count[value] > 0) {
      result.output_values.push_back(value);
      result.loop2output.push_back(
          upstream_is_anchor
              ? upstream.loop2output[i]
              : ConcatVector(loop_lift_route, upstream.loop2output[i]));
    } else {
      result.outputs_use_count.erase(value);
    }
  }
  for (size_t i = 0; i < downstream.output_values.size(); ++i) {
    auto value = downstream.output_values[i];
    result.output_values.push_back(value);
    result.outputs_use_count[value] = downstream.outputs_use_count.at(value);
    result.loop2output.push_back(
        upstream_is_anchor
            ? ConcatVector(loop_sink_route, downstream.loop2output[i])
            : downstream.loop2output[i]);
  }
  result.loop = upstream_is_anchor ? upstream.loop : downstream.loop;
  result.reduce_axis_num =
      std::max(upstream.reduce_axis_num, downstream.reduce_axis_num);
  return result;
}

LoopAxisMapping LoopAxisMappingMerge(const LoopAxisMapping& upstream,
                                     const LoopAxisMapping& downstream,
                                     bool upstream_is_anchor) {
  auto result =
      LoopAxisMappingMergeImpl(upstream, downstream, upstream_is_anchor);
  result.SimplifyForwardMapping();
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping TrivialSinkLoopAxisMappingMerge(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream) {
  auto result = LoopAxisMappingMergeImpl(upstream, downstream, false);
  auto upstream_out_value = upstream.output_values[0];
  auto indices = FindPosInVector(result.output_values, upstream_out_value);
  if (!indices.empty()) {
    auto idx = indices.front();
    result.output_values.erase(result.output_values.begin() + idx);
    result.loop2output.erase(result.loop2output.begin() + idx);
    result.outputs_use_count.erase(upstream_out_value);
  }
  result.SimplifyForwardMapping();
  result.SetReverseMapping();
  return result;
}

std::vector<int> GetFakeReduceAxisIdx(const std::vector<symbol::DimExpr>& loop,
                                      const AxisTransformRoute& route,
                                      int reduce_axis_num) {
  AxisTransformSimulator simulator(route, loop);
  auto reduce_trivial_related_ids =
      simulator.GetRelatedAxisIds(simulator.source_ids_);
  std::set<std::string> trivial_non_related_ids;
  for (const auto& axis_id : simulator.target_ids_) {
    if (!reduce_trivial_related_ids.count(axis_id)) {
      trivial_non_related_ids.insert(axis_id);
    }
  }
  std::vector<int> fake_reduce_idx;
  for (int i = loop.size() - reduce_axis_num; i < loop.size(); ++i) {
    auto reduce_axis_id = simulator.source_ids_[i];
    auto indices = FindPosInVector(simulator.target_ids_, reduce_axis_id);
    if (!indices.empty()) {
      fake_reduce_idx.push_back(indices.front());
      continue;
    }
    for (const auto& axis_id : trivial_non_related_ids) {
      if (loop[i] == simulator.axis_symbols_.at(axis_id)) {
        fake_reduce_idx.push_back(
            FindPosInVector(simulator.target_ids_, axis_id).front());
        trivial_non_related_ids.erase(axis_id);
        break;
      }
    }
  }
  return fake_reduce_idx;
}

LoopAxisMapping ReducePlusTrivialLoopAxisMappingMerge(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream) {
  // Signal downstream reduce plus trivial fusion loop is downstream trivial
  // loop plus upstream reduce loop.
  PADDLE_ENFORCE(
      upstream.reduce_axis_num > 0 && downstream.reduce_axis_num == 0,
      ::common::errors::InvalidArgument(
          "Upstream should be reduce pattern and "
          "downstream should be trivial pattern."));
  auto loop_sink_route = GetLoopSinkRoute(upstream, downstream);
  if (HasUnsupportedTransform(loop_sink_route)) {
    // TODO(huangjiyi): fix unsupported transform in RT fusion
    auto result = LoopAxisMappingMergeImpl(upstream, downstream, false);
    result.DisableLoopAxisMapping();
    return result;
  }
  auto reduce_axis_num = upstream.reduce_axis_num;
  auto reduce_axis = ArangeVector<int64_t>(
      upstream.loop.size() - reduce_axis_num, upstream.loop.size());
  auto reduce_loop = SliceVector(upstream.loop,
                                 upstream.loop.size() - reduce_axis_num,
                                 upstream.loop.size());
  // Check whether downstream trivial can reuse upstream reduce axis.
  auto fake_reduce_idx =
      GetFakeReduceAxisIdx(upstream.loop, loop_sink_route, reduce_axis_num);
  VLOG(4) << "fake_reduce_idx: " << cinn::utils::Join(fake_reduce_idx, ",");
  LoopAxisMapping result;
  if (fake_reduce_idx.empty()) {
    AxisTransform append_reduce_axis =
        std::make_shared<AppendAxisTransform>(reduce_axis, reduce_loop);
    auto upstream_copy = upstream;
    for (auto& route : upstream_copy.input2loop) {
      route.push_back(append_reduce_axis);
    }
    upstream_copy.loop.insert(
        upstream_copy.loop.end(), reduce_loop.begin(), reduce_loop.end());
    result = LoopAxisMappingMergeImpl(upstream_copy, downstream, false);
    result.loop = ConcatVector(downstream.loop, reduce_loop);
    AxisTransform delete_reduce_axis = std::make_shared<DeleteAxisTransform>(
        ArangeVector<int64_t>(downstream.loop.size(), result.loop.size()),
        reduce_loop);
    for (auto& route : result.loop2output) {
      route.insert(route.begin(), delete_reduce_axis);
    }
    auto fake_reduce_idx = ArangeVector<int64_t>(
        downstream.loop.size(), downstream.loop.size() + reduce_axis_num);
    AxisTransform append_fake_reduce_idx =
        std::make_shared<AppendAxisTransform>(fake_reduce_idx, reduce_loop);
    for (int i = upstream.input2loop.size(); i < result.input2loop.size();
         ++i) {
      result.input2loop[i].push_back(append_fake_reduce_idx);
    }
  } else {
    // Transpose fake reduce axis to the end
    auto perm = ArangeVector<int>(0, downstream.loop.size());
    for (auto index : fake_reduce_idx) {
      perm.push_back(index);
    }
    std::sort(fake_reduce_idx.begin(), fake_reduce_idx.end());
    std::reverse(fake_reduce_idx.begin(), fake_reduce_idx.end());
    for (auto index : fake_reduce_idx) {
      perm.erase(perm.begin() + index);
    }
    result = LoopAxisMappingMergeImpl(upstream, downstream, false);
    AxisTransformRoute fake_reduce_axis_transforms;
    if (perm != ArangeVector<int>(0, downstream.loop.size())) {
      result.loop = TransposeVector(result.loop, perm);
      auto transpose_trans = std::make_shared<TransposeTransform>(perm);
      fake_reduce_axis_transforms.push_back(transpose_trans);
    }
    // Check whether fake reduce axis reuse all reduce axis
    if (fake_reduce_idx.size() < reduce_axis_num) {
      std::vector<int64_t> one_reduce_axis;
      for (int i = 0; i < reduce_loop.size(); ++i) {
        bool has_reuse = false;
        for (const auto& downstream_idx : fake_reduce_idx) {
          if (reduce_loop[i] == downstream.loop[downstream_idx]) {
            has_reuse = true;
            break;
          }
        }
        if (!has_reuse) {
          PADDLE_ENFORCE_EQ(reduce_loop[i],
                            symbol::DimExpr(1),
                            ::common::errors::PreconditionNotMet(
                                "Reduce axis not been reused must be 1."));
          one_reduce_axis.push_back(downstream.loop.size() -
                                    fake_reduce_idx.size() + i);
        }
      }
      auto append_one_reduce_axis =
          std::make_shared<AppendAxisTransform>(one_reduce_axis);
      fake_reduce_axis_transforms.push_back(append_one_reduce_axis);
    }
    for (auto& route : result.input2loop) {
      route.insert(route.end(),
                   fake_reduce_axis_transforms.begin(),
                   fake_reduce_axis_transforms.end());
    }
    for (auto& route : result.loop2output) {
      route.insert(route.begin(),
                   fake_reduce_axis_transforms.begin(),
                   fake_reduce_axis_transforms.end());
    }
  }
  result.SimplifyForwardMapping();
  result.SetReverseMapping();
  return result;
}

bool IsAdjacentRelation(const LoopAxisMapping& lhs,
                        const LoopAxisMapping& rhs) {
  return AnyFirstInSecond(lhs.output_values, rhs.input_values) ||
         AnyFirstInSecond(rhs.output_values, lhs.input_values);
}

LoopAxisMapping HorizontalLoopAxisMappingMerge(const LoopAxisMapping& source,
                                               const LoopAxisMapping& target) {
  PADDLE_ENFORCE(
      !IsAdjacentRelation(source, target),
      ::common::errors::InvalidArgument(
          "Patterns to be merged in horizontal fusion cannot be adjacent."));
  LoopAxisMapping result;
  auto loop_transform = GetHorizontalLoopTransform(source, target);
  PADDLE_ENFORCE(loop_transform.has_value(),
                 ::common::errors::InvalidArgument(
                     "Can not find valid horizontal loop transform."));
  auto reverse_loop_transform = ReverseTransformRoute(loop_transform.value());
  result.input_values = ConcatVector(source.input_values, target.input_values);
  result.output_values =
      ConcatVector(source.output_values, target.output_values);
  for (const auto& transform : source.input2loop) {
    result.input2loop.push_back(
        ConcatVector(transform, loop_transform.value()));
  }
  result.input2loop = ConcatVector(result.input2loop, target.input2loop);
  for (const auto& transform : source.loop2output) {
    result.loop2output.push_back(
        ConcatVector(reverse_loop_transform, transform));
  }
  result.loop2output = ConcatVector(result.loop2output, target.loop2output);
  result.outputs_use_count = source.outputs_use_count;
  for (const auto& [output, use_count] : target.outputs_use_count) {
    result.outputs_use_count[output] = use_count;
  }
  result.loop = target.loop;
  result.reduce_axis_num =
      std::max(source.reduce_axis_num, target.reduce_axis_num);
  result.SimplifyForwardMapping();
  result.SetReverseMapping();
  return result;
}

std::optional<AxisTransformRoute> GetValidLoopTransformRoute(
    const LoopAxisMapping& source,
    const LoopAxisMapping& target,
    const AxisTransformRoute& loop_transform_route) {
  VLOG(4) << "Source loop: [" << cinn::utils::Join(source.loop, ", ")
          << "], reduce_axis_num: " << source.reduce_axis_num;
  VLOG(4) << "Target loop: [" << cinn::utils::Join(target.loop, ", ")
          << "], reduce_axis_num: " << target.reduce_axis_num;
  VLOG(4) << "Loop transform route: " << loop_transform_route;
  if (source.reduce_axis_num > 0 && target.reduce_axis_num == 0) {
    VLOG(4) << "Cannot transform reduce loop to trivial loop.";
    return std::nullopt;
  } else if (source.reduce_axis_num > 0 && target.reduce_axis_num > 0) {
    if (source.reduce_axis_num != target.reduce_axis_num) {
      VLOG(4) << "Cannot transform reduce loop to different reduce axis num.";
      return std::nullopt;
    }
    if (!ShapeProductEqual(source.loop, target.loop)) {
      VLOG(4) << "Cannot apply append axis transform between reduce loop.";
      return std::nullopt;
    }
    auto get_reduce_loop = [](const LoopAxisMapping& mapping) {
      return SliceVector(mapping.loop,
                         mapping.loop.size() - mapping.reduce_axis_num,
                         mapping.loop.size());
    };
    auto source_reduce_loop = get_reduce_loop(source);
    auto target_reduce_loop = get_reduce_loop(target);
    for (size_t i = 0; i < source_reduce_loop.size(); ++i) {
      if (source_reduce_loop[i] != target_reduce_loop[i]) {
        VLOG(4) << "Cannot transform reduce loop to unaligned reduce axis.";
        return std::nullopt;
      }
    }
  }
  bool rr_fusion = source.reduce_axis_num > 0 && target.reduce_axis_num > 0;

  size_t id = 0;
  auto unique_id = [&]() { return "I" + std::to_string(id++); };

  AxisTransformRoute result;
  std::vector<std::string> axis_ids;
  std::unordered_map<std::string, symbol::DimExpr> axis_symbols;
  std::set<std::string> deleted_axes;
  std::unordered_set<std::string> unused_axes;
  std::map<std::string, std::set<std::string>> axis_relation_map;
  for (const auto& symbol : source.loop) {
    auto axis_id = unique_id();
    axis_ids.push_back(axis_id);
    axis_symbols[axis_id] = symbol;
  }
  size_t cur_axis_size = axis_ids.size();
  const auto source_ids = axis_ids;

  if (rr_fusion) {
    // Because reduce axis can not be transformed, we need to add
    // same fake axis to substitute reduce axis for transformation.
    std::vector<int64_t> reduce_axis = ArangeVector<int64_t>(
        cur_axis_size - source.reduce_axis_num, cur_axis_size);
    std::vector<symbol::DimExpr> reduce_shape =
        GatherVector(source.loop, reduce_axis);
    result.push_back(
        std::make_shared<AppendAxisTransform>(reduce_axis, reduce_shape));
  }

  auto apply_transpose = [&](const TransposeTransformPtr& transform) {
    axis_ids = TransposeVector(axis_ids, transform->perm);
    result.push_back(transform);
  };
  auto apply_append_axis = [&](const AppendAxisTransformPtr& transform) {
    for (size_t i = 0; i < transform->axis.size(); ++i) {
      auto axis = transform->axis[i];
      auto symbol = transform->shape[i];
      bool can_reuse = false;
      for (const auto& deleted_axis : deleted_axes) {
        if (axis_symbols.at(deleted_axis) == symbol) {
          // Can reuse deleted axis, move deleted axis to the append position.
          int deleted_axis_pos = FindPosInVector(axis_ids, deleted_axis).back();
          auto new_axis_id = unique_id();
          if (deleted_axis_pos == axis) {
            axis_ids[deleted_axis_pos] = new_axis_id;
          } else {
            auto perm = ArangeVector<int32_t>(0, axis_ids.size());
            perm.erase(perm.begin() + deleted_axis_pos);
            perm.insert(perm.begin() + axis, deleted_axis_pos);
            axis_ids.erase(axis_ids.begin() + deleted_axis_pos);
            axis_ids.insert(axis_ids.begin() + axis, new_axis_id);
            result.push_back(std::make_shared<TransposeTransform>(perm));
          }
          axis_symbols[new_axis_id] = symbol;
          deleted_axes.erase(deleted_axis);
          cur_axis_size++;
          can_reuse = true;
          VLOG(4) << "Reuse axis: " << new_axis_id << " -> " << deleted_axis
                  << ", cur deleted_axes: {"
                  << cinn::utils::Join(SetToVector(deleted_axes), ", ") << "}";
          break;
        }
      }
      // If can not reuse deleted axis, insert new axis and mark it as unused.
      if (!can_reuse) {
        auto axis_id = unique_id();
        axis_ids.insert(axis_ids.begin() + axis, axis_id);
        axis_symbols[axis_id] = symbol;
        unused_axes.insert(axis_id);
        cur_axis_size++;
        result.push_back(std::make_shared<AppendAxisTransform>(
            std::vector<int64_t>{axis}, std::vector<symbol::DimExpr>{symbol}));
        VLOG(4) << "Insert new unused axis: " << axis_id
                << ", cur unused_axes: {"
                << cinn::utils::Join(SetToVector(unused_axes), ", ") << "}";
      }
    }
  };
  auto apply_delete_axis = [&](const DeleteAxisTransformPtr& transform) {
    for (int i = transform->axis.size() - 1; i >= 0; --i) {
      auto axis = transform->axis[i];
      auto axis_id = axis_ids[axis];
      auto symbol = axis_symbols.at(axis_id);
      if (symbol == symbol::DimExpr(1) || unused_axes.count(axis_id)) {
        // Unused axis or axis with size 1 can be deleted directly.
        axis_ids.erase(axis_ids.begin() + axis);
        unused_axes.erase(axis_id);
        result.push_back(std::make_shared<DeleteAxisTransform>(
            std::vector<int64_t>{axis}, std::vector<symbol::DimExpr>{symbol}));
        VLOG(4) << "Delete unused or size 1 axis: " << axis_id
                << ", cur unused_axes: {"
                << cinn::utils::Join(SetToVector(unused_axes), ", ") << "}";
      } else {
        // Used axis can not be deleted directly, we need to transpose it to
        // the end to ensure accuracy of subsequent transform.
        std::vector<std::string> new_axis_ids;
        // No need to transpose if the axis is already at the end.
        if (axis != cur_axis_size - 1) {
          std::vector<int> perm;
          for (int idx = 0; idx < axis_ids.size(); ++idx) {
            if (idx == axis) continue;
            new_axis_ids.push_back(axis_ids[idx]);
            perm.push_back(idx);
          }
          new_axis_ids.push_back(axis_id);
          perm.push_back(axis);
          result.push_back(std::make_shared<TransposeTransform>(perm));
        } else {
          new_axis_ids = axis_ids;
        }
        deleted_axes.insert(axis_id);
        axis_ids = new_axis_ids;
        VLOG(4) << "Pretend to delete axis: " << axis_id
                << ", cur deleted_axes: {"
                << cinn::utils::Join(SetToVector(deleted_axes), ", ") << "}";
      }
      cur_axis_size--;
    }
  };
  auto apply_reshape = [&](const ReshapeTransformPtr& transform) {
    auto in_shape = transform->in_shape;
    auto out_shape = transform->out_shape;
    std::vector<std::string> new_axis_ids;
    if (!ShapeProductEqual(in_shape, out_shape)) {
      for (const auto& symbol : out_shape) {
        auto axis_id = unique_id();
        new_axis_ids.push_back(axis_id);
        axis_symbols[axis_id] = symbol;
      }
      for (const auto& in_axis : axis_ids) {
        for (const auto& out_axis : new_axis_ids) {
          axis_relation_map[in_axis].insert(out_axis);
        }
      }
    } else {
      const auto& partition_indices = PartitionReshapeAxes(in_shape, out_shape);
      for (int idx = 1; idx < partition_indices.size(); ++idx) {
        const auto& [in_start, out_start] = partition_indices[idx - 1];
        const auto& [in_end, out_end] = partition_indices[idx];
        if (in_end == in_start + 1 && out_end == out_start + 1) {
          new_axis_ids.push_back(axis_ids[in_start]);
        } else {
          bool is_unused = true;
          for (int i = in_start; i < in_end; ++i) {
            if (axis_symbols.at(axis_ids[i]) != symbol::DimExpr(1) &&
                !unused_axes.count(axis_ids[i])) {
              is_unused = false;
              break;
            }
          }
          for (int i = out_start; i < out_end; ++i) {
            if (out_shape[i] == symbol::DimExpr(1)) {
              auto axis_id = unique_id();
              new_axis_ids.push_back(axis_id);
              axis_symbols[axis_id] = symbol::DimExpr(1);
            } else {
              std::string axis_id;
              for (int j = in_start; j < in_end; ++j) {
                if (in_shape[j] == symbol::DimExpr(1)) {
                  continue;
                } else if (in_shape[j] == out_shape[i]) {
                  axis_id = axis_ids[j];
                  break;
                } else {
                  if (axis_id.empty()) axis_id = unique_id();
                  axis_relation_map[axis_ids[j]].insert(axis_id);
                }
              }
              new_axis_ids.push_back(axis_id);
              if (!axis_symbols.count(axis_id)) {
                axis_symbols[axis_id] = out_shape[i];
                if (is_unused) unused_axes.insert(axis_id);
              }
            }
          }
        }
      }
    }
    new_axis_ids = ConcatVector(
        new_axis_ids, SliceVector(axis_ids, in_shape.size(), axis_ids.size()));
    axis_ids = new_axis_ids;
    cur_axis_size = cur_axis_size - in_shape.size() + out_shape.size();
    result.push_back(transform);
  };

  auto apply_transform = adt::match{
      [&](const IdentityTransformPtr& trans) {},
      [&](const TransposeTransformPtr& trans) { apply_transpose(trans); },
      [&](const AppendAxisTransformPtr& trans) { apply_append_axis(trans); },
      [&](const DeleteAxisTransformPtr& trans) { apply_delete_axis(trans); },
      [&](const ReshapeTransformPtr& trans) { apply_reshape(trans); },
      [&](const auto& trans) {
        PADDLE_THROW(
            ::common::errors::Unimplemented("Unknown transform type."));
      }};

  auto axis_debug_info = [&]() -> std::string {
    std::vector<symbol::DimExpr> shape;
    for (const auto& id : axis_ids) {
      shape.push_back(axis_symbols.at(id));
    }
    return "Axis ids: [" + cinn::utils::Join(axis_ids, ", ") + "], shape: [" +
           cinn::utils::Join(shape, ", ") +
           "], cur_size: " + std::to_string(cur_axis_size);
  };

  VLOG(4) << "Source axis ids: " << axis_debug_info();
  for (auto& transform : loop_transform_route) {
    if (std::holds_alternative<UnsupportedTransformPtr>(transform)) {
      VLOG(4) << "Can not find valid loop transform because of unsupported "
                 "transform.";
      return std::nullopt;
    } else {
      std::visit(apply_transform, transform);
      VLOG(4) << "After Applying " << transform
              << ", axis ids: " << axis_debug_info();
    }
  }

  if (!deleted_axes.empty()) {
    // Check if all deleted axes are used, otherwise the transform is invalid.
    VLOG(4) << "Can not find valid loop transform because of unreused deleted "
               "axes.";
    return std::nullopt;
  }
  if (rr_fusion) {
    // Check if all reduce axes are reused and there is no relationship
    // between reduce axes and non reduce axes.
    auto [source_trivial_ids, source_reduce_ids] =
        SplitVector(source_ids, source_ids.size() - source.reduce_axis_num);

    auto get_related_ids = [&](const std::vector<std::string>& ids) {
      std::deque<std::string> queue(ids.begin(), ids.end());
      std::set<std::string> related_ids;
      while (!queue.empty()) {
        auto cur_id = queue.front();
        queue.pop_front();
        if (related_ids.count(cur_id)) continue;
        related_ids.insert(cur_id);
        if (axis_relation_map.count(cur_id)) {
          for (const auto& id : axis_relation_map.at(cur_id)) {
            queue.push_back(id);
          }
        }
      }
      return related_ids;
    };
    auto source_reduce_related_ids = get_related_ids(source_reduce_ids);
    auto source_trivial_related_ids = get_related_ids(source_trivial_ids);

    auto [target_trivial_ids, target_reduce_ids] =
        SplitVector(axis_ids, axis_ids.size() - target.reduce_axis_num);

    if (!SetIntersection(source_reduce_related_ids, ToSet(target_trivial_ids))
             .empty() ||
        !SetIntersection(source_trivial_related_ids, ToSet(target_reduce_ids))
             .empty()) {
      VLOG(4) << "Can not find valid loop transform because of relationship "
                 "between reduce axis and non reduce axis.";
      return std::nullopt;
    }
    // Remove fake reduce axes.
    std::vector<int64_t> reduce_axis = ArangeVector<int64_t>(
        axis_ids.size() - target.reduce_axis_num, axis_ids.size());
    std::vector<symbol::DimExpr> reduce_shape =
        GatherVector(target.loop, reduce_axis);
    result.push_back(
        std::make_shared<DeleteAxisTransform>(reduce_axis, reduce_shape));
  }

  if (result.empty()) result.push_back(IdentityTransform::InstancePtr());
  result = SimplifyTransformRoute(result, source.loop);
  VLOG(4) << "Found loop transform: " << result;
  return result;
}

std::optional<AxisTransformRoute> GetValidAdjacentLoopTransform(
    const LoopAxisMapping& upstream,
    const LoopAxisMapping& downstream,
    bool upstream_is_anchor) {
  VLOG(4) << "Try to get valid loop transform route "
          << (upstream_is_anchor ? "from downstream to upstream."
                                 : "from upstream to downstream.");
  auto source = upstream_is_anchor ? downstream : upstream;
  auto target = upstream_is_anchor ? upstream : downstream;

  const auto& loop_transform_route =
      upstream_is_anchor ? GetLoopLiftRoute(upstream, downstream)
                         : GetLoopSinkRoute(upstream, downstream);
  auto result =
      GetValidLoopTransformRoute(source, target, loop_transform_route);

  if (result.has_value() && source.reduce_axis_num == 0 &&
      target.reduce_axis_num > 0) {
    // Check whether reduce trivial fusion with larger reduce dims.
    const auto& reduce_to_trivial_route =
        upstream_is_anchor ? GetLoopSinkRoute(target, source)
                           : GetLoopLiftRoute(source, target);
    auto fake_reduce_idx = GetFakeReduceAxisIdx(
        target.loop, reduce_to_trivial_route, target.reduce_axis_num);
    if (!fake_reduce_idx.empty()) {
      const auto reduce_dims_product =
          GetShapeProduct(target.loop,
                          target.loop.size() - target.reduce_axis_num,
                          target.loop.size());
      if (reduce_dims_product.isa<std::int64_t>() &&
          reduce_dims_product.dyn_cast<std::int64_t>() > 1024 * 8) {
        VLOG(4) << "Can not fuse trivial to reduce with large reduce dims: "
                << reduce_dims_product.dyn_cast<std::int64_t>();
        return std::nullopt;
      }
    }
  }
  return result;
}

std::optional<AxisTransformRoute> GetValidHorizontalLoopTransform(
    const LoopAxisMapping& source, const LoopAxisMapping& target) {
  VLOG(4) << "Try to get valid horizontal loop transform route.";
  auto loop_transform = GetHorizontalLoopTransform(source, target);
  if (loop_transform == std::nullopt) return std::nullopt;
  const auto reduce_dims_product =
      GetShapeProduct(target.loop,
                      target.loop.size() - target.reduce_axis_num,
                      target.loop.size());
  if (source.reduce_axis_num == 0 && target.reduce_axis_num > 0 &&
      !HasSharedInput(source, target)) {
    // Disable horizontal fusion between trivial and reduce without
    // shared inputs when reduce axis num is large.
    if (reduce_dims_product.isa<std::int64_t>() &&
        reduce_dims_product.dyn_cast<std::int64_t>() > 1024) {
      VLOG(4) << "Can not fuse trivial to reduce with large reduce dims: "
              << reduce_dims_product.dyn_cast<std::int64_t>();
      return std::nullopt;
    }
  }
  if (!reduce_dims_product.isa<std::int64_t>()) {
    const auto [shared_inputs, _unused] =
        SplitFirstWhetherInSecond(source.input_values, target.input_values);
    int input_nums = source.input_values.size() + target.input_values.size();
    if (static_cast<float>(shared_inputs.size()) / input_nums < 1. / 6 &&
        input_nums - shared_inputs.size() > 4) {
      // Disable horizontal fusion with dynamic shape when shared input values
      // are less than 1/3 per input while non shared input nums more than 4.
      VLOG(4) << "Can not fuse with dynamic shape when shared inputs are few. ";
      return std::nullopt;
    }
  }
  return GetValidLoopTransformRoute(source, target, loop_transform.value());
}

}  // namespace cinn::fusion

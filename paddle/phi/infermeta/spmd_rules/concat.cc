/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/concat.h"

#include <limits>

#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo ConcatInferSpmd(const std::vector<DistMetaTensor>& x, int axis) {
  /*
# paddle.concat requires all tensors must either have the same shape (except
# in the concatenating dimension) or be "empty". "Empty" here strictly means
# tensor.shape is torch.Size([0]). When tensor.ndim > 1, it will be treated
# as a non-empty tensor and the shape must match on non-cat dimensions.
 */

  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return phi::vectorize<int64_t>(meta.dims());
                 });
  auto is_empty = [](const std::vector<int64_t>& shape) {
    return shape.empty() || shape.at(0) == 0;
  };
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), is_empty);
  if (all_empty) {
    return SpmdInfo();
  }
  auto not_empty = [is_empty](const std::vector<int64_t>& shape) {
    return !is_empty(shape);
  };
  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), not_empty);
  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  int64_t ndim = static_cast<int64_t>(tensor_shapes[non_empty_index].size());
  // normlize dim
  int64_t dim = axis;
  dim = dim < 0 ? dim + ndim : dim;

  std::vector<TensorDistAttr> input_attrs;
  // 2、make sure all tensors replicated on concat dim
  auto n_inputs = x.size();
  for (size_t i = 0; i < n_inputs; ++i) {
    const auto& dist_attr = x[i].dist_attr();
    if (not_empty(tensor_shapes[i]) && IsDimSharded(dist_attr, dim)) {
      auto sharded_dist_attr = ReplicateTensorDim(dist_attr, dim);
      input_attrs.emplace_back(sharded_dist_attr);
    } else {
      input_attrs.emplace_back(dist_attr);
    }
  }
  // 3、align non-concat dimensions according to cost
  std::vector<std::vector<std::shared_ptr<PlacementStatus>>> inputs_placements;
  std::transform(
      input_attrs.begin(),
      input_attrs.end(),
      std::back_inserter(inputs_placements),
      [](const TensorDistAttr& attr) { return attr.to_placement(); });
  const auto& process_mess = input_attrs[non_empty_index].process_mesh();
  auto has_mismatch = [&](int32_t mesh_dim) {
    bool mismatch = false;
    for (size_t i = 0; i < n_inputs; i++) {
      if ((!is_empty(tensor_shapes[i])) &&
          !PlacementEqual(inputs_placements[non_empty_index][mesh_dim],
                          inputs_placements[i][mesh_dim])) {
        mismatch = true;
        break;
      }
    }
    return mismatch;
  };
  bool need_reshard = false;
  int32_t n_mesh_dim = process_mess.ndim();
  std::vector<std::shared_ptr<PlacementStatus>> best_placements(
      n_mesh_dim, std::make_shared<ReplicatedStatus>());
  // a dim can not be sharded twice along diffrent mesh_dim
  std::set<int64_t> sharded_dims = {dim};

  for (int32_t mesh_dim = 0; mesh_dim < process_mess.ndim(); ++mesh_dim) {
    if (!has_mismatch(mesh_dim)) {
      // use the old placement
      auto& best = inputs_placements[non_empty_index][mesh_dim];
      if (best->is_shard()) {
        auto shard_placement = std::dynamic_pointer_cast<ShardStatus>(best);
        sharded_dims.insert(shard_placement->get_axis());
      }
      best_placements[mesh_dim] = best;
    }
  }

  for (int32_t mesh_dim = 0; mesh_dim < process_mess.ndim(); ++mesh_dim) {
    if (!has_mismatch(mesh_dim)) {
      continue;
    }
    need_reshard = true;
    std::vector<double> costs;
    for (int32_t shard_dim = 0; shard_dim < ndim; shard_dim++) {
      double cost = std::numeric_limits<double>::infinity();
      if (!sharded_dims.count(shard_dim)) {
        cost = 0.0;
        for (size_t i = 0; i < n_inputs; i++) {
          auto& tensor_shape = tensor_shapes[i];
          auto& tensor_dist_attr = input_attrs[i];
          if (is_empty(tensor_shape)) {
            continue;
          }

          if (tensor_shape[shard_dim] < process_mess.dim_size(mesh_dim)) {
            // should not be selected
            cost += std::numeric_limits<double>::infinity();
            continue;
          }
          if (IsDimSharded(tensor_dist_attr, shard_dim)) {
            continue;
          }
          int64_t num = std::accumulate(tensor_shape.begin(),
                                        tensor_shape.end(),
                                        1,
                                        std::multiplies<int64_t>());
          if (num == static_cast<int64_t>(0)) {
            continue;
          }
          std::vector<int64_t> local_shape =
              GetLocalShape(tensor_shape, process_mess, inputs_placements[i]);
          cost += std::accumulate(local_shape.begin(),
                                  local_shape.end(),
                                  1,
                                  std::multiplies<int64_t>()) *
                  process_mess.dim_size(mesh_dim);
        }
      }
      costs.push_back(cost);
    }
    auto min_itr = std::min_element(costs.begin(), costs.end());
    auto min_dim = min_itr - costs.begin();
    if (!sharded_dims.count(min_dim)) {
      best_placements[mesh_dim] = std::make_shared<ShardStatus>(min_dim);
      sharded_dims.insert(min_dim);
    }
  }
  // set placement to the best placements
  if (need_reshard) {
    std::vector<TensorDistAttr> new_input_attrs;
    for (auto& e : input_attrs) {
      new_input_attrs.emplace_back(FromPlacements(e, best_placements));
    }
    std::swap(input_attrs, new_input_attrs);
  }
  return {{input_attrs}, {input_attrs[non_empty_index]}};
}

SpmdInfo ConcatInferSpmdReverse(const std::vector<DistMetaTensor>& x,
                                const DistMetaTensor& output,
                                int axis) {
  // TODO(liuzhenhai): add latter
  return SpmdInfo();
}
SpmdInfo ConcatInferSpmdDynamic(const std::vector<DistMetaTensor>& x,
                                const Scalar& axis) {
  return ConcatInferSpmd(x, axis.to<int32_t>());
}
}  // namespace distributed
}  // namespace phi

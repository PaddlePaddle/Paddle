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
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo ConcatInferSpmd(const std::vector<DistMetaTensor>& x,
                         const Scalar& axis) {
  /*
# paddle.concat requires all tensors must either have the same shape (except
# in the concatenating dimension) or be "empty". "Empty" here strictly means
# tensor.shape is torch.Size([0]). When tensor.ndim > 1, it will be treated
# as a non-empty tensor and the shape must match on non-cat dimensions.
 */
  std::vector<std::vector<int64_t>> tensor_shapes;
  auto get_shape = [](const DistMetaTensor& meta) {
    return phi::vectorize<int64_t>(meta.dims());
  };
  std::transform(
      x.begin(), x.end(), std::back_inserter(tensor_shapes), get_shape);
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
  auto& non_empty_example =
      *std::find_if(tensor_shapes.begin(), tensor_shapes.end(), not_empty);
  int64_t ndim = static_cast<int64_t>(non_empty_example.size());
  // normlize dim
  int64_t dim = axis.to<int64_t>();
  dim = dim < 0 ? dim + ndim : dim;

  std::vector<TensorDistAttr> input_attrs;
  // 1、make sure all tensors replicated on concat dim
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

  // 2、align non-concat dimensions according to cost
  std::vector<TensorDistAttr> best_dist_attr;
  for (const auto& dist_attr : input_attrs) {
  }
}
}  // namespace distributed
}  // namespace phi

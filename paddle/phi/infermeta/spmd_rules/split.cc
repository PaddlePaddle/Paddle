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

#include "paddle/phi/infermeta/spmd_rules/split.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo SplitWithNumInferSpmd(const DistMetaTensor& x, int num, int axis) {
  // Step0: Verify input args based on split logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                   "dims_mapping size [%d] are not matched.",
                                   x_ndim,
                                   x_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";
  if (axis < 0) {
    axis += x_ndim;
  }

  // get einsum notation for input, use a special
  // notation 'k' to mark the splitted axis in input
  std::string x_axes = alphabet.substr(0, x_ndim);
  x_axes[axis] = 'k';

  // get einsum notation for output
  std::string out_axes(x_axes);
  // the splitted axis cannot be sharded, set its notation
  // with the special '1' to set its dim mapping to -1.
  out_axes[axis] = '1';

  // Step2: Sharding Propogation
  // Step2.1: merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // get the dist attributes for all outputs, the
  // dist attributes are same for all outputs.
  std::vector<TensorDistAttr> out_dist_attrs;
  for (int i = 0; i < num; i++) {
    out_dist_attrs.emplace_back(CopyTensorDistAttrForOutput(x_dist_attr_src));
    out_dist_attrs[i].set_dims_mapping(out_dims_mapping);
  }

  // Step2.3 get new dist attribute for input. the splitted
  // cannot be sharded, if it is sharded, set it to replicated.
  TensorDistAttr x_dist_attr_dst(x_dist_attr_src);
  x_dims_mapping[axis] = -1;
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  // Step3 Handle input tensor partial (TODO)
  VLOG(4) << "SplitWithNumInferSpmd:";
  VLOG(4) << "Einsum Notation: " << x_axes << "-->" << out_axes;
  VLOG(4) << "Input shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  for (int64_t i = 0; i < num; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " dims_mapping: ["
            << str_join(out_dims_mapping) << "]";
  }
  VLOG(4) << std::endl;

  return {{x_dist_attr_dst}, out_dist_attrs};
}

SpmdInfo SplitWithNumInferSpmdReverse(
    const DistMetaTensor& x,
    const std::vector<const DistMetaTensor*>& outs,
    int num,
    int axis) {
  // Step0: Verify input args based on split logic
  int nouts = outs.size();
  int out_ndim = phi::vectorize(outs[0]->dims()).size();
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto x_dist_attr = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr.dims_mapping();
  PADDLE_ENFORCE_EQ(nouts,
                    num,
                    phi::errors::InvalidArgument(
                        "The size of Output Tensors [%d] is not equal "
                        "to the specified split number [%d]",
                        nouts,
                        num));
  PADDLE_ENFORCE_EQ(
      x_ndim,
      out_ndim,
      phi::errors::InvalidArgument("The Tensor X's rank [%d] is not equal "
                                   "to the Tensor Out's rank [%d]",
                                   x_ndim,
                                   out_ndim));
  for (int i = 0; i < num; i++) {
    auto shape = phi::vectorize(outs[i]->dims());
    int ndim = shape.size();
    auto dist_attr = outs[i]->dist_attr();
    int dims_mapping_size = dist_attr.dims_mapping().size();
    PADDLE_ENFORCE_EQ(
        ndim,
        dims_mapping_size,
        phi::errors::InvalidArgument("The Tensor Out[%d]'s rank [%d] and Its "
                                     "dims_mapping size [%d] are not matched.",
                                     i,
                                     ndim,
                                     dims_mapping_size));
  }

  // Step1: Build Einsum Notation
  if (axis < 0) {
    axis += x_ndim;
  }
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";

  // get einsum notation for input, use a special
  // notation 'k' to mark the splitted axis in input
  std::string x_axes = alphabet.substr(0, x_ndim);
  x_axes[axis] = 'k';

  // get einsum notation for output
  std::string out_axes(x_axes);
  out_axes[axis] = 'k';

  // Step2: Sharding Propogation
  // Step2.1: merge output shardings
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  for (int i = 0; i < nouts; i++) {
    std::vector<int64_t> out_dims_mapping = outs[i]->dist_attr().dims_mapping();
    axes_sharding_info.emplace_back(std::make_pair(out_axes, out_dims_mapping));
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // Step2.2: infer input dims mapping from output dims mapping
  // the split axis in input is set to -1.
  x_dims_mapping = GetDimsMappingForAxes(x_axes, axis_to_dim_map, true);
  x_dims_mapping[axis] = -1;
  x_dist_attr.set_dims_mapping(x_dims_mapping);

  // step2.3 get new dist attribute for output. the splitted
  // cannot be sharded, if it is sharded, set it to replicated.
  std::vector<TensorDistAttr> out_dist_attrs;
  for (int i = 0; i < nouts; i++) {
    out_dist_attrs.emplace_back(outs[i]->dist_attr());
    std::vector<int64_t> out_dims_mapping =
        GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
    out_dims_mapping[axis] = -1;
    out_dist_attrs[i].set_dims_mapping(out_dims_mapping);
  }

  // step3 Handle input tensor partial (TODO)

  VLOG(4) << "SplitWithNumInferSpmdReverse:";
  VLOG(4) << "Einsum Notation: " << x_axes << "-->" << out_axes;
  for (int i = 0; i < nouts; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(phi::vectorize(outs[i]->dims())) << "] "
            << "src_dims_mapping: ["
            << str_join(outs[i]->dist_attr().dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(out_dist_attrs[i].dims_mapping()) << "]";
  }
  VLOG(4) << "Input shape: [" << str_join(x_shape) << "] "
          << "dims_mapping: [" << str_join(x_dims_mapping) << "]\n\n";

  return {{x_dist_attr}, out_dist_attrs};
}

SpmdInfo SplitInferSpmd(const DistMetaTensor& x,
                        const std::vector<int>& sections,
                        int axis) {
  int num = sections.size();
  return SplitWithNumInferSpmd(x, num, axis);
}

SpmdInfo SplitInferSpmdReverse(const DistMetaTensor& x,
                               const std::vector<const DistMetaTensor*>& outs,
                               const std::vector<int>& sections,
                               int axis) {
  int num = sections.size();
  return SplitWithNumInferSpmdReverse(x, outs, num, axis);
}

}  // namespace distributed
}  // namespace phi

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

#include "paddle/phi/infermeta/spmd_rules/reduction.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
std::string GetOutputNotation(int input_ndim,
                              const std::string& input_axes,
                              std::vector<int> reduce_dims,
                              bool keep_dim) {
  // convert the negative dim value to normal dim value
  for (auto& reduce_dim : reduce_dims) {
    if (reduce_dim < 0) {
      reduce_dim = input_ndim + reduce_dim;
    }
  }

  std::string output_axes = "";
  for (int i = 0; i < input_ndim; i++) {
    std::vector<int>::iterator iter =
        std::find(reduce_dims.begin(), reduce_dims.end(), i);
    if (iter != reduce_dims.end()) {
      // if i is reduce dim, the corresponding input axis
      // will not be appended at the end of output_axes
      if (keep_dim) {
        output_axes.append(1, '1');
      }
    } else {
      // otherwise, the corresponding input axis
      // will be appended at the end of output_axes
      output_axes.append(1, input_axes[i]);
    }
  }

  return output_axes;
}

SpmdInfo ReductionInferSpmd(const DistMetaTensor& x,
                            const std::vector<int>& axis,
                            bool keep_dim) {
  // Step0: Verify input args based on reduction logic
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
  // get einsum notation for input
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim);

  // get einsum notation for output
  std::string out_axes = GetOutputNotation(x_ndim, alphabet, axis, keep_dim);

  // Step2: Sharding Propogation
  // Step2.1: Merge input shardings
  std::pair<std::string, std::vector<int64_t>> x_sharding_info(x_axes,
                                                               x_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({x_sharding_info});

  // Step2.2: Infer output dimsmapping from merged input dimsmapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step3: handle partial
  // Step3.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, out_axes);
  out_dist_attr.set_partial_status(
      partial_on_dims /*, handle reduce_type in future  */);

  // Step3.2  handle input tensor partial (TODO)
  // If the op is a linear op, i.e. `linearity` is true, it supports
  // the input to be partial. Otherwise, the input cannot be partial
  // on reduced axes, we should reshard the input when the reduced
  // axes are parital.
  VLOG(4) << "ReductionInferSpmd:";
  VLOG(4) << "axis: " << str_join(axis) << ", keep_dim: " << keep_dim;
  VLOG(4) << "Einsum Notation: " << x_axes << " --> " << out_axes;
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "] "
          << "partial_on_dims: [" + str_join(partial_on_dims) + "]\n\n";

  return {{x_dist_attr_src}, {out_dist_attr}};
}

SpmdInfo ReductionInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const std::vector<int>& axis,
                                   bool keep_dim) {
  // Step0: Verify input args based on reduction logic
  auto x_shape = phi::vectorize(x.dims());
  auto out_shape = phi::vectorize(out.dims());
  int x_ndim = x_shape.size();
  int out_ndim = out_shape.size();
  auto out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                   "dims_mapping size [%d] are not matched.",
                                   out_ndim,
                                   out_dims_mapping.size()));

  // Step1: Build einsum notation
  // get einsum notation for input
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim);

  // get einsum notation for output
  std::string out_axes = GetOutputNotation(x_ndim, alphabet, axis, keep_dim);

  // Step2: Sharding propogation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}});

  // Step2.2: Infer input dims mapping from output dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map, true);

  // initialize input dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr x_dist_attr_dst(x.dist_attr());
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  // Step3: handle partial (TODO)

  VLOG(4) << "ReductionInferSpmdReverse: ";
  VLOG(4) << "Output shape:[" << str_join(out_shape) << "] dims_mapping: ["
          << str_join(out_dims_mapping) << "]";
  VLOG(4) << "Input0: "
          << "shape: [" << str_join(x_shape) << "] "
          << "dims_mapping: [" << str_join(x_dims_mapping) << "]\n\n";

  return {{x_dist_attr_dst}, {out_dist_attr_src}};
}

}  // namespace distributed
}  // namespace phi

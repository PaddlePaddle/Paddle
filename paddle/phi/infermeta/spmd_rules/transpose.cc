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

#include "paddle/phi/infermeta/spmd_rules/transpose.h"
#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
std::string GetTransposeOutputNotation(int input_ndim,
                                       const std::string& x_axes,
                                       std::vector<int> perm_dims) {
  // convert the negative dim value to normal dim value
  for (int i = 0, n = perm_dims.size(); i < n; ++i) {
    if (perm_dims[i] < 0) {
      perm_dims[i] = input_ndim + perm_dims[i];
    }
  }

  std::string out_axes = "";
  for (int64_t i = 0; i < input_ndim; i++) {
    out_axes.append(1, x_axes[perm_dims[i]]);
  }

  return out_axes;
}
////////////////// InferMeta(Contains SPMD) Functions //////////////////
SpmdInfo TransposeInferSpmd(const DistMetaTensor& x,
                            const std::vector<int>& perm) {
  // Step0: Verify input args based on transpose logic
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
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // get einsum notation for input
  std::string x_axes = alphabet.substr(0, x_ndim);

  // get einsum notation for output
  std::string out_axes = GetTransposeOutputNotation(x_ndim, x_axes, perm);

  // Step2: Sharding Propogation
  // Step2.1: Merge input shardings
  std::pair<std::string, std::vector<int64_t>> x_sharding_info(
      {x_axes, x_dims_mapping});
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({x_sharding_info});

  // Step2.2: Infer output dimsmapping from merged input dimsmapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step3  Handle Partial (TODO)

  VLOG(4) << "TransposeInferSpmd:";
  VLOG(4) << "Input: shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Perm: [" << str_join(perm) << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  return {{x_dist_attr_src}, {out_dist_attr}};
}

SpmdInfo TransposeInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const std::vector<int>& perm) {
  // Step0: Verify input args based on transpose logic
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

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // get einsum notation for input
  std::string x_axes = alphabet.substr(0, x_ndim);

  // get einsum notation for output
  std::string out_axes = GetTransposeOutputNotation(x_ndim, x_axes, perm);

  // Step2: Sharding Propogation
  // Step2.1: merge input shardings
  std::pair<std::string, std::vector<int64_t>> out_sharding_info(
      {out_axes, out_dims_mapping});
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({out_sharding_info});

  // step2.2: infer input dims mapping from merged output dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr x_dist_attr = CopyTensorDistAttrForOutput(x.dist_attr());
  x_dist_attr.set_dims_mapping(x_dims_mapping);

  // Step3  Handle partial (TODO)

  VLOG(4) << "TransposeInferSpmdReverse:";
  VLOG(4) << "Output shape: [" << str_join(out_shape) << "] "
          << "dims_mapping: [" << str_join(out_dims_mapping) << "]";
  VLOG(4) << "Perm: [" << str_join(perm) << "]";
  VLOG(4) << "Input shape: [" << str_join(x_shape) << "] "
          << "dims_mapping: [" << str_join(x_dims_mapping) << "]\n\n";

  return {{x_dist_attr}, {out_dist_attr_src}};
}

}  // namespace distributed
}  // namespace phi

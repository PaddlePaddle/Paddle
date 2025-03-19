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

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

void BuildEinsumNotation(const size_t x_ndim,
                         std::vector<int> perm,
                         std::string* p_x_axes,
                         std::string* p_out_axes) {
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // get einsum notation for x
  *p_x_axes = alphabet.substr(0, x_ndim);

  // convert perm
  for (size_t i = 0; i < x_ndim; i++) {
    if (perm[i] < 0) {
      perm[i] += x_ndim;
    }
  }

  // get einsum notation for out
  *p_out_axes = "";
  for (size_t i = 0; i < x_ndim; i++) {
    p_out_axes->append(1, p_x_axes->at(perm[i]));
  }
}

////////////////// InferMeta(Contains SPMD) Functions //////////////////
SpmdInfo TransposeInferSpmd(const DistMetaTensor& x,
                            const std::vector<int>& perm) {
  // Step0: Verify input args based on transpose logic
  std::vector<int64_t> x_shape = common::vectorize(x.dims());
  size_t x_ndim = x_shape.size();
  const TensorDistAttr& x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));
  // check perm size
  PADDLE_ENFORCE_EQ(
      x_ndim,
      perm.size(),
      common::errors::InvalidArgument("The Tensor X's rank [%d] and "
                                      "perm size [%d] are not matched.",
                                      x_ndim,
                                      perm.size()));

  // Step1: Build Einsum Notation
  std::string x_axes;
  std::string out_axes;
  BuildEinsumNotation(x_ndim, perm, &x_axes, &out_axes);

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::pair<std::string, std::vector<int64_t>> x_sharding_info(
      {x_axes, x_dims_mapping});
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({x_sharding_info});

  // Step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_partial_status(x_dist_attr_src.partial_status());
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  out_dist_attr.set_partial_status(x_dist_attr_dst.partial_status());

  VLOG(4) << "TransposeInferSpmd:";
  VLOG(4) << "Input: shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Perm: [" << str_join(perm) << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo TransposeInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const std::vector<int>& perm) {
  // Step0: Verify input args based on transpose logic
  const std::vector<int64_t> x_shape = common::vectorize(x.dims());
  const std::vector<int64_t> out_shape = common::vectorize(out.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int out_ndim = static_cast<int>(out_shape.size());
  TensorDistAttr out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      x_ndim,
      out_ndim,
      common::errors::InvalidArgument("The Tensor X's rank [%d] and "
                                      "Out's rank [%d] are not matched.",
                                      x_ndim,
                                      out_ndim));
  // check perm size
  PADDLE_ENFORCE_EQ(
      out_ndim,
      perm.size(),
      common::errors::InvalidArgument("The Tensor Out's rank [%d] and "
                                      "perm size [%d] are not matched.",
                                      out_ndim,
                                      perm.size()));

  // Step1: Build Einsum Notation
  std::string x_axes;
  std::string out_axes;
  BuildEinsumNotation(x_ndim, perm, &x_axes, &out_axes);

  // Step2: Sharding Propagation
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

  auto out_dist_attr_dst = CopyTensorDistAttrForOutput(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Step3  Handle partial (TODO)

  VLOG(4) << "TransposeInferSpmdReverse:";
  VLOG(4) << "Output shape: [" << str_join(out_shape) << "] "
          << "dims_mapping: [" << str_join(out_dims_mapping) << "]";
  VLOG(4) << "Perm: [" << str_join(perm) << "]";
  VLOG(4) << "Input shape: [" << str_join(x_shape) << "] "
          << "dims_mapping: [" << str_join(x_dims_mapping) << "]\n\n";

  return {{x_dist_attr}, {out_dist_attr_dst}};
}

SpmdInfo TransposeGradInferSpmd(const DistMetaTensor& out_grad,
                                const std::vector<int>& perm) {
  const std::vector<int64_t> out_grad_shape =
      common::vectorize(out_grad.dims());
  size_t out_grad_ndim = out_grad_shape.size();
  const std::vector<int64_t> out_grad_dims_mapping =
      out_grad.dist_attr().dims_mapping();
  size_t out_grad_dims_mapping_size = out_grad_dims_mapping.size();
  PADDLE_ENFORCE_EQ(out_grad_ndim,
                    out_grad_dims_mapping_size,
                    common::errors::InvalidArgument(
                        "The Tensor Out_grad's rank [%d] and "
                        "Out_grad's dims_mapping size [%d] are not matched.",
                        out_grad_ndim,
                        out_grad_dims_mapping_size));
  size_t perm_size = perm.size();
  PADDLE_ENFORCE_EQ(out_grad_ndim,
                    perm_size,
                    common::errors::InvalidArgument(
                        "The Tensor Out_grad's rank [%d] and perm size "
                        "[%d] are not matched.",
                        out_grad_ndim,
                        perm_size));
  std::vector<int64_t> x_dims_mapping(out_grad_ndim, -1);
  for (size_t i = 0; i < perm.size(); ++i) {
    int origin_index = perm[i] >= 0 ? perm[i] : out_grad_ndim + perm[i];
    x_dims_mapping[origin_index] = out_grad_dims_mapping[i];
  }

  auto out_grad_dist_attr = CopyTensorDistAttrForOutput(out_grad.dist_attr());
  out_grad_dist_attr.set_dims_mapping(out_grad_dims_mapping);
  auto x_grad_dist_attr = CopyTensorDistAttrForOutput(out_grad.dist_attr());
  x_grad_dist_attr.set_dims_mapping(x_dims_mapping);
  return {{out_grad_dist_attr}, {x_grad_dist_attr}};
}

}  // namespace phi::distributed

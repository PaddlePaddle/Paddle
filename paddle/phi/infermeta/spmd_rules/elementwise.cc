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

#include "paddle/phi/infermeta/spmd_rules/elementwise.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
std::string GetInputBroadcastNotation(const std::vector<int64_t>& shape,
                                      const int max_ndim,
                                      const std::string& alphabet,
                                      std::vector<int>* broadcast_axis_count) {
  int ndim = shape.size();
  int start_dim = max_ndim - ndim;
  std::string axes_notation = GetBroadcastAxes(ndim, max_ndim, alphabet);

  for (int idim = 0; idim < max_ndim; idim++) {
    // deal with the broadcast axes, record the
    // input number at each broadcast axis
    if (idim < start_dim) {
      (*broadcast_axis_count)[idim] += 1;
    } else if (shape[idim - start_dim] == 1) {
      (*broadcast_axis_count)[idim] += 1;
      // mark the broadcast axis to a special "1"
      axes_notation[idim - start_dim] = '1';
    }
  }
  return axes_notation;
}

void GetBinaryNotations(const std::vector<int64_t>& x_shape,
                        const std::vector<int64_t>& y_shape,
                        std::string* x_axes,
                        std::string* y_axes,
                        std::string* out_axes) {
  int x_ndim = x_shape.size();
  int y_ndim = y_shape.size();
  int max_ndim = std::max(x_ndim, y_ndim);
  int ninputs = 2;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::vector<int> input_ndims({x_ndim, y_ndim});

  // get einsum notation for each input, deal with broadcast
  std::vector<int> broadcast_axis_count(max_ndim, 0);
  *x_axes = GetInputBroadcastNotation(
      x_shape, max_ndim, alphabet, &broadcast_axis_count);
  *y_axes = GetInputBroadcastNotation(
      y_shape, max_ndim, alphabet, &broadcast_axis_count);

  // get einsum notation for output
  *out_axes = GetBroadcastAxes(max_ndim, max_ndim, alphabet);
  for (int64_t idim = 0; idim < max_ndim; idim++) {
    // if all inputs broadcast at this dimension,
    // mark this axis in output as broadcast
    if (broadcast_axis_count[idim] == ninputs) {
      (*out_axes)[idim] = '1';
    }
  }
}

SpmdInfo ElementwiseUnaryInferSpmd(const DistMetaTensor& x) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    phi::errors::InvalidArgument(
                        "ElementwiseUnary, The Tensor X's rank [%d] and X's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // Step2: Sharding Propogation
  // Step2.1: Merge input shardings
  std::pair<std::string, std::vector<int64_t>> axes_sharding_info(
      x_axes, x_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({axes_sharding_info});

  // step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step3: Handle partial
  // Handle input tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferForward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] ";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  return {{x_dist_attr_src}, {out_dist_attr}};
}

SpmdInfo ElementwiseUnaryInferSpmdReverse(const DistMetaTensor& x,
                                          const DistMetaTensor& out) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto out_shape = phi::vectorize(out.dims());
  int out_ndim = out_shape.size();
  TensorDistAttr out_dist_attr = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      phi::errors::InvalidArgument(
          "ElementwiseUnaryReverse, The Tensor Out's rank [%d] and X's "
          "dims_mapping size [%d] are not matched.",
          out_ndim,
          out_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      out_ndim,
      x_ndim,
      phi::errors::InvalidArgument(
          "ElementwiseUnaryReverse, The Tensor Out's rank [%d] and X's "
          "rank [%d] are not matched.",
          out_ndim,
          x_ndim));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // Step2: Sharding Propogation
  // Step2.1: Merge output shardings
  std::pair<std::string, std::vector<int64_t>> axes_sharding_info(
      out_axes, out_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({axes_sharding_info});

  // step2.2: Infer input dims mapping from merged input dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  TensorDistAttr x_dist_attr(x.dist_attr());
  x_dist_attr.set_dims_mapping(x_dims_mapping);

  // Step3: Handle partial
  // Handle output tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferReverse:";
  VLOG(4) << "Output0 shape: [" << str_join(out_shape) << "] "
          << "dims_mapping: [" << str_join(out_dims_mapping) << "] ";
  VLOG(4) << "Input0 dims_mapping: [" + str_join(x_dims_mapping) + "]\n\n";

  return {{x_dist_attr}, {out_dist_attr}};
}

SpmdInfo ElementwiseBinaryInferSpmd(const DistMetaTensor& x,
                                    const DistMetaTensor& y) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto y_shape = phi::vectorize(y.dims());
  int y_ndim = y_shape.size();
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  TensorDistAttr y_dist_attr_src = y.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    phi::errors::InvalidArgument(
                        "ElementwiseBinary, The Tensor X's rank [%d] and X's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(y_ndim,
                    y_dims_mapping.size(),
                    phi::errors::InvalidArgument(
                        "ElementwiseBinary, The Tensor Y's rank [%d] and Y's "
                        "dims_mapping size [%d] are not matched.",
                        y_ndim,
                        y_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string x_axes, y_axes, out_axes;
  GetBinaryNotations(x_shape, y_shape, &x_axes, &y_axes, &out_axes);

  // Step2: Sharding Propogation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping}, {y_axes, y_dims_mapping}});

  // Step2.2: Infer output dimsmapping from merged input dimsmapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step2.3: Update inputs' dims mapping with merged one.
  TensorDistAttr x_dist_attr_dst(x_dist_attr_src);
  TensorDistAttr y_dist_attr_dst(y_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  y_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(y_axes, axis_to_dim_map));

  // Step3: Handle partial
  // Handle input tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferForward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Input1 shape: [" << str_join(y_shape) << "] "
          << "src_dims_mapping: [" << str_join(y_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(y_dist_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  return {{x_dist_attr_dst, y_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo ElementwiseBinaryInferSpmdReverse(const DistMetaTensor& x,
                                           const DistMetaTensor& y,
                                           const DistMetaTensor& out) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto y_shape = phi::vectorize(y.dims());
  int y_ndim = y_shape.size();
  auto out_shape = phi::vectorize(out.dims());
  int out_ndim = out_shape.size();
  int max_ndim = std::max(x_ndim, y_ndim);
  TensorDistAttr out_dist_attr = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      phi::errors::InvalidArgument(
          "ElementwiseBinaryReverse, The Tensor Out's rank [%d] and Out's "
          "dims_mapping size [%d] are not matched.",
          out_ndim,
          out_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      out_ndim,
      max_ndim,
      phi::errors::InvalidArgument(
          "ElementwiseBinaryReverse, The Tensor Out's rank [%d] and the "
          "max rank of inputs [%d] are not matched.",
          out_ndim,
          max_ndim));

  // Step1: Build Einsum Notation
  std::string x_axes, y_axes, out_axes;
  GetBinaryNotations(x_shape, y_shape, &x_axes, &y_axes, &out_axes);

  // Step2: Sharding Propogation
  // Step2.1: Merge output shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}});

  // Step2.2: Infer input dims mappings from merged output dims mapping
  TensorDistAttr x_dist_attr_dst = x.dist_attr();
  TensorDistAttr y_dist_attr_dst = y.dist_attr();
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<int64_t> y_dims_mapping =
      GetDimsMappingForAxes(y_axes, axis_to_dim_map);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  y_dist_attr_dst.set_dims_mapping(y_dims_mapping);

  // Step3: Handle partial
  // Handle input tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferReverse:";
  VLOG(4) << "Output shape: [" << str_join(out_shape) << "] dims_mapping: ["
          << str_join(out_dims_mapping) << "]";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Input1 shape: [" << str_join(y_shape) << "] "
          << "dims_mapping: [" << str_join(y_dims_mapping) << "]\n\n";

  return {{x_dist_attr_dst, y_dist_attr_dst}, {out_dist_attr}};
}

}  // namespace distributed
}  // namespace phi

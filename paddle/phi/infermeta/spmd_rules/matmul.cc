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

#include "paddle/phi/infermeta/spmd_rules/matmul.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////

TensorDistAttr GetMatmulInferedDistAttr(
    const TensorDistAttr& origin_dist_attr,
    const std::vector<int64_t>& shape,
    const std::string& tensor_axis,
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    bool trans_axis) {
  TensorDistAttr dist_attr = CopyTensorDistAttrForOutput(origin_dist_attr);
  std::vector<int64_t> infered_dims_mapping;
  infered_dims_mapping.reserve(tensor_axis.size());

  for (size_t i = 0; i < tensor_axis.size(); ++i) {
    if (shape.size() > i && shape[i] == 1) {
      infered_dims_mapping.push_back(-1);
    } else {
      auto itr = axis_to_dim_map.find(tensor_axis.substr(i, 1));
      if (itr == axis_to_dim_map.end()) {
        // infer the k axis as -1 in inferbackward.
        infered_dims_mapping.push_back(-1);
      } else {
        infered_dims_mapping.push_back(itr->second);
      }
    }
  }

  if (trans_axis) {
    std::iter_swap(infered_dims_mapping.end() - 2,
                   infered_dims_mapping.end() - 1);
  }

  dist_attr.set_dims_mapping(infered_dims_mapping);
  return dist_attr;
}

void FillMatmulOperandNotation(const int x_ndim,
                               const int y_ndim,
                               std::string* x_axes,
                               std::string* y_axes,
                               std::string* out_axes) {
  int max_ndim = std::max(x_ndim, y_ndim);
  // reserve the char k, m, n for matrix product notation: mk,kn -> mn
  std::string alphabet = "abcdefghijlopqrstuvwxyz";

  // Handle 4 different matmul cases in Paddle
  // vector * vector = scala
  if (x_ndim == 1 && y_ndim == 1) {
    *x_axes = "k";
    *y_axes = "k";
    *out_axes = "";
    // vector * batched matrix
  } else if (x_ndim == 1 && y_ndim > 1) {
    *x_axes = "k";
    std::string y_broadcast_axes =
        GetBroadcastAxes(y_ndim - 2, y_ndim - 2, alphabet);
    *y_axes = y_broadcast_axes + "kn";
    *out_axes = y_broadcast_axes + "n";
    // batched matrix * vector
  } else if (x_ndim > 1 && y_ndim == 1) {
    *y_axes = "k";
    std::string x_broadcast_axes =
        GetBroadcastAxes(x_ndim - 2, x_ndim - 2, alphabet);
    *x_axes = x_broadcast_axes + "mk";
    *out_axes = x_broadcast_axes + "m";
    // batched matrix * batched matrix
  } else if (x_ndim > 1 && y_ndim > 1) {
    std::string x_broadcast_axes =
        GetBroadcastAxes(x_ndim - 2, max_ndim - 2, alphabet);
    std::string y_broadcast_axes =
        GetBroadcastAxes(y_ndim - 2, max_ndim - 2, alphabet);
    *x_axes = x_broadcast_axes + "mk";
    *y_axes = y_broadcast_axes + "kn";

    if (x_ndim > y_ndim) {
      *out_axes = x_broadcast_axes + "mn";
    } else {
      *out_axes = y_broadcast_axes + "mn";
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "MatmulSPMDRule Receive Unsupported x_dim [%d] and y_dim [%d].",
        x_ndim,
        y_ndim));
  }
}

////////////////// InferMeta(Contains SPMD) Functions //////////////////

SpmdInfo MatmulSpmdInferForward(const DistMetaTensor& x,
                                const DistMetaTensor& y,
                                bool trans_x,
                                bool trans_y) {
  // Step0: verify input args based on matmul logic
  auto x_shape = phi::vectorize(x.dims());
  auto y_shape = phi::vectorize(y.dims());
  int x_ndim = x_shape.size();
  int y_ndim = y_shape.size();
  auto x_dist_attr_src = x.dist_attr();
  auto y_dist_attr_src = y.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                   "dims_mapping size [%d] are not matched.",
                                   x_ndim,
                                   x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      y_ndim,
      y_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Y's rank [%d] and Y's "
                                   "dims_mapping size [%d] are not matched.",
                                   y_ndim,
                                   y_dims_mapping.size()));

  VLOG(6) << "MatmulSPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; Y shape: [" << str_join(y_shape)
          << "], y_dims_mapping: [" << str_join(y_dims_mapping)
          << "]; trans_x: "
          << "[" << (trans_x ? "true" : "false") << "]; "
          << "trans_y: "
          << "[" << (trans_y ? "true" : "false") << "]; ";

  // Step1: build Einsum Notation
  std::string x_axes;
  std::string y_axes;
  std::string out_axes;
  FillMatmulOperandNotation(x_ndim, y_ndim, &x_axes, &y_axes, &out_axes);

  // Step2: Sharding Propogation
  if (trans_x) {
    PADDLE_ENFORCE_GE(x_ndim,
                      2,
                      phi::errors::InvalidArgument(
                          "When trans_x is True, the size of X "
                          "tensor should be greater than 2,  but got [%d].",
                          x_ndim));
    std::iter_swap(x_dims_mapping.end() - 2, x_dims_mapping.end() - 1);
  }
  if (trans_y) {
    PADDLE_ENFORCE_GE(y_ndim,
                      2,
                      phi::errors::InvalidArgument(
                          "When trans_y is True, the size of Y "
                          "tensor should be greater than 2,  but got [%d].",
                          y_ndim));
    std::iter_swap(y_dims_mapping.end() - 2, y_dims_mapping.end() - 1);
  }
  // Step2.1: Sharding Merge
  std::pair<std::string, std::vector<int64_t>> x_pair(x_axes, x_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> y_pair(y_axes, y_dims_mapping);
  auto axis_to_dim_map = ShardingMergeForTensors({x_pair, y_pair});

  // Step2.2: Infer Output's Dims Mapping.
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(out_axes.size());
  for (size_t i = 0; i < out_axes.size(); ++i) {
    out_dims_mapping.push_back(axis_to_dim_map[out_axes.substr(i, 1)]);
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Step2.3: Merge and get Inputs' New Dims Mapping.
  TensorDistAttr x_dist_attr_dst = GetMatmulInferedDistAttr(
      x_dist_attr_src, x_shape, x_axes, axis_to_dim_map, trans_x);
  TensorDistAttr y_dist_attr_dst = GetMatmulInferedDistAttr(
      y_dist_attr_src, y_shape, y_axes, axis_to_dim_map, trans_y);

  // Step2.3: Handle Partial
  // Step2.3.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, out_axes);
  output_dist_attr_dst.set_partial_status(partial_on_dims);

  // Step2.3.2  handle input tensor partial (TODO)
  VLOG(4) << "MatmulSPMDRule InferForward: "
          << "Einsum notation: [" << x_axes << "," << y_axes << " --> "
          << out_axes << "]. " << std::endl
          << "X shape: [" << str_join(x_shape) << "], src_dims_mapping: ["
          << str_join(x_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(x_dist_attr_dst.dims_mapping()) << "]; Y shape: ["
          << str_join(y_shape) << "], src_dims_mapping: ["
          << str_join(y_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(y_dist_attr_dst.dims_mapping())
          << "]; Output dims_mapping: [" << str_join(out_dims_mapping)
          << "], partial_on_dims: [" << str_join(partial_on_dims) << "]";

  return {{x_dist_attr_dst, y_dist_attr_dst}, {output_dist_attr_dst}};
}

SpmdInfo MatmulSpmdInferBackward(const DistMetaTensor& x,
                                 const DistMetaTensor& y,
                                 const DistMetaTensor& out,
                                 bool trans_x,
                                 bool trans_y) {
  auto out_shape = phi::vectorize(out.dims());
  int out_ndim = out_shape.size();

  auto x_shape = phi::vectorize(x.dims());
  auto y_shape = phi::vectorize(y.dims());
  int x_ndim = x_shape.size();
  int y_ndim = y_shape.size();
  int max_ndim = std::max(x_ndim, y_ndim);
  PADDLE_ENFORCE_EQ(max_ndim,
                    out_ndim,
                    phi::errors::InvalidArgument(
                        "The max ndim of inputs should be equal out_ndim in "
                        "Matmul, but got max ndim: [%d] and out_ndim: [%d].",
                        max_ndim,
                        out_ndim));

  auto out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();

  // step1: build Einsum Notation
  std::string x_axes;
  std::string y_axes;
  std::string out_axes;
  FillMatmulOperandNotation(x_ndim, y_ndim, &x_axes, &y_axes, &out_axes);

  // step2: Sharding Propogation
  // should not use input dims mapping for backward sharding merge
  auto axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}}, false);

  TensorDistAttr x_dist_attr_dst = GetMatmulInferedDistAttr(
      x.dist_attr(), x_shape, x_axes, axis_to_dim_map, trans_x);
  TensorDistAttr y_dist_attr_dst = GetMatmulInferedDistAttr(
      y.dist_attr(), y_shape, y_axes, axis_to_dim_map, trans_y);

  // step3: Handle Partial
  // NOTE we skip the partial backward inference in Partial Stage-I.
  // output partial --> axis k is sharded.

  VLOG(4) << "MatmulSPMDRule InferBackward: "
          << "Einsum notation: [" << x_axes << "," << y_axes << " --> "
          << out_axes << "]. " << std::endl
          << "Out shape: [" << str_join(out_shape) << "], src_dims_mapping: ["
          << str_join(out_dims_mapping) << "], dst_dims_mapping: ["
          << str_join(out_dims_mapping) << "]; Input X dims_mapping: ["
          << str_join(x_dist_attr_dst.dims_mapping())
          << "], Input Y dims_mapping:["
          << str_join(y_dist_attr_dst.dims_mapping()) << "].";

  return {{x_dist_attr_dst, y_dist_attr_dst}, {out_dist_attr_src}};
}

}  // namespace distributed
}  // namespace phi

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

#include "paddle/phi/infermeta/spmd_rules/embedding.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo EmbeddingInferSpmd(const DistMetaTensor& x,
                            const DistMetaTensor& weight,
                            int padding_idx,
                            bool sparse) {
  // Step0: Verify input args based on embedding logic
  auto x_shape = phi::vectorize(x.dims());
  auto weight_shape = phi::vectorize(weight.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int weight_ndim = static_cast<int>(weight_shape.size());
  auto x_dist_attr_src = x.dist_attr();
  auto weight_dist_attr_src = weight.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> weight_dims_mapping =
      weight_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                   "dims_mapping size [%d] are not matched.",
                                   x_ndim,
                                   x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      weight_ndim,
      weight_dims_mapping.size(),
      phi::errors::InvalidArgument("Tensor W's tensor rank [%d] and W's "
                                   "dims_mapping size [%d] are not matched.",
                                   weight_ndim,
                                   weight_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      weight_ndim,
      2,
      phi::errors::InvalidArgument("Embedding table should have TWO dimension, "
                                   "but got a tensor with [%d] dimension.",
                                   weight_ndim));

  // determine parallel mode
  int64_t weight_row_axis_mapping = weight_dims_mapping[0];

  // padding_idx s not supported by c_embedding kernel.
  // (TODO) might be could reshard as replicated when padding_idx != -1
  if (padding_idx != -1) {
    PADDLE_ENFORCE_EQ(
        weight_row_axis_mapping,
        -1,
        phi::errors::InvalidArgument(
            "Row-wise parallel of embedding table does NOT support Padding "
            "Idx, "
            "but got padding_idx [%d] and row axis of embedding table is "
            "sharded by mesh dimension [%d].",
            padding_idx,
            weight_ndim));
  }

  // (TODO) might be could reshard as replicated when sparse
  if (sparse) {
    PADDLE_ENFORCE_EQ(
        weight_row_axis_mapping,
        -1,
        phi::errors::InvalidArgument(
            "Row-wise parallel of embedding table does NOT support Sparse, but "
            "row axis of embedding table is sharded by mesh dimension [%d].",
            weight_row_axis_mapping));
  }

  VLOG(6) << "EmbeddingSPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; Weight shape: ["
          << str_join(weight_shape) << "], weight_dims_mapping: ["
          << str_join(weight_dims_mapping) << "]; padding_idx: "
          << "[" << padding_idx << "]; "
          << "sparse: "
          << "[" << (sparse ? "true" : "false") << "]; ";

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghilmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string weight_axes = "jk";
  std::string out_axes = x_axes + "k";

  // Step2: Sharding Propogation
  // Step2.1: merge input shardings
  auto axis_to_dim_map = ShardingMergeForTensors(
      {{x_axes, x_dims_mapping}, {weight_axes, weight_dims_mapping}}, false);

  // Step2.2: infer output's dims mapping.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step2.3: merge potential conflict in inputs,
  // update input dims mapping with merged shardings.
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  TensorDistAttr weight_dist_attr_dst =
      CopyTensorDistAttrForOutput(weight_dist_attr_src);
  weight_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(weight_axes, axis_to_dim_map));

  // Step3: Handle Partial
  // (TODO) support case where embedding table is partial at very beginning.
  std::vector<int64_t> partial_on_dims;
  if (weight_row_axis_mapping > -1) {
    partial_on_dims.push_back(weight_row_axis_mapping);
  }
  out_dist_attr.set_partial_status(partial_on_dims);

  VLOG(4) << "EmbeddingInferSpmd:\n"
          << "Einsum notation: [" << x_axes << "," << weight_axes << " --> "
          << out_axes << "]. " << std::endl
          << "X shape: [" << str_join(x_shape) << "], src_dims_mapping: ["
          << str_join(x_dims_mapping) << "], dst_dims_mapping: ["
          << str_join(x_dist_attr_dst.dims_mapping()) << "]\n W shape: ["
          << str_join(weight_shape) << "], src_dims_mapping: ["
          << str_join(weight_dims_mapping) << "], dst_dims_mapping: ["
          << str_join(weight_dist_attr_dst.dims_mapping())
          << "]\n Out dims_mapping: [" << str_join(out_dims_mapping)
          << "], partial_on_dims: [" << str_join(partial_on_dims) << "]\n\n";

  return {{x_dist_attr_dst, weight_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo EmbeddingInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& weight,
                                   const DistMetaTensor& out,
                                   int padding_idx,
                                   bool sparse) {
  // Step0: Verify input args based on embedding logic
  // InferBackward is called after InferForward, so we skip some checks.
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto out_shape = phi::vectorize(out.dims());
  int out_ndim = static_cast<int>(out_shape.size());

  PADDLE_ENFORCE_EQ(x_ndim,
                    out_ndim - 1,
                    phi::errors::InvalidArgument(
                        "There should be x_ndim + 1 = out_ndim in Embedding, "
                        "but got x_ndim: [%d] and out_ndim: [%d].",
                        x_ndim,
                        out_ndim));

  auto out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghilmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(out_ndim - 1, out_ndim - 1, alphabet);
  std::string weight_axes = "jk";
  std::string out_axes = x_axes + "k";

  // step2: Sharding Propogation
  // should not use input dims mapping for backward sharding merge
  auto axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}}, false);
  TensorDistAttr x_dist_attr = CopyTensorDistAttrForOutput(x.dist_attr());
  x_dist_attr.set_dims_mapping(GetDimsMappingForAxes(
      x_axes, axis_to_dim_map, /*unsharded_miss_axis=*/true));
  TensorDistAttr weight_dist_attr =
      CopyTensorDistAttrForOutput(weight.dist_attr());
  weight_dist_attr.set_dims_mapping(GetDimsMappingForAxes(
      weight_axes, axis_to_dim_map, /*unsharded_miss_axis=*/true));

  // step3: Handle Partial
  // NOTE we skip the partial backward inference in Partial Stage-I.
  // output partial --> weight sharded on first axis.

  VLOG(4) << "EmbeddingInferSpmdReverse:\n"
          << "Einsum notation: [" << x_axes << "," << weight_axes << " --> "
          << out_axes << "]. " << std::endl
          << "Out shape: [" << str_join(out_shape) << "], src_dims_mapping: ["
          << str_join(out_dims_mapping) << "], dst_dims_mapping: ["
          << str_join(out_dims_mapping) << "]\n Input X dims_mapping: ["
          << str_join(x_dist_attr.dims_mapping())
          << "]\n Input Weight dims_mapping:["
          << str_join(weight_dist_attr.dims_mapping()) << "]\n\n";

  return {{x_dist_attr, weight_dist_attr}, {out_dist_attr_src}};
}

}  // namespace distributed
}  // namespace phi

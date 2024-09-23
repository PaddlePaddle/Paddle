/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/c_embedding.h"
#include "paddle/phi/infermeta/spmd_rules/embedding.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/matmul.h"
#include "paddle/phi/infermeta/spmd_rules/reshape.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo CEmbeddingInferSpmd(const DistMetaTensor& weight,
                             const DistMetaTensor& x,
                             int padding_idx,
                             bool sparse) {
  // Step0: Verify input args based on c_embedding logic
  auto x_shape = common::vectorize(x.dims());
  auto weight_shape = common::vectorize(weight.dims());
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
      common::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      weight_ndim,
      weight_dims_mapping.size(),
      common::errors::InvalidArgument("Tensor W's tensor rank [%d] and W's "
                                      "dims_mapping size [%d] are not matched.",
                                      weight_ndim,
                                      weight_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(weight_ndim,
                    2,
                    common::errors::InvalidArgument(
                        "CEmbedding table should have TWO dimension, "
                        "but got a tensor with [%d] dimension.",
                        weight_ndim));

  // determine parallel mode
  int64_t weight_row_axis_mapping = weight_dims_mapping[0];

  VLOG(6) << "CEmbeddingSPMDRule InferForward Inputs: "
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

  // Step2: Sharding Propagation
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
  // (TODO) support case where c_embedding table is partial at very beginning.
  std::vector<int64_t> partial_on_dims;
  if (weight_row_axis_mapping > -1) {
    partial_on_dims.push_back(weight_row_axis_mapping);
  }
  out_dist_attr.set_partial_status(partial_on_dims);

  VLOG(4) << "CEmbeddingInferSpmd:\n"
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

SpmdInfo CEmbeddingGradInferSpmd(const DistMetaTensor& weight,
                                 const DistMetaTensor& x,
                                 const DistMetaTensor& out_grad,
                                 int64_t padding_idx,
                                 bool sparse) {
  PADDLE_ENFORCE_EQ(out_grad.dims().size(),
                    out_grad.dist_attr().dims_mapping().size(),
                    common::errors::InvalidArgument(
                        "The Tensor out_grad's rank [%d] and out_grad's "
                        "dims_mapping size [%d] are not matched.",
                        out_grad.dims(),
                        out_grad.dist_attr().dims_mapping().size()));

  if (sparse) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "CEmbeddingGradInferSpmd does't support sparse currently."));
  }

  // Propagate sharding info using composite operators.
  // The whole mathematical expression of CEmbeddingGrad is:
  // w_grad = einsum('...j, ...k->jk', onehot(x, j), out_grad)

  // TODO(cxxly): Simplifies the code logic of sharding propagation using
  // primitive operators.
  DistMetaTensor x_dst(x.dims(), x.dist_attr());
  DistMetaTensor w_dst(weight.dims(), weight.dist_attr());
  DistMetaTensor out_grad_dst(out_grad.dims(), out_grad.dist_attr());
  DistMetaTensor w_grad(weight.dims(), weight.dist_attr());

  // Step1: t0 = onehot(x_dst, w_dst.shape[0]) = eye(num_classes)[x_dst]
  auto t0_dims_mapping = x_dst.dist_attr().dims_mapping();
  t0_dims_mapping.emplace_back(-1);
  TensorDistAttr t0_dist_attr(x.dist_attr());
  t0_dist_attr.set_dims_mapping(t0_dims_mapping);
  auto t0_shape = phi::vectorize(x.dims());
  t0_shape.emplace_back(w_dst.dims()[0]);
  DistMetaTensor t0(phi::make_ddim(t0_shape), t0_dist_attr);

  // Step2: w_grad = einsum('...j, ...k -> jk', t0, out_grad_dst)
  // Step 2.1: Build Einsum Notation
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";
  std::string t0_axes =
      GetBroadcastAxes(t0.dims().size(), t0.dims().size(), alphabet);
  std::string out_grad_dst_axes = t0_axes.substr(0, t0_axes.length() - 1) + "k";
  std::string w_grad_axes = t0_axes.substr(t0_axes.length() - 1, 1) + "k";

  // Step2.2: Sharding Propagation
  // Step2.2.1: merge input shardings
  auto axis_to_dim_map = ShardingMergeForTensors(
      {{t0_axes, t0.dist_attr().dims_mapping()},
       {out_grad_dst_axes, out_grad_dst.dist_attr().dims_mapping()}},
      false);

  // Step2.2.2: infer output's dims mapping.
  auto w_grad_dist_attr = w_grad.dist_attr();
  std::vector<int64_t> w_grad_dims_mapping =
      GetDimsMappingForAxes(w_grad_axes, axis_to_dim_map);

  // Step2.2.3: merge potential conflict in inputs,
  // update input dims mapping with merged shardings.
  t0_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(t0_axes, axis_to_dim_map));
  auto out_grad_dst_dist_attr =
      CopyTensorDistAttrForOutput(out_grad_dst.dist_attr());
  out_grad_dst_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(out_grad_dst_axes, axis_to_dim_map));

  // Step2.2.4: Handle Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, w_grad_axes);
  w_grad_dist_attr.set_partial_status(partial_on_dims);

  // Step2.3: Update inputs info.
  // NOTE: Reshard happened on intermediate operators must be ensure propagated
  // back to first inputs.
  t0 = DistMetaTensor(t0.dims(), t0_dist_attr);
  const auto& t0_dims = t0.dist_attr().dims_mapping();
  if (x_dst.dist_attr().dims_mapping() !=
      std::vector<int64_t>(t0_dims.begin(), t0_dims.end() - 1)) {
    TensorDistAttr t0_new(t0.dist_attr());
    t0_new.set_dims_mapping(
        std::vector<int64_t>(t0_dims.begin(), t0_dims.end() - 1));
    x_dst = DistMetaTensor(x_dst.dims(), t0_new);
  }
  out_grad_dst = DistMetaTensor(out_grad_dst.dims(), out_grad_dst_dist_attr);
  w_grad = DistMetaTensor(w_grad.dims(), w_grad_dist_attr);

  VLOG(6) << "CEmbeddingGradInferSpmd:\n"
          << "Input x shape: [" << str_join(phi::vectorize(x.dims()))
          << "], src_dims_mapping: [" << str_join(x.dist_attr().dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(x_dst.dist_attr().dims_mapping()) << "]\n"
          << "Input weight shape: [" << str_join(phi::vectorize(weight.dims()))
          << "], src_dims_mapping: ["
          << str_join(weight.dist_attr().dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(w_dst.dist_attr().dims_mapping()) << "]\n"
          << "Input out_grad shape: ["
          << str_join(phi::vectorize(out_grad.dims()))
          << "], src_dims_mapping: ["
          << str_join(out_grad.dist_attr().dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(out_grad_dst.dist_attr().dims_mapping()) << "]\n"
          << "Output w_grad shape: [" << str_join(phi::vectorize(w_grad.dims()))
          << "], dims_mapping: [" << str_join(w_grad.dist_attr().dims_mapping())
          << "]\n\n";

  return {{w_dst.dist_attr(), x_dst.dist_attr(), out_grad_dst.dist_attr()},
          {w_grad.dist_attr()}};
}

}  // namespace phi::distributed

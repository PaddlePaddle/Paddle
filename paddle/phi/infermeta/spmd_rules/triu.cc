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

#include "paddle/phi/infermeta/spmd_rules/triu.h"

#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
using phi::distributed::auto_parallel::str_join;

SpmdInfo TriuInferSpmdBase(const DistMetaTensor& x) {
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  const auto& x_dist_attr_src = x.dist_attr();
  const std::vector<int64_t>& x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  PADDLE_ENFORCE_GE(x_ndim,
                    2,
                    common::errors::InvalidArgument(
                        "The Tensor x's rank [%d] must be ge than 2"));

  std::vector<int64_t> dims_to_unshard;
  for (int i = x_ndim - 2; i < x_ndim; ++i) {
    dims_to_unshard.push_back(i);
  }
  auto x_dist_attr = UnShardTensorDims(x_dist_attr_src, dims_to_unshard);
  auto out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr);
  out_dist_attr.set_dims_mapping(x_dist_attr.dims_mapping());

  VLOG(4) << "TriuInferSpmd:";

  VLOG(4) << "x shape: [" << str_join(x_shape) << "]"
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr.dims_mapping())
          << "]";

  VLOG(4) << "Output"
          << " dims_mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << std::endl;

  return SpmdInfo{{x_dist_attr}, {out_dist_attr}};
}

SpmdInfo TriuInferSpmd(const DistMetaTensor& x, int diagonal) {
  return TriuInferSpmdBase(x);
}

SpmdInfo TriuInferSpmdReverseBase(const DistMetaTensor& x,
                                  const DistMetaTensor& out) {
  auto out_shape = common::vectorize(out.dims());
  int out_ndim = out_shape.size();
  const auto& out_dist_attr_src = out.dist_attr();
  const std::vector<int64_t>& out_dims_mapping =
      out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));

  PADDLE_ENFORCE_GE(out_ndim,
                    2,
                    common::errors::InvalidArgument(
                        "The Tensor x's rank [%d] must be ge than 2"));

  std::vector<int64_t> dims_to_unshard;
  for (int i = out_ndim - 2; i < out_ndim; ++i) {
    dims_to_unshard.push_back(i);
  }
  auto out_dist_attr = UnShardTensorDims(out_dist_attr_src, dims_to_unshard);
  auto x_dist_attr = CopyTensorDistAttrForOutput(out_dist_attr);
  x_dist_attr.set_dims_mapping(out_dist_attr.dims_mapping());
  VLOG(4) << "TriuInferSpmdReverse:";

  VLOG(4) << "out shape: [" << str_join(out_shape) << "]"
          << "src_dims_mapping: [" << str_join(out_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";

  VLOG(4) << "x: "
          << "dst_dims_mapping: [" << str_join(x_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << std::endl;
  return SpmdInfo{{x_dist_attr}, {out_dist_attr}};
}

SpmdInfo TriuInferSpmdReverse(const DistMetaTensor& x,
                              const DistMetaTensor& out,
                              int diagonal) {
  return TriuInferSpmdReverseBase(x, out);
}

SpmdInfo TriuGradInferSpmdBase(const DistMetaTensor& out_grad) {
  auto out_shape = common::vectorize(out_grad.dims());
  int out_ndim = out_shape.size();
  const auto& out_dist_attr_src = out_grad.dist_attr();
  const std::vector<int64_t>& out_dims_mapping =
      out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(out_ndim,
                    out_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor out_grad's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        out_ndim,
                        out_dims_mapping.size()));

  PADDLE_ENFORCE_GE(out_ndim,
                    2,
                    common::errors::InvalidArgument(
                        "The Tensor x's rank [%d] must be ge than 2"));

  std::vector<int64_t> dims_to_unshard;
  for (int i = out_ndim - 2; i < out_ndim; ++i) {
    dims_to_unshard.push_back(i);
  }
  // partial status is erased
  auto out_grad_dist_attr =
      UnShardTensorDims(out_dist_attr_src, dims_to_unshard);
  out_grad_dist_attr.set_dims_mapping(out_grad_dist_attr.dims_mapping());
  auto in_grad_dist_attr = CopyTensorDistAttrForOutput(out_grad_dist_attr);
  in_grad_dist_attr.set_dims_mapping(out_grad_dist_attr.dims_mapping());

  VLOG(4) << "TriuGradInferSpmdBase:";

  VLOG(4) << "out_grad: " << str_join(out_shape) << "]"
          << "src_dims_mapping: [" << str_join(out_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: ["
          << str_join(out_grad_dist_attr.dims_mapping()) << "]";

  VLOG(4) << "in grad"
          << "dst_dims_mapping: [" << str_join(in_grad_dist_attr.dims_mapping())
          << "]";

  return SpmdInfo{{out_grad_dist_attr}, {in_grad_dist_attr}};
}

SpmdInfo TriuGradInferSpmd(const DistMetaTensor& out_grad, int diagonal) {
  return TriuGradInferSpmdBase(out_grad);
}

SpmdInfo TrilTriuInferSpmd(const DistMetaTensor& x, int diagonal, bool lower) {
  return TriuInferSpmdBase(x);
}

SpmdInfo TrilTriuInferSpmdReverse(const DistMetaTensor& x,
                                  const DistMetaTensor& out,
                                  int diagonal,
                                  bool lower) {
  return TriuInferSpmdReverseBase(x, out);
}
}  // namespace phi::distributed

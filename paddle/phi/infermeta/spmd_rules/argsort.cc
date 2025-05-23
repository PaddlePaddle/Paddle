/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/argsort.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo ArgSortInferSpmd(const DistMetaTensor& x,
                          int axis,
                          bool descending,
                          bool stable) {
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      errors::InvalidArgument(
          "ArgSort input rank [%d] should be equal to dims_mapping size [%d].",
          x_ndim,
          x_dims_mapping.size()));

  axis = axis < 0 ? axis + x_ndim : axis;

  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      common::errors::InvalidArgument(
          "The axis of argsort should be in range [0, %d), but got %d.",
          x_ndim,
          axis));

  std::vector<int64_t> x_dims_mapping_dst(x_dims_mapping);
  x_dims_mapping_dst[axis] = -1;
  std::vector<int64_t> y_dims_mapping_dst(x_dims_mapping_dst);
  std::vector<int64_t> indices_dims_mapping_dst(x_dims_mapping_dst);
  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  auto y_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  auto indices_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  y_dist_attr_dst.set_dims_mapping(y_dims_mapping_dst);
  indices_dist_attr_dst.set_dims_mapping(indices_dims_mapping_dst);

  VLOG(4) << "ArgSortInferSpmdBase:" << std::endl;
  VLOG(4) << "x_dist_attr_src: " << x_dist_attr_src.to_string()
          << " x_dist_attr_dst: " << x_dist_attr_dst.to_string() << std::endl;
  VLOG(4) << "y_dist_attr_dst: " << y_dist_attr_dst.to_string() << std::endl;

  return {{x_dist_attr_dst}, {y_dist_attr_dst, indices_dist_attr_dst}};
}

SpmdInfo ArgSortGradInferSpmd(const DistMetaTensor& indices,
                              const DistMetaTensor& x,
                              const DistMetaTensor& out_grad,
                              int axis,
                              bool descending,
                              bool stable) {
  // step 0: check invalidation of parameters
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      errors::InvalidArgument("ArgSortGrad input x rank [%d] should be equal "
                              "to dims_mapping size [%d].",
                              x_ndim,
                              x_dims_mapping.size()));

  auto ind_shape = common::vectorize(indices.dims());
  int ind_ndim = static_cast<int>(ind_shape.size());
  auto ind_dist_attr_src = indices.dist_attr();
  std::vector<int64_t> ind_dims_mapping = ind_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      ind_ndim,
      ind_dims_mapping.size(),
      errors::InvalidArgument("ArgSortGrad indices rank [%d] should be equal "
                              "to dims_mapping size [%d].",
                              ind_ndim,
                              ind_dims_mapping.size()));

  auto out_grad_shape = common::vectorize(out_grad.dims());
  int out_grad_ndim = static_cast<int>(out_grad_shape.size());
  auto out_grad_dist_attr_src = out_grad.dist_attr();
  std::vector<int64_t> out_grad_dims_mapping =
      out_grad_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_grad_ndim,
      out_grad_dims_mapping.size(),
      errors::InvalidArgument("ArgSortGrad out_grad rank [%d] should be equal "
                              "to dims_mapping size [%d].",
                              out_grad_ndim,
                              out_grad_dims_mapping.size()));

  PADDLE_ENFORCE_EQ(
      x_ndim == ind_ndim && x_ndim == out_grad_ndim,
      1,
      errors::InvalidArgument("ArgSortGrad x rank [%d] should be equal to "
                              "indices rank [%d] and out_grad rank [%d].",
                              x_ndim,
                              ind_ndim,
                              out_grad_ndim));

  for (int i = 0; i < x_ndim; ++i) {
    PADDLE_ENFORCE_EQ(
        x_dims_mapping[i] == ind_dims_mapping[i],
        1,
        errors::InvalidArgument("ArgSortGrad x dims_mapping[%d]=[%d] should be "
                                "equal to indices dims_mapping[%d]=[%d].",
                                i,
                                x_dims_mapping[i],
                                i,
                                ind_dims_mapping[i]));
  }

  axis = axis < 0 ? axis + x_ndim : axis;

  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      common::errors::InvalidArgument(
          "The axis of argsort should be in range [0, %d), but got %d.",
          x_ndim,
          axis));

  // step 1: infer spmd info
  std::vector<int64_t> x_dims_mapping_dst(x_dims_mapping);
  x_dims_mapping_dst[axis] = -1;
  std::vector<int64_t> out_grad_dims_mapping_dst(x_dims_mapping_dst);
  std::vector<int64_t> indices_dims_mapping_dst(x_dims_mapping_dst);
  std::vector<int64_t> x_grad_dims_mapping_dst(x_dims_mapping_dst);

  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  auto out_grad_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  auto indices_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  auto x_grad_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);

  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_dst);
  indices_dist_attr_dst.set_dims_mapping(indices_dims_mapping_dst);
  x_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  VLOG(4) << "ArgSortGradInferSpmdBase:" << std::endl;
  VLOG(4) << "indices_dist_attr_src: " << ind_dist_attr_src.to_string()
          << " indices_dist_attr_dst: " << indices_dist_attr_dst.to_string()
          << std::endl;
  VLOG(4) << "x_dist_attr_src: " << x_dist_attr_src.to_string()
          << " x_dist_attr_dst: " << x_dist_attr_dst.to_string() << std::endl;
  VLOG(4) << "out_grad_dist_attr_src: " << out_grad_dist_attr_dst.to_string()
          << " out_grad_dist_attr_dst: " << out_grad_dist_attr_dst.to_string()
          << std::endl;
  VLOG(4) << "x_grad_dist_attr_dst: " << x_grad_dist_attr_dst.to_string()
          << std::endl;
  return {{indices_dist_attr_dst, x_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr_dst}};
}

}  // namespace phi::distributed

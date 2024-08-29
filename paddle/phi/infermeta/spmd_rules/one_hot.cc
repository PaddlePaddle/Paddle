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

#include "paddle/phi/infermeta/spmd_rules/one_hot.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo OneHotInferSpmd(const DistMetaTensor& x, int num_classes) {
  // Step0: Verify input args based on split logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping_src = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping_src.size(),
      common::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping_src.size()));

  std::vector<int64_t> out_dims_mapping(x_dims_mapping_src);
  out_dims_mapping.emplace_back(-1);
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step3 Handle input tensor partial (TODO)
  VLOG(4) << "OneHotInferSpmd:";
  VLOG(4) << "x shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping_src) << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping_src) << "]";
  VLOG(4) << "Out dims_mapping: [" << str_join(out_dims_mapping) << "]";
  VLOG(4) << std::endl;
  return {{x_dist_attr_src}, {out_dist_attr}};
}

SpmdInfo OneHotInferSpmdReverse(const DistMetaTensor& x,
                                const DistMetaTensor& out,
                                int num_classes) {
  // Step0: Verify input args based on split logic
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(out);

  std::vector<int64_t> out_dims_mapping_dst(out_dims_mapping_src);
  out_dims_mapping_dst[out_ndim - 1] = -1;
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping_dst);

  std::vector<int64_t> x_dims_mapping_dst(out_dims_mapping_dst.begin(),
                                          out_dims_mapping_dst.end() - 1);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  VLOG(4) << "OneHotInferSpmdReverse:";
  VLOG(4) << "out shape: [" << str_join(out_shape) << "] "
          << "src_dims_mapping: [" << str_join(out_dims_mapping_src) << "] "
          << "dst_dims_mapping: [" << str_join(out_dims_mapping_dst) << "]";
  VLOG(4) << "x shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping_src) << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping_dst) << "]";
  VLOG(4) << std::endl;
  return {{x_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo OneHotInferSpmdDynamic(const DistMetaTensor& x,
                                const Scalar& num_classes) {
  return OneHotInferSpmd(x, num_classes.to<int32_t>());
}

}  // namespace phi::distributed

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

#include "paddle/phi/infermeta/spmd_rules/unbind.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo UnbindInferSpmd(const DistMetaTensor& x, int axis) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  if (axis < 0) {
    axis += x_ndim;
  }
  PADDLE_ENFORCE_LT(
      axis,
      x_ndim,
      common::errors::InvalidArgument("[%d] [%d] The axis [%d] should be less "
                                      "than the rank of input tensor [%d].",
                                      __FILE__,
                                      __LINE__,
                                      axis,
                                      x_ndim));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // get einsum notation for input
  std::string x_axes = alphabet.substr(0, x_ndim);
  // get einsum notation for output
  std::string out_axes(x_axes);
  out_axes.erase(axis, 1);

  // Step2: Sharding Propagation
  // Step2.1: merge input shardings
  std::vector<int64_t> x_dims_mapping_dst(x_dims_mapping_src);
  x_dims_mapping_dst[axis] = -1;
  TensorDistAttr x_dist_attr_dst(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping_dst}});

  // Step2.2: infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // get the dist attributes for all outputs, the
  // dist attributes are same for all outputs.
  int noutputs = x_shape[axis];
  std::vector<TensorDistAttr> out_dist_attrs;
  for (int i = 0; i < noutputs; i++) {
    out_dist_attrs.emplace_back(CopyTensorDistAttrForOutput(x_dist_attr_src));
    out_dist_attrs[i].set_dims_mapping(out_dims_mapping);
  }

  // Step3 Handle input tensor partial (TODO)
  VLOG(4) << "UnbindInferSpmd:";
  VLOG(4) << "Einsum Notation: " << x_axes << "-->" << out_axes;
  VLOG(4) << "x:";
  VLOG(4) << "\tshape: [" << str_join(x_shape) << "] ";
  VLOG(4) << "\tsrc_dist_attr: [" << x_dist_attr_src.to_string() << "]";
  VLOG(4) << "\tdst_dist_attr: [" << x_dist_attr_dst.to_string() << "]";
  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "out" << std::to_string(i);
    VLOG(4) << "\tdist_attr: [" << out_dist_attrs[i].to_string() << "]";
  }
  VLOG(4) << std::endl;
  // TODO(liuzhenhai): remedy this
  // should return list in list []
  // return {{x_dist_attr_dst}, {out_dist_attrs}};
  return {{x_dist_attr_dst}, ToArgDistAttr(out_dist_attrs)};
}

SpmdInfo UnbindInferSpmdReverse(const DistMetaTensor& x,
                                const std::vector<const DistMetaTensor*>& outs,
                                int axis) {
  // Step0: Verify input args based on split logic
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  int nouts = static_cast<int>(outs.size());

  for (int i = 0; i < nouts; i++) {
    auto shape = common::vectorize(outs[i]->dims());
    int ndim = static_cast<int>(shape.size());
    auto dist_attr = outs[i]->dist_attr();
    int dims_mapping_size = static_cast<int>(dist_attr.dims_mapping().size());
    PADDLE_ENFORCE_EQ(ndim,
                      dims_mapping_size,
                      common::errors::InvalidArgument(
                          "The Tensor Out[%d]'s rank [%d] and Its "
                          "dims_mapping size [%d] are not matched.",
                          i,
                          ndim,
                          dims_mapping_size));
  }

  // Step1: Build Einsum Notation
  if (axis < 0) {
    axis += x_ndim;
  }
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim);
  std::string out_axes(x_axes);
  out_axes.erase(axis, 1);

  // Step2: Sharding Propagation
  // Step2.1: merge output shardings
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  for (int i = 0; i < nouts; i++) {
    std::vector<int64_t> out_dims_mapping = outs[i]->dist_attr().dims_mapping();
    axes_sharding_info.emplace_back(out_axes, out_dims_mapping);
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // Step2.2: infer input dims mapping from output dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map, true);

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  // step2.3 get new dist attribute for output. the splitted
  // cannot be sharded, if it is sharded, set it to replicated.
  std::vector<TensorDistAttr> out_dist_attrs_dst;
  for (int i = 0; i < nouts; i++) {
    out_dist_attrs_dst.emplace_back(
        CopyTensorDistAttrForOutput(outs[i]->dist_attr()));
    std::vector<int64_t> out_dims_mapping =
        GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
    out_dist_attrs_dst[i].set_dims_mapping(out_dims_mapping);
  }

  // step3 Handle input tensor partial (TODO)

  VLOG(4) << "UnbindInferSpmdReverse:";
  for (int i = 0; i < nouts; i++) {
    VLOG(4) << "out" << std::to_string(i) << ":";
    VLOG(4) << "\tsrc_dist_attr: [" << outs[i]->dist_attr().to_string() << "]";
    VLOG(4) << "\tdst_dist_attr: [" << out_dist_attrs_dst[i].to_string() << "]";
  }
  VLOG(4) << "x:";
  VLOG(4) << "\tsrc_dist_attr: [" << x_dist_attr_src.to_string() << "]";
  VLOG(4) << "\tdst_dist_attr: [" << x_dist_attr_dst.to_string() << "]";
  return {{x_dist_attr_dst}, ToArgDistAttr(out_dist_attrs_dst)};
}

SpmdInfo UnbindInferSpmdDynamic(const DistMetaTensor& x, int axis) {
  auto tmp = UnbindInferSpmd(x, axis);
  // bridge the diff concerning vector output between static and dynamic auto
  // parallel ToDo(liuzhenhai): unify the difference between static and dynamic
  SpmdInfo ret;
  ret.first = tmp.first;
  std::vector<TensorDistAttr> out_dist_attrs;
  out_dist_attrs.reserve(tmp.second.size());
  for (const auto& out : tmp.second) {
    out_dist_attrs.push_back(PADDLE_GET_CONST(TensorDistAttr, out));
  }
  ret.second = {out_dist_attrs};
  return ret;
}

}  // namespace phi::distributed

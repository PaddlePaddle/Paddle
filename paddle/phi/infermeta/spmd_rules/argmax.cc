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

#include "paddle/phi/infermeta/spmd_rules/argmax.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo ArgMaxInferSpmdBase(const DistMetaTensor& x,
                             int axis,
                             bool keepdims,
                             bool flatten) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  axis = axis < 0 ? x_ndim + axis : axis;

  std::vector<int64_t> x_dims_mapping_dst(x_dims_mapping_src);
  std::vector<int64_t> out_dims_mapping;
  if (flatten) {
    x_dims_mapping_dst.assign(x_ndim, -1);
    if (keepdims) {
      out_dims_mapping.assign(x_ndim, -1);
    } else {
      out_dims_mapping.push_back(-1);
    }
  } else {
    x_dims_mapping_dst[axis] = -1;
    out_dims_mapping.assign(x_dims_mapping_dst.begin(),
                            x_dims_mapping_dst.end());
    if (!keepdims) {
      out_dims_mapping.erase(out_dims_mapping.begin() + axis);
    }
  }

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "ArgMaxInferSpmd:";
  VLOG(4) << "x:";
  VLOG(4) << "src_dist_attr: [" << x_dist_attr_src.to_string() << "] "
          << "dst_dist_attr: [" << x_dist_attr_dst.to_string() << "]";
  VLOG(4) << "out:";
  VLOG(4) << "dist_attr: [" << out_dist_attr.to_string() << "]" << std::endl;
  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo ArgMaxInferSpmdReverseBase(const DistMetaTensor& x,
                                    const DistMetaTensor& out,
                                    int axis,
                                    bool keepdims,
                                    bool flatten) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(out);
  axis = axis < 0 ? x_ndim + axis : axis;
  std::vector<int64_t> x_dims_mapping_dst;
  std::vector<int64_t> out_dims_mapping_dst(out_dims_mapping_src);

  if (flatten) {
    if (keepdims) {
      out_dims_mapping_dst.assign(x_ndim, -1);
    } else {
      out_dims_mapping_dst.push_back(-1);
    }
    x_dims_mapping_dst.assign(x_ndim, -1);
  } else {
    x_dims_mapping_dst.assign(out_dims_mapping_dst.begin(),
                              out_dims_mapping_dst.end());
    if (!keepdims) {
      x_dims_mapping_dst.insert(x_dims_mapping_dst.begin() + axis, -1);
    }
  }

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping_dst);

  VLOG(4) << "ArgMaxInferSpmdReverse:";
  VLOG(4) << "out:";
  VLOG(4) << "src_dist_attr: [" << out_dist_attr_src.to_string() << "] "
          << "dst_dist_attr: [" << out_dist_attr_dst.to_string() << "]";
  VLOG(4) << "x:";
  VLOG(4) << "src_dist_attr: [" << x_dist_attr_src.to_string() << "] "
          << "dst_dist_attr: [" << x_dist_attr_dst.to_string() << "]"
          << std::endl;
  return {{x_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo ArgMaxInferSpmdDynamic(const DistMetaTensor& x,
                                const Scalar& axis,
                                bool keepdims,
                                bool flatten,
                                DataType dtype) {
  return ArgMaxInferSpmdBase(x, axis.to<int32_t>(), keepdims, flatten);
}

}  // namespace phi::distributed 

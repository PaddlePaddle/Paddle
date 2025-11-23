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

#include "paddle/phi/infermeta/spmd_rules/roi_align.h"
#include "glog/logging.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo RoiAlignInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& boxes,
                           const DistMetaTensor& boxes_num,
                           int pooled_height,
                           int pooled_width,
                           float spatial_scale,
                           int sampling_ratio,
                           bool aligned) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(boxes);

  std::vector<int64_t> x_dims_mapping_dst(x_ndim, -1);
  x_dims_mapping_dst[1] = x_dims_mapping_src[1];
  std::vector<int64_t> boxes_dims_mapping_dst(boxes_ndim, -1);

  std::vector<int64_t> boxes_num_dims_mapping_dst;
  TensorDistAttr boxes_num_dist_attr_dst;

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr boxes_dist_attr_dst =
      CopyTensorDistAttrForOutput(boxes_dist_attr_src);

  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_dst);

  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  boxes_dist_attr_dst.set_dims_mapping(boxes_dims_mapping_dst);
  out_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  if (boxes_num.initialized()) {
    EXTRACT_SHAPE_AND_DIST_ATTR(boxes_num);
    boxes_num_dims_mapping_dst = {-1};
    boxes_num_dist_attr_dst =
        CopyTensorDistAttrForOutput(boxes_num_dist_attr_src);
    boxes_num_dist_attr_dst.set_dims_mapping(boxes_num_dims_mapping_dst);
    VLOG(4) << "RoiAlignInferSpmd: Done.";
    LOG_SPMD_INPUT(boxes_num);
  } else {
    boxes_num_dist_attr_dst = TensorDistAttr();
    VLOG(4) << "RoiAlignInferSpmd: Done.";
  }
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(boxes);
  LOG_SPMD_OUTPUT(out_dist_attr_dst);

  return {{x_dist_attr_dst, boxes_dist_attr_dst, boxes_num_dist_attr_dst},
          {out_dist_attr_dst}};
}

SpmdInfo RoiAlignGradInferSpmd(const DistMetaTensor& x,
                               const DistMetaTensor& boxes,
                               const DistMetaTensor& boxes_num,
                               const DistMetaTensor& out_grad,
                               int pooled_height,
                               int pooled_width,
                               float spatial_scale,
                               int sampling_ratio,
                               bool aligned) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(boxes);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  int64_t c_status = ShardingMergeForAxis(
      "c", x_dims_mapping_src[1], out_grad_dims_mapping_src[1]);
  std::vector<int64_t> x_dims_mapping_dst(x_ndim, -1);
  x_dims_mapping_dst[1] = c_status;
  std::vector<int64_t> boxes_dims_mapping_dst(boxes_ndim, -1);
  std::vector<int64_t> out_grad_dims_mapping_dst(out_grad_ndim, -1);
  out_grad_dims_mapping_dst[1] = c_status;
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr x_grad_attr_dst = x_dist_attr_dst;
  TensorDistAttr boxes_dist_attr_dst =
      CopyTensorDistAttrForOutput(boxes_dist_attr_src);
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  x_grad_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  boxes_dist_attr_dst.set_dims_mapping(boxes_dims_mapping_dst);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_dst);

  TensorDistAttr boxes_num_dist_attr_dst;
  std::vector<int64_t> boxes_num_dims_mapping_dst;
  if (boxes_num.initialized()) {
    EXTRACT_SHAPE_AND_DIST_ATTR(boxes_num);
    boxes_num_dims_mapping_dst = {-1};
    boxes_num_dist_attr_dst =
        CopyTensorDistAttrForOutput(boxes_num_dist_attr_src);
    boxes_num_dist_attr_dst.set_dims_mapping(boxes_num_dims_mapping_dst);
    VLOG(4) << "RoiAlignGradInferSpmd: Done.";
    LOG_SPMD_INPUT(boxes_num);
  } else {
    boxes_num_dist_attr_dst = TensorDistAttr();
    VLOG(4) << "RoiAlignGradInferSpmd: Done.";
  }
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(boxes);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad_attr_dst);

  return {{x_dist_attr_dst,
           boxes_dist_attr_dst,
           boxes_num_dist_attr_dst,
           out_grad_dist_attr_dst},
          {x_grad_attr_dst}};
}

}  // namespace distributed
}  // namespace phi

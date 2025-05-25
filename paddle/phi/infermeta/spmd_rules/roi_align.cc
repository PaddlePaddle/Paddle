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
  std::string x_axes = "nchw";
  std::string boxes_axes = "kf";
  std::string boxes_num_axes;
  std::string out_axes = "kchw";

  std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
  x_dims_mapping[0] = -1;
  x_dims_mapping[2] = -1;
  x_dims_mapping[3] = -1;
  std::vector<int64_t> boxes_dims_mapping(boxes_dims_mapping_src);
  boxes_dims_mapping[0] = -1;
  boxes_dims_mapping[1] = -1;
  std::vector<int64_t> boxes_num_dims_mapping;
  if (boxes_num.initialized()) {
    EXTRACT_SHAPE_AND_DIST_ATTR(boxes_num);
    boxes_num_axes = "n";
    boxes_num_dims_mapping = boxes_num_dims_mapping_src;
    boxes_num_dims_mapping[0] = -1;
  } else {
    boxes_num_axes = "";
    boxes_num_dims_mapping = {};
  }
  std::pair<std::string, std::vector<int64_t>> x_pair(x_axes, x_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> boxes_pair(boxes_axes,
                                                          boxes_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> boxes_num_pair(
      boxes_num_axes, boxes_num_dims_mapping);
  auto axis_to_dim_map =
      ShardingMergeForTensors({x_pair, boxes_pair, boxes_num_pair});

  std::vector<int64_t> x_dims_mapping_dst =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);

  std::vector<int64_t> boxes_dims_mapping_dst =
      GetDimsMappingForAxes(boxes_axes, axis_to_dim_map);
  std::vector<int64_t> boxes_num_dims_mapping_dst =
      GetDimsMappingForAxes(boxes_num_axes, axis_to_dim_map);
  std::vector<int64_t> out_dims_mapping_dst =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr boxes_dist_attr_dst =
      CopyTensorDistAttrForOutput(boxes_dist_attr_src);
  TensorDistAttr boxes_num_dist_attr_dst;
  if (boxes_num.initialized()) {
    boxes_num_dist_attr_dst = TensorDistAttr();
  } else {
    boxes_num_dist_attr_dst =
        CopyTensorDistAttrForOutput(boxes_num_dist_attr_dst);
    boxes_num_dist_attr_dst.set_dims_mapping(boxes_num_dims_mapping_dst);
  }
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_dst);

  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  boxes_dist_attr_dst.set_dims_mapping(boxes_dims_mapping_dst);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping_dst);
  VLOG(4) << "RoiAlignInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(boxes);
  if (boxes_num_dims_mapping.empty()) {
    EXTRACT_SHAPE_AND_DIST_ATTR(boxes_num);
    LOG_SPMD_INPUT(boxes_num);
  }
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
  std::string x_axes = "nchw";
  std::string boxes_axes = "kf";
  std::string boxes_num_axes;
  std::string out_grad_axes = "kchw";

  std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
  x_dims_mapping[0] = -1;
  x_dims_mapping[2] = -1;
  x_dims_mapping[3] = -1;
  std::vector<int64_t> out_grad_mapping(out_grad_dims_mapping_src);
  out_grad_mapping[0] = -1;
  out_grad_mapping[2] = -1;
  out_grad_mapping[3] = -1;
  std::vector<int64_t> boxes_dims_mapping(boxes_dims_mapping_src);
  boxes_dims_mapping[0] = -1;
  boxes_dims_mapping[1] = -1;
  std::vector<int64_t> boxes_num_dims_mapping;
  if (boxes_num.initialized()) {
    EXTRACT_SHAPE_AND_DIST_ATTR(boxes_num);
    boxes_num_axes = "n";
    boxes_num_dims_mapping = boxes_num_dims_mapping_src;
    boxes_num_dims_mapping[0] = -1;
  } else {
    boxes_num_axes = "";
    boxes_num_dims_mapping = {};
  }
  std::pair<std::string, std::vector<int64_t>> x_pair(x_axes, x_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> boxes_pair(boxes_axes,
                                                          boxes_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> boxes_num_pair(
      boxes_num_axes, boxes_num_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> out_grad_num_pair(
      out_grad_axes, out_grad_dims_mapping_src);
  auto axis_to_dim_map = ShardingMergeForTensors(
      {x_pair, boxes_pair, boxes_num_pair, out_grad_num_pair});

  std::vector<int64_t> x_dims_mapping_dst =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);

  std::vector<int64_t> boxes_dims_mapping_dst =
      GetDimsMappingForAxes(boxes_axes, axis_to_dim_map);
  std::vector<int64_t> boxes_num_dims_mapping_dst =
      GetDimsMappingForAxes(boxes_num_axes, axis_to_dim_map);
  std::vector<int64_t> out_grad_dims_mapping_dst =
      GetDimsMappingForAxes(out_grad_axes, axis_to_dim_map);

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr x_grad_attr_dst = x_dist_attr_dst;
  std::vector<int64_t> partial_on_dims;
  for (size_t i = 0; i < static_cast<size_t>(out_grad_ndim); ++i) {
    auto mapping = out_grad_dims_mapping_dst[i];
    if (mapping != -1) {
      partial_on_dims.push_back(mapping);
    }
  }
  x_grad_attr_dst.set_partial_status(partial_on_dims);
  TensorDistAttr boxes_dist_attr_dst =
      CopyTensorDistAttrForOutput(boxes_dist_attr_src);
  TensorDistAttr boxes_num_dist_attr_dst;
  if (boxes_num.initialized()) {
    boxes_num_dist_attr_dst = TensorDistAttr();
  } else {
    boxes_num_dist_attr_dst =
        CopyTensorDistAttrForOutput(boxes_num_dist_attr_dst);
    boxes_num_dist_attr_dst.set_dims_mapping(boxes_num_dims_mapping_dst);
  }
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);

  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  boxes_dist_attr_dst.set_dims_mapping(boxes_dims_mapping_dst);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_dst);
  VLOG(4) << "RoiAlignInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(boxes);
  if (boxes_num_dims_mapping.empty()) {
    EXTRACT_SHAPE_AND_DIST_ATTR(boxes_num);
    LOG_SPMD_INPUT(boxes_num);
  }
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

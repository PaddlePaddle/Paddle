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

#include "paddle/phi/infermeta/spmd_rules/c_softmax_with_multi_label_cross_entropy.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
using phi::distributed::auto_parallel::str_join;

void GetMultiLabelCrossEntropyNotations(int x_ndim,
                                        std::string* x_axes_src,
                                        std::string* x_axes_dst,
                                        std::string* label_axes_src,
                                        std::string* label_axes_dst,
                                        std::string* smooth_weight_axes_src,
                                        std::string* smooth_weight_axes_dst,
                                        std::string* loss_axes,
                                        std::string* softmax_axes_dst,
                                        bool sum_multi_label_loss) {
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  *x_axes_src = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  *x_axes_dst = *x_axes_src;
  *label_axes_src = *x_axes_src;
  (*label_axes_src)[x_ndim - 1] = (*x_axes_src)[x_ndim - 1] + 1;
  *label_axes_dst = *label_axes_src;
  *smooth_weight_axes_src = *label_axes_src;
  *smooth_weight_axes_dst = *smooth_weight_axes_src;
  if (sum_multi_label_loss) {
    *loss_axes = *x_axes_src;
    (*loss_axes)[x_ndim - 1] = '1';
  } else {
    *loss_axes = *label_axes_src;
  }
  *softmax_axes_dst = *x_axes_dst;
}

SpmdInfo CSoftmaxWithMultiLabelCrossEntropyInferSpmd(
    const DistMetaTensor& x,
    const DistMetaTensor& label,
    const DistMetaTensor& smooth_weight,
    int ignore_index,
    bool sum_multi_label_loss,
    int rank,
    int nranks) {
  // Step0: Verify input args based on c_softmax_with_multi_label_cross_entropy
  // logic
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(label);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(smooth_weight);

  VLOG(4) << "CSoftmaxWithMultiLabelCrossEntropySPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping_src: ["
          << str_join(x_dims_mapping_src) << "]; Label shape: ["
          << str_join(label_shape) << "], Label dims mapping: ["
          << str_join(label_dims_mapping_src) << "]; Smooth_weight shape: ["
          << str_join(smooth_weight_shape) << "], Smooth_weight dim mapping: ["
          << str_join(smooth_weight_dims_mapping_src) << "], ignore_index: ["
          << ignore_index << "]";

  // Step1: Build Einsum Notation
  std::string x_axes_src, x_axes_dst, label_axes_src, label_axes_dst,
      smooth_weight_axes_src, smooth_weight_axes_dst, loss_axes,
      softmax_axes_dst;
  GetMultiLabelCrossEntropyNotations(x_ndim,
                                     &x_axes_src,
                                     &x_axes_dst,
                                     &label_axes_src,
                                     &label_axes_dst,
                                     &smooth_weight_axes_src,
                                     &smooth_weight_axes_dst,
                                     &loss_axes,
                                     &softmax_axes_dst,
                                     sum_multi_label_loss);

  // Step2: Sharding Propagation
  // Step2.1: merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes_src, x_dims_mapping_src},
           {label_axes_src, label_dims_mapping_src},
           {smooth_weight_axes_src, smooth_weight_dims_mapping_src}});

  // Step2.2: infer output dims mappings
  TensorDistAttr loss_dist_attr_dst =
      CopyTensorDistAttrForOutput(label_dist_attr_src);
  loss_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(loss_axes, axis_to_dim_map));
  TensorDistAttr softmax_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  softmax_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(softmax_axes_dst, axis_to_dim_map));

  // Step2.3: update input dims mappings with merged one
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes_dst, axis_to_dim_map));
  TensorDistAttr label_dist_attr_dst =
      CopyTensorDistAttrForOutput(label_dist_attr_src);
  label_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(label_axes_dst, axis_to_dim_map));
  TensorDistAttr smooth_weight_dist_attr_dst =
      CopyTensorDistAttrForOutput(smooth_weight_dist_attr_src);
  smooth_weight_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(smooth_weight_axes_dst, axis_to_dim_map));

  VLOG(4) << "CSoftmaxWithMultiLabelCrossEntropyInferSpmd:";
  VLOG(4) << "ignore_index: [" << ignore_index << "].";

  VLOG(4) << "Einsum notation: [" << x_axes_src << "," << label_axes_src
          << " --> " << softmax_axes_dst << "," << loss_axes << "].\n";

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(label);
  LOG_SPMD_INPUT(smooth_weight);
  LOG_SPMD_OUTPUT(softmax_dist_attr_dst);
  LOG_SPMD_OUTPUT(loss_dist_attr_dst);

  return SpmdInfo(
      {x_dist_attr_dst, label_dist_attr_dst, smooth_weight_dist_attr_dst},
      {softmax_dist_attr_dst, loss_dist_attr_dst});
}

SpmdInfo CSoftmaxWithMultiLabelCrossEntropyGradSpmd(
    const DistMetaTensor& softmax,
    const DistMetaTensor& label,
    const DistMetaTensor& smooth_weight,
    const DistMetaTensor& loss_grad,
    int ignore_index,
    bool sum_multi_label_loss,
    int rank,
    int nranks) {
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(softmax);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(label);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(smooth_weight);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(loss_grad);

  std::string label_axes_src, label_axes_dst, smooth_weight_axes_src,
      smooth_weight_axes_dst, softmax_axes_src, softmax_axes_dst,
      loss_grad_axes;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  auto x_axes_src = alphabet.substr(0, loss_grad_ndim);
  auto x_axes_dst = x_axes_src;
  label_axes_src = x_axes_src;
  label_axes_src[loss_grad_ndim - 1] = x_axes_src[loss_grad_ndim - 1] + 1;
  label_axes_dst = label_axes_src;
  smooth_weight_axes_src = label_axes_src;
  smooth_weight_axes_dst = label_axes_src;
  if (sum_multi_label_loss) {
    loss_grad_axes = x_axes_src;
    loss_grad_axes[loss_grad_ndim - 1] = '1';
  } else {
    loss_grad_axes = label_axes_src;
  }

  softmax_axes_src = x_axes_dst;
  softmax_axes_dst = x_axes_dst;

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{label_axes_src, label_dims_mapping_src},
           {smooth_weight_axes_src, smooth_weight_dims_mapping_src},
           {softmax_axes_src, softmax_dims_mapping_src},
           {loss_grad_axes, loss_grad_dims_mapping_src}});

  auto label_dist_attr_dst = CopyTensorDistAttrForOutput(label_dist_attr_src);
  auto label_dims_mapping_dst =
      GetDimsMappingForAxes(label_axes_dst, axis_to_dim_map, true);
  label_dist_attr_dst.set_dims_mapping(label_dims_mapping_dst);

  auto smooth_weight_dist_attr_dst =
      CopyTensorDistAttrForOutput(smooth_weight_dist_attr_src);
  auto smooth_weight_dims_mapping_dst =
      GetDimsMappingForAxes(smooth_weight_axes_dst, axis_to_dim_map, true);
  smooth_weight_dist_attr_dst.set_dims_mapping(smooth_weight_dims_mapping_dst);

  auto softmax_dist_attr_dst =
      CopyTensorDistAttrForOutput(softmax_dist_attr_src);
  auto softmax_dims_mapping_dst =
      GetDimsMappingForAxes(softmax_axes_dst, axis_to_dim_map, true);
  softmax_dist_attr_dst.set_dims_mapping(softmax_dims_mapping_dst);

  auto loss_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(loss_grad_dist_attr_src);
  auto loss_grad_dims_mapping_dst =
      GetDimsMappingForAxes(loss_grad_axes, axis_to_dim_map, true);
  loss_grad_dist_attr_dst.set_dims_mapping(loss_grad_dims_mapping_dst);

  auto x_grad = CopyTensorDistAttrForOutput(softmax_dist_attr_dst);
  x_grad.set_dims_mapping(softmax_dims_mapping_dst);

  LOG_SPMD_INPUT(softmax);
  LOG_SPMD_INPUT(label);
  LOG_SPMD_INPUT(smooth_weight);
  LOG_SPMD_INPUT(loss_grad);
  LOG_SPMD_OUTPUT(x_grad);

  return {{softmax_dist_attr_dst,
           label_dist_attr_dst,
           smooth_weight_dist_attr_dst,
           loss_grad_dist_attr_dst},
          {x_grad}};
}
}  // namespace phi::distributed

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

#include "paddle/phi/infermeta/spmd_rules/cross_entropy_with_softmax.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
void GetCrossEntropyNotations(int x_ndim,
                              int axis,
                              bool soft_label,
                              bool use_softmax,
                              std::string* x_axes,
                              std::string* label_axes,
                              std::string* loss_axes,
                              std::string* softmax_out_axes) {
  std::string alphabet =
      "abcdefghijlmnopqrstuvwxyz";  // k for softmax_normalize axis
  *x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  (*x_axes)[axis] = 'k';
  *label_axes = *x_axes;
  if (!soft_label) {
    (*label_axes)[axis] = '1';
  }
  *loss_axes = *x_axes;
  (*loss_axes)[axis] = '1';
  // optional output
  if (use_softmax) {
    *softmax_out_axes = *x_axes;
  } else {
    *softmax_out_axes = "";
  }
}

SpmdInfo CrossEntropyWithSoftmaxInferSpmd(const DistMetaTensor& x,
                                          const DistMetaTensor& label,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode,
                                          int ignore_index,
                                          int axis) {
  // Step0: Verify input args based on cross_entropy_with_softmax logic
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping_src = x_dist_attr_src.dims_mapping();
  auto label_shape = phi::vectorize(label.dims());
  TensorDistAttr label_dist_attr_src = label.dist_attr();
  std::vector<int64_t> label_dims_mapping_src =
      label_dist_attr_src.dims_mapping();

  VLOG(6) << "CrossEntropyWithSoftmaxSPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping_src: ["
          << str_join(x_dims_mapping_src) << "]; Label shape: ["
          << str_join(label_shape) << "], Label dims mapping: ["
          << str_join(label_dims_mapping_src) << "]; axis: "
          << "[" << axis << "], ignore_index: [" << ignore_index
          << "], numeric_stable_mode: [" << numeric_stable_mode
          << "], use_softmax: [" << use_softmax << "], soft_label: ["
          << soft_label << "].";

  // normalize axis
  if (axis < 0) {
    axis = x_ndim + axis;
  }

  // trying to shard the normal axis of softmax, BUT
  // c_softmax_with_entropy kernel not support:
  // 1. soft label
  // 2. axis != -1
  // support above two features in future.
  if (x_dims_mapping_src[axis] > -1) {
    PADDLE_ENFORCE_EQ(
        soft_label,
        false,
        phi::errors::InvalidArgument(
            "Trying to shard the softmax_normalize axis of the input tensor, "
            "but the soft_label is set as True, which is not supported yet!"));

    PADDLE_ENFORCE_EQ(
        axis,
        x_ndim - 1,
        phi::errors::InvalidArgument(
            "Trying to shard the softmax_normalize axis of the input tensor, "
            "but the softmax_normalize axis is not the last axis, which is not "
            "supported yet! The softmax_normalize is [%d].",
            axis));

    PADDLE_ENFORCE_EQ(use_softmax,
                      true,
                      phi::errors::InvalidArgument(
                          "Trying to shard the softmax_normalize axis of the "
                          "input tensor, use_softmax must be set to True !"));
  }

  // Step1: Build Einsum Notation
  std::string x_axes, label_axes, loss_axes, softmax_out_axes;
  GetCrossEntropyNotations(x_ndim,
                           axis,
                           soft_label,
                           use_softmax,
                           &x_axes,
                           &label_axes,
                           &loss_axes,
                           &softmax_out_axes);

  // Step2: Sharding Propogation
  // Step2.1: merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping_src}, {label_axes, label_dims_mapping_src}});

  // Step2.2: infer output dims mappings
  TensorDistAttr loss_dist_attr_dst =
      CopyTensorDistAttrForOutput(label_dist_attr_src);
  loss_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(loss_axes, axis_to_dim_map));
  TensorDistAttr softmax_out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  softmax_out_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(softmax_out_axes, axis_to_dim_map));

  // Step2.3: update input dims mappings with merged one
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  TensorDistAttr label_dist_attr_dst =
      CopyTensorDistAttrForOutput(label_dist_attr_src);
  label_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(label_axes, axis_to_dim_map));

  VLOG(4) << "CrossEntropyInferSpmd:";
  VLOG(4) << "axis: [" << axis << "], ignore_index: [" << ignore_index
          << "], numeric_stable_mode: ["
          << (numeric_stable_mode ? "true" : "false") << "], use_softmax: ["
          << (use_softmax ? "true" : "false") << "], soft_label: ["
          << (soft_label ? "true" : "false") << "].";
  VLOG(4) << "Einsum notation: [" << x_axes << "," << label_axes << " --> "
          << softmax_out_axes << "," << loss_axes << "].\n"
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping_src: ["
          << str_join(x_dims_mapping_src) << "], x_dims_mapping_dst: ["
          << str_join(x_dist_attr_dst.dims_mapping()) << "]\n Label shape: ["
          << str_join(label_shape) << "], label_dims_mapping_src: ["
          << str_join(label_dims_mapping_src) << "], label_dims_mapping_dst: ["
          << str_join(label_dist_attr_dst.dims_mapping())
          << "]\nLoss dims_mapping: ["
          << str_join(loss_dist_attr_dst.dims_mapping())
          << "]\nSoftmaxOut dims_mapping: ["
          << str_join(softmax_out_dist_attr_dst.dims_mapping()) << "]\n\n";

  // todo if softmax_normalize axis is sharded, notify downstream phi api to
  // select c_softmax_with_entropy_kernel.

  // according to the phi api implemetation, the softmax_out tensor will alway
  // be genereated not matter the value of use_softmax.
  return {{x_dist_attr_dst, label_dist_attr_dst},
          {softmax_out_dist_attr_dst, loss_dist_attr_dst}};
}

SpmdInfo CrossEntropyWithSoftmaxInferSpmdReverse(
    const DistMetaTensor& x,
    const DistMetaTensor& label,
    const DistMetaTensor& softmax_out,
    const DistMetaTensor& loss,
    bool soft_label,
    bool use_softmax,
    bool numeric_stable_mode,
    int ignore_index,
    int axis) {
  // Step0: Verify input args based on cross_entropy_with_softmax logic
  auto loss_shape = phi::vectorize(loss.dims());
  int loss_ndim = loss_shape.size();
  TensorDistAttr loss_dist_attr_src = loss.dist_attr();
  std::vector<int64_t> loss_dims_mapping_src =
      loss_dist_attr_src.dims_mapping();
  auto s_out_shape = phi::vectorize(softmax_out.dims());
  int s_out_ndim = s_out_shape.size();
  TensorDistAttr s_out_dist_attr_src = softmax_out.dist_attr();
  std::vector<int64_t> s_out_dims_mapping_src =
      s_out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      loss_ndim,
      loss_dims_mapping_src.size(),
      phi::errors::InvalidArgument(
          "CrossEntropyReverse, The Tensor Loss's rank [%d] and Loss's "
          "dims_mapping size [%d] are not matched.",
          loss_ndim,
          loss_dims_mapping_src.size()));
  PADDLE_ENFORCE_EQ(
      s_out_ndim,
      s_out_dims_mapping_src.size(),
      phi::errors::InvalidArgument(
          "CrossEntropyReverse, The Tensor SoftmaxOut's rank [%d] and "
          "SoftmaxOut's dims_mapping size [%d] are not matched.",
          s_out_ndim,
          s_out_dims_mapping_src.size()));

  // Step1: Build Einsum Notation
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  auto label_shape = phi::vectorize(label.dims());

  // normalize axis
  if (axis < 0) {
    axis = x_ndim + axis;
  }

  std::string x_axes, label_axes, loss_axes, softmax_out_axes;
  GetCrossEntropyNotations(x_ndim,
                           axis,
                           soft_label,
                           use_softmax,
                           &x_axes,
                           &label_axes,
                           &loss_axes,
                           &softmax_out_axes);

  // Step2: Sharding Propogation
  // Step2.1 merge output dims mappings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{loss_axes, loss_dims_mapping_src},
                               {softmax_out_axes, s_out_dims_mapping_src}});

  // Step2.2 infer inputs' dims mappings from merged dims mapping
  std::vector<int64_t> x_dims_mapping, label_dims_mapping;
  // infer and X's dims mapping
  x_dims_mapping = GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  // infer and label's dims mapping
  label_dims_mapping = GetDimsMappingForAxes(label_axes, axis_to_dim_map);

  // Step2.3 update outputs' dims mappings with merged dims mapping
  std::vector<int64_t> s_out_dims_mapping_dst =
      GetDimsMappingForAxes(softmax_out_axes, axis_to_dim_map);
  std::vector<int64_t> loss_dims_mapping_dst =
      GetDimsMappingForAxes(loss_axes, axis_to_dim_map);

  VLOG(2) << "axis: [" << axis << "], ignore_index: [" << ignore_index
          << "], numeric_stable_mode: ["
          << (numeric_stable_mode ? "true" : "false") << "], use_softmax: ["
          << (use_softmax ? "true" : "false") << "], soft_label: ["
          << (soft_label ? "true" : "false") << "].";
  VLOG(2) << "x_dims_mapping infered:[" << str_join(x_dims_mapping) << "]";
  // in some cases, the softmax_norm axis cannot be sharded
  if (x_dims_mapping[axis] > -1) {
    if (!use_softmax) {
      x_dims_mapping[axis] = -1;
    } else {
      if (axis != x_ndim - 1) {
        x_dims_mapping[axis] = -1;
        s_out_dims_mapping_dst[axis] = -1;
        label_dims_mapping[axis] = -1;
      } else if (soft_label) {
        x_dims_mapping[axis] = -1;
        s_out_dims_mapping_dst[axis] = -1;
        label_dims_mapping[axis] = -1;
      }
    }
    VLOG(2) << "x_dims_mapping modified:[" << str_join(x_dims_mapping) << "]";
  }

  TensorDistAttr x_dist_attr = x.dist_attr();
  x_dist_attr.set_dims_mapping(x_dims_mapping);
  TensorDistAttr label_dist_attr = label.dist_attr();
  label_dist_attr.set_dims_mapping(label_dims_mapping);
  TensorDistAttr loss_dist_attr_dst(loss_dist_attr_src);
  loss_dist_attr_dst.set_dims_mapping(loss_dims_mapping_dst);
  TensorDistAttr s_out_dist_attr_dst(s_out_dist_attr_src);
  s_out_dist_attr_dst.set_dims_mapping(s_out_dims_mapping_dst);

  // step3: Handle partial state (TODO)

  VLOG(4) << "CrossEntropyInferSpmdReverse:";
  VLOG(4) << "axis: [" << axis << "], ignore_index: [" << ignore_index
          << "], numeric_stable_mode: ["
          << (numeric_stable_mode ? "true" : "false") << "], use_softmax: ["
          << (use_softmax ? "true" : "false") << "], soft_label: ["
          << (soft_label ? "true" : "false") << "].";
  VLOG(4) << "Einsum notation: [" << x_axes << "," << label_axes << " --> "
          << softmax_out_axes << "," << loss_axes << "].\n"
          << "Loss shape: [" << str_join(loss_shape)
          << "], loss_dims_mapping_src: [" << str_join(loss_dims_mapping_src)
          << "], loss_dims_mapping_dst: [" << str_join(loss_dims_mapping_dst)
          << "]\nSoftmaxOut shape: [" << str_join(s_out_shape)
          << "], softmaxout_dims_mapping_src: ["
          << str_join(s_out_dims_mapping_src)
          << "], softmaxout_dims_mapping_dst: ["
          << str_join(s_out_dims_mapping_dst) << "]\nX dims_mapping: ["
          << str_join(x_dims_mapping) << "]\nLabel dims_mapping: ["
          << str_join(label_dims_mapping) << "]\n\n";

  // according to the phi api implemetation, the softmax_out tensor will alway
  // be genereated not matter the value of use_softmax.
  return {{x_dist_attr, label_dist_attr},
          {s_out_dist_attr_dst, loss_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi

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

#include "paddle/phi/infermeta/spmd_rules/moe_gate_dispatch.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo MoEGateDispatchFwdInferSpmd(const DistMetaTensor& x,
                                     const DistMetaTensor& gate_logits,
                                     int64_t k,
                                     int64_t capacity,
                                     bool use_pad) {
  /*
  inputs:
    x: [S, H], S = b*s
    gate_logits: [S, E]
  outputs:
    y: [E, C, H] is use_pad is true, else [S, K, H], currently only support
  use_pad=true combine_weights: [S, K] scatter_index: [K, S] expert_offset: [E]
    expert_id: [S, K]
  */
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(gate_logits);

  // do some check
  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      2,
      errors::InvalidArgument(
          "x should be a 2-D tensor, but got x_shape.size() == %d",
          x_shape.size()));
  PADDLE_ENFORCE_EQ(
      gate_logits_shape.size(),
      2,
      errors::InvalidArgument("gate_logits should be a 2-D tensor, but "
                              "got gate_logits_shape.size() == %d",
                              gate_logits_shape.size()));
  // infer axes dims_mapping
  std::string x_axes = "sh";
  std::string gate_logits_axes = "se";

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping_src},
           {gate_logits_axes, gate_logits_dims_mapping_src}});
  axis_to_dim_map["k"] = -1;  // not allowed dim k to be sharded

  // input axes
  std::vector<int64_t> x_dims_mapping_dst =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<int64_t> gate_logits_dims_mapping_dst =
      GetDimsMappingForAxes(gate_logits_axes, axis_to_dim_map);
  // infer input dist attr
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  TensorDistAttr gate_logits_dist_attr_dst =
      CopyTensorDistAttrForOutput(gate_logits_dist_attr_src);
  gate_logits_dist_attr_dst.set_dims_mapping(gate_logits_dims_mapping_dst);

  // output axes
  std::string y_axes = "esh";
  std::vector<int64_t> y_dims_mapping =
      GetDimsMappingForAxes(y_axes, axis_to_dim_map);

  std::string combine_weights_axes = "sk";
  std::vector<int64_t> combine_weights_dims_mapping =
      GetDimsMappingForAxes(combine_weights_axes, axis_to_dim_map);

  std::string scatter_index_axes = "ks";
  std::vector<int64_t> scatter_index_dims_mapping =
      GetDimsMappingForAxes(scatter_index_axes, axis_to_dim_map);
  std::string expert_offset_axes = "e";
  std::vector<int64_t> expert_offset_dims_mapping =
      GetDimsMappingForAxes(expert_offset_axes, axis_to_dim_map);
  std::string expert_id_axes = "sk";
  std::vector<int64_t> expert_id_dims_mapping =
      GetDimsMappingForAxes(expert_id_axes, axis_to_dim_map);
  // infer output dist attr
  TensorDistAttr y_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  y_dist_attr_dst.set_dims_mapping(y_dims_mapping);
  TensorDistAttr combine_weights_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  combine_weights_dist_attr.set_dims_mapping(combine_weights_dims_mapping);
  TensorDistAttr scatter_index_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  scatter_index_dist_attr.set_dims_mapping(scatter_index_dims_mapping);
  TensorDistAttr expert_offset_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  expert_offset_dist_attr.set_dims_mapping(expert_offset_dims_mapping);
  TensorDistAttr expert_id_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  expert_id_dist_attr.set_dims_mapping(expert_id_dims_mapping);
  return {{x_dist_attr_dst, gate_logits_dist_attr_dst},
          {y_dist_attr_dst,
           combine_weights_dist_attr,
           scatter_index_dist_attr,
           expert_offset_dist_attr,
           expert_id_dist_attr}};
}

SpmdInfo MoEGateDispatchBwdInferSpmd(const DistMetaTensor& combine_weights,
                                     const DistMetaTensor& scatter_index,
                                     const DistMetaTensor& expert_id,
                                     const DistMetaTensor& grad_y,
                                     const DistMetaTensor& grad_combine_weights,
                                     int64_t k,
                                     int64_t capacity,
                                     bool use_pad) {
  /*
    inputs:
      combine_weights: [S, K]
      scatter_index: [K, S]
      expert_id: [S, K]
      grad_y: [E, C, H] is use_pad is true, else [S, K, H], currently only
    support use_pad=true grad_combine_weights: [S, K] outputs: grad_x: [S, H]
      grad_gate_logits: [S, E]
   */
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(combine_weights);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(scatter_index);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(expert_id);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(grad_y);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(grad_combine_weights);
  // do some check
  PADDLE_ENFORCE_EQ(
      combine_weights_shape.size(),
      2,
      errors::InvalidArgument("combine_weights should be a 2-D tensor, but "
                              "got combine_weights_shape.size() == %d",
                              combine_weights_shape.size()));
  PADDLE_ENFORCE_EQ(
      scatter_index_shape.size(),
      2,
      errors::InvalidArgument("scatter_index should be a 2-D tensor, but "
                              "got scatter_index_shape.size() == %d",
                              scatter_index_shape.size()));
  PADDLE_ENFORCE_EQ(
      expert_id_shape.size(),
      2,
      errors::InvalidArgument("expert_id should be a 2-D tensor, but "
                              "got expert_id_shape.size() == %d",
                              expert_id_shape.size()));
  PADDLE_ENFORCE_EQ(
      grad_y_shape.size(),
      3,
      errors::InvalidArgument("grad_y should be a 3-D tensor, but "
                              "got grad_y_shape.size() == %d",
                              grad_y_shape.size()));
  PADDLE_ENFORCE_EQ(grad_combine_weights_shape.size(),
                    2,
                    errors::InvalidArgument(
                        "grad_combine_weights should be a 2-D tensor, but "
                        "got grad_combine_weights_shape.size() == %d",
                        grad_combine_weights_shape.size()));

  // infer axes dims_mapping
  std::string combine_weights_axes = "sk";
  std::string scatter_index_axes = "ks";
  std::string expert_id_axes = "sk";
  std::string grad_y_axes = "esh";
  std::string grad_combine_weights_axes = "sk";
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{combine_weights_axes, combine_weights_dims_mapping_src},
           {scatter_index_axes, scatter_index_dims_mapping_src},
           {expert_id_axes, expert_id_dims_mapping_src},
           {grad_y_axes, grad_y_dims_mapping_src},
           {grad_combine_weights_axes, grad_combine_weights_dims_mapping_src}});
  // axis_to_dim_map["e"] = -1;  // not allowed dim e to be sharded
  // input axes
  std::vector<int64_t> combine_weights_dims_mapping_dst =
      GetDimsMappingForAxes(combine_weights_axes, axis_to_dim_map);
  std::vector<int64_t> scatter_index_dims_mapping_dst =
      GetDimsMappingForAxes(scatter_index_axes, axis_to_dim_map);
  std::vector<int64_t> expert_id_dims_mapping_dst =
      GetDimsMappingForAxes(expert_id_axes, axis_to_dim_map);
  std::vector<int64_t> grad_y_dims_mapping_dst =
      GetDimsMappingForAxes(grad_y_axes, axis_to_dim_map);
  std::vector<int64_t> grad_combine_weights_dims_mapping_dst =
      GetDimsMappingForAxes(grad_combine_weights_axes, axis_to_dim_map);
  // infer input dist attr
  TensorDistAttr combine_weights_dist_attr_dst =
      CopyTensorDistAttrForOutput(combine_weights_dist_attr_src);
  combine_weights_dist_attr_dst.set_dims_mapping(
      combine_weights_dims_mapping_dst);
  TensorDistAttr scatter_index_dist_attr_dst =
      CopyTensorDistAttrForOutput(scatter_index_dist_attr_src);
  scatter_index_dist_attr_dst.set_dims_mapping(scatter_index_dims_mapping_dst);

  TensorDistAttr expert_id_dist_attr_dst =
      CopyTensorDistAttrForOutput(expert_id_dist_attr_src);
  expert_id_dist_attr_dst.set_dims_mapping(expert_id_dims_mapping_dst);
  TensorDistAttr grad_y_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_y_dist_attr_src);
  grad_y_dist_attr_dst.set_dims_mapping(grad_y_dims_mapping_dst);
  TensorDistAttr grad_combine_weights_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_combine_weights_dist_attr_src);
  grad_combine_weights_dist_attr_dst.set_dims_mapping(
      grad_combine_weights_dims_mapping_dst);

  // output axes
  std::string grad_x_axes = "sh";
  std::string grad_gate_logits = "se";
  std::vector<int64_t> grad_x_dims_mapping =
      GetDimsMappingForAxes(grad_x_axes, axis_to_dim_map);
  std::vector<int64_t> grad_gate_logits_dims_mapping =
      GetDimsMappingForAxes(grad_gate_logits, axis_to_dim_map);
  // output dist attr
  TensorDistAttr grad_x_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_y_dist_attr_src);
  grad_x_dist_attr_dst.set_dims_mapping(grad_x_dims_mapping);
  TensorDistAttr grad_gate_logits_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_y_dist_attr_src);
  grad_gate_logits_dist_attr_dst.set_dims_mapping(
      grad_gate_logits_dims_mapping);
  return {{combine_weights_dist_attr_dst,
           scatter_index_dist_attr_dst,
           expert_id_dist_attr_dst,
           grad_y_dist_attr_dst,
           grad_combine_weights_dist_attr_dst},
          {grad_x_dist_attr_dst, grad_gate_logits_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi

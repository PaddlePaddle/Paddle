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

#include "paddle/phi/infermeta/spmd_rules/slice.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

std::vector<int64_t> BuildOutputAxisToInputAxisMap(
    const std::vector<int64_t>& decrease_axis, int input_ndim) {
  std::vector<int64_t> output_axis_to_input_axis(input_ndim -
                                                 decrease_axis.size());
  int index = 0;
  for (int i = 0; i < input_ndim; ++i) {
    if (std::find(decrease_axis.begin(), decrease_axis.end(), i) ==
        decrease_axis.end()) {
      output_axis_to_input_axis[index] = i;
      ++index;
    }
  }
  return output_axis_to_input_axis;
}

SpmdInfo SliceInferSpmdBase(const DistMetaTensor& input,
                            const std::vector<int64_t>& axes,
                            const std::vector<int64_t>& decrease_axis,
                            const std::vector<int>& starts,
                            const std::vector<int>& ends,
                            const std::vector<int>& strides) {
  // Step0: Verify input args based on slice logic
  auto input_shape = common::vectorize(input.dims());
  int input_ndim = static_cast<int>(input_shape.size());
  int output_ndim = input_ndim - static_cast<int>(decrease_axis.size());
  auto input_dist_attr_src = input.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(input_ndim,
                    input_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        input_ndim,
                        input_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

  // get einsum notation for input
  std::string input_axes = alphabet.substr(0, input_ndim);

  auto output_input_axis_mapping =
      BuildOutputAxisToInputAxisMap(decrease_axis, input_ndim);
  // get einsum notation for output
  std::string out_axes = alphabet.substr(0, output_ndim);
  for (int i = 0; i < output_ndim; i++) {
    auto input_axis = output_input_axis_mapping[i];
    out_axes[i] = input_axes[input_axis];
  }

  // Step2.3 get new dist attribute for input. the sliced
  // cannot be sharded, if it is sharded, set it to replicated.
  std::vector<int64_t> input_process_mesh =
      input_dist_attr_src.process_mesh().shape();
  TensorDistAttr input_dist_attr_dst =
      CopyTensorDistAttrForOutput(input_dist_attr_src);
  for (auto axe : axes) {
    int axis = axe < 0 ? axe + input_ndim : axe;
    if (!(axis == (input_ndim - 1) && strides.size() == 1 &&
          (input_shape[axis] / input_process_mesh[input_dims_mapping[axis]]) %
                  strides[0] ==
              0)) {
      input_dims_mapping[axis] = -1;
    }
  }
  input_dist_attr_dst.set_dims_mapping(input_dims_mapping);

  std::vector<int64_t> out_dims_mapping(output_ndim);
  for (int i = 0; i < output_ndim; i++) {
    auto input_axis = output_input_axis_mapping[i];
    out_dims_mapping[i] = input_dims_mapping[input_axis];
  }
  TensorDistAttr out_dist_attr =
      CopyTensorDistAttrForOutput(input_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "SliceInferSpmd:";
  VLOG(4) << "Einsum Notation: " << input_axes << "-->" << out_axes;
  VLOG(4) << "Input shape: [" << str_join(input_shape) << "] "
          << "axes: [" << str_join(axes) << "] "
          << "src_dims_mapping: ["
          << str_join(input_dist_attr_src.dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(input_dims_mapping) << "]";
  VLOG(4) << "Output"
          << " dims_mapping: [" << str_join(out_dims_mapping) << "]";
  VLOG(4) << std::endl;

  return {{input_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo SliceInferSpmd(const DistMetaTensor& input,
                        const std::vector<int64_t>& axes,
                        const std::vector<int>& starts,
                        const std::vector<int>& ends,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis) {
  // starts, ends, infer_flags and decrease_axis have no impact on the
  // derivation, only to align with the definition in phi api
  return SliceInferSpmdBase(input, axes, {}, starts, ends, {});
}

SpmdInfo SliceInferSpmdReverseBase(const DistMetaTensor& input,
                                   const DistMetaTensor& output,
                                   const std::vector<int64_t>& axes,
                                   const std::vector<int64_t>& decrease_axis,
                                   const std::vector<int>& starts,
                                   const std::vector<int>& ends,
                                   const std::vector<int>& strides) {
  auto output_shape = common::vectorize(output.dims());
  int out_ndim = output_shape.size();
  auto out_dist_attr = output.dist_attr();
  int out_dims_mapping_size =
      static_cast<int>(out_dist_attr.dims_mapping().size());
  auto input_shape = common::vectorize(input.dims());
  int input_ndim = static_cast<int>(input_shape.size());
  auto input_dist_attr = input.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr.dims_mapping();

  int decrease_axis_num = decrease_axis.size();

  PADDLE_ENFORCE_EQ(input_ndim,
                    out_ndim + decrease_axis_num,
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] is not equal "
                        "to the Tensor Output's rank [%d]",
                        input_ndim,
                        out_ndim));

  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor Output's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping_size));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

  // get einsum notation for input
  std::string input_axes = alphabet.substr(0, input_ndim);

  auto output_input_axis_mapping =
      BuildOutputAxisToInputAxisMap(decrease_axis, input_ndim);
  // get einsum notation for output
  std::string out_axes = alphabet.substr(0, out_ndim);
  for (int i = 0; i < out_ndim; i++) {
    auto input_axis = output_input_axis_mapping[i];
    out_axes[i] = input_axes[input_axis];
  }

  std::vector<int64_t> input_process_mesh =
      input_dist_attr.process_mesh().shape();
  for (auto axe : axes) {
    int axis = axe < 0 ? axe + input_ndim : axe;
    // the sliced axis cannot be sharded, set its notation
    // with the special '1' to set its dim mapping to -1.
    if (!(axis == (input_ndim - 1) && strides.size() == 1 &&
          (input_shape[axis] / input_process_mesh[input_dims_mapping[axis]]) %
                  strides[0] ==
              0)) {
      input_axes[axis] = '1';
    }
  }

  // Step2: Sharding Propagation
  // Step2.1: merge output shardings
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  std::vector<int64_t> out_dims_mapping = output.dist_attr().dims_mapping();
  axes_sharding_info.emplace_back(out_axes, out_dims_mapping);

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // Step2.2: infer input dims mapping from output dims mapping. the sliced
  // cannot be sharded, if it is sharded, set it to replicated.
  input_dims_mapping = GetDimsMappingForAxes(input_axes, axis_to_dim_map, true);

  auto input_dist_attr_dst = CopyTensorDistAttrForOutput(input_dist_attr);
  input_dist_attr_dst.set_dims_mapping(input_dims_mapping);

  // step2.3 get new dist attribute for output. the sliced
  // cannot be sharded, if it is sharded, set it to replicated.
  out_dims_mapping = GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
  std::vector<int64_t> output_process_mesh =
      out_dist_attr.process_mesh().shape();
  for (auto axe : axes) {
    int axis = axe < 0 ? axe + input_ndim : axe;
    if (!(axis == (out_ndim - 1) && strides.size() == 1 &&
          (output_shape[axis] / output_process_mesh[out_dims_mapping[axis]]) %
                  strides[0] ==
              0)) {
      out_dims_mapping[axis] = -1;
    }
  }
  auto out_dist_attr_dst = CopyTensorDistAttrForOutput(out_dist_attr);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "SliceInferSpmdReverse:";
  VLOG(4) << "Einsum Notation: " << input_axes << "-->" << out_axes;
  VLOG(4) << "Output"
          << " shape: [" << str_join(common::vectorize(output.dims())) << "] "
          << "axes: [" << str_join(axes) << "] "
          << "src_dims_mapping: ["
          << str_join(output.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(out_dist_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Input shape: [" << str_join(input_shape) << "] "
          << "dims_mapping: [" << str_join(input_dims_mapping) << "]\n\n";

  return {{input_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo SliceInferSpmdReverse(const DistMetaTensor& input,
                               const DistMetaTensor& output,
                               const std::vector<int64_t>& axes,
                               const std::vector<int>& starts,
                               const std::vector<int>& ends,
                               const std::vector<int64_t>& infer_flags,
                               const std::vector<int64_t>& decrease_axis) {
  // starts, ends, infer_flags and decrease_axis have no impact on the
  // derivation, only to align with the definition in phi api
  return SliceInferSpmdReverseBase(input, output, axes, {}, starts, ends, {});
}

SpmdInfo SliceInferSpmdDynamic(const DistMetaTensor& input,
                               const std::vector<int64_t>& axes,
                               const IntArray& starts,
                               const IntArray& ends,
                               const std::vector<int64_t>& infer_flags,
                               const std::vector<int64_t>& decrease_axis) {
  // starts, ends, infer_flags and decrease_axis have no impact on the
  // derivation, only to align with the definition in phi api
  std::vector<int> start_indexes(starts.GetData().begin(),
                                 starts.GetData().end());
  std::vector<int> end_indexes(ends.GetData().begin(), ends.GetData().end());
  return SliceInferSpmdBase(input, axes, decrease_axis, {}, {}, {});
}

SpmdInfo ViewSliceInferSpmd(const DistMetaTensor& input,
                            int64_t begin_idx,
                            int64_t end_idx) {
  auto input_dist_attr_src = input.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr_src.dims_mapping();
  input_dims_mapping[0] = -1;
  input_dist_attr_src.set_dims_mapping(input_dims_mapping);
  return {{input_dist_attr_src}, {input_dist_attr_src}};
}

SpmdInfo SliceGradInferBase(const DistMetaTensor& input,
                            const DistMetaTensor& out_grad,
                            const std::vector<int64_t>& axes,
                            const std::vector<int64_t>& decrease_axis) {
  // Step0: Verify input args based on slice logic
  auto input_shape = common::vectorize(input.dims());
  int input_ndim = static_cast<int>(input_shape.size());
  auto input_dist_attr = input.dist_attr();

  input_dist_attr = UnShardTensorDims(input_dist_attr, axes);
  std::vector<int64_t> input_dims_mapping = input_dist_attr.dims_mapping();

  auto output_axis_to_input_axis_mapping =
      BuildOutputAxisToInputAxisMap(decrease_axis, input_ndim);
  std::unordered_map<int, int> reverse_output_axis_to_input_axis_mapping;
  for (size_t i = 0; i < output_axis_to_input_axis_mapping.size(); ++i) {
    reverse_output_axis_to_input_axis_mapping
        [output_axis_to_input_axis_mapping[i]] = i;
  }
  std::vector<int64_t> mapped_axes;
  for (const auto& axe : axes) {
    int axis = axe < 0 ? axe + input_ndim : axe;
    if (reverse_output_axis_to_input_axis_mapping.count(axis) > 0) {
      mapped_axes.push_back(reverse_output_axis_to_input_axis_mapping[axis]);
    }
  }
  auto out_dist_attr = out_grad.dist_attr();
  out_dist_attr = UnShardTensorDims(out_dist_attr, mapped_axes);
  auto output_shape = common::vectorize(out_grad.dims());
  int out_ndim = output_shape.size();
  int out_dims_mapping_size =
      static_cast<int>(out_dist_attr.dims_mapping().size());
  int decrease_axis_num = decrease_axis.size();

  PADDLE_ENFORCE_EQ(input_ndim,
                    out_ndim + decrease_axis_num,
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] is not equal "
                        "to the Tensor Output's rank [%d]",
                        input_ndim,
                        out_ndim));

  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor Output's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping_size));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

  // get einsum notation for input
  std::string input_axes = alphabet.substr(0, input_ndim);

  // get einsum notation for output
  std::string out_axes(out_ndim, ' ');
  for (int i = 0; i < out_ndim; ++i) {
    out_axes[i] = input_axes[output_axis_to_input_axis_mapping[i]];
  }

  // Step2: Sharding Propagation
  // Step2.1: merge input shardings
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(out_axes, out_dist_attr.dims_mapping());
  axes_sharding_info.emplace_back(input_axes, input_dist_attr.dims_mapping());
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // Step2.2: infer output dims mapping from merged input dims mapping
  auto input_dim_mapping_dst =
      GetDimsMappingForAxes(input_axes, axis_to_dim_map, true);

  // get the dist attributes for output
  TensorDistAttr input_grad = CopyTensorDistAttrForOutput(input_dist_attr);
  input_dist_attr.set_dims_mapping(input_dim_mapping_dst);
  input_grad.set_dims_mapping(input_dim_mapping_dst);

  auto out_grad_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
  out_dist_attr.set_dims_mapping(out_grad_dims_mapping);

  VLOG(4) << "SliceGradInfer:";

  VLOG(4) << "input"
          << " shape: [" << str_join(input_shape) << "] "
          << "src_dims_mapping: [" << str_join(input.dist_attr().dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(input_dist_attr.dims_mapping())
          << "]";

  VLOG(4) << "Output Grad"
          << " shape: [" << str_join(output_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(out_grad.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";

  VLOG(4) << "input Grad"
          << " shape: [" << str_join(output_shape) << "] "
          << "dims_mapping: [" << str_join(input_grad.dims_mapping()) << "] ";

  return {{input_dist_attr, out_dist_attr}, {input_grad}};
}

SpmdInfo SliceGradInferSpmdDynamic(const DistMetaTensor& input,
                                   const DistMetaTensor& out_grad,
                                   const std::vector<int64_t>& axes,
                                   const IntArray& starts,
                                   const IntArray& ends,
                                   const std::vector<int64_t>& infer_flags,
                                   const std::vector<int64_t>& decrease_axis) {
  // starts, ends, infer_flags and decrease_axis have no impact on the
  // derivation, only to align with the definition in phi api
  return SliceGradInferBase(input, out_grad, axes, decrease_axis);
}

SpmdInfo StridedSliceInferSpmd(const DistMetaTensor& input,
                               const std::vector<int>& axes,
                               const std::vector<int>& starts,
                               const std::vector<int>& ends,
                               const std::vector<int>& strides) {
  // starts, ends and strides have no impact on the derivation,
  // only to align with the definition in phi api
  std::vector<int64_t> axes_bridge(axes.begin(), axes.end());
  return SliceInferSpmdBase(input, axes_bridge, {}, starts, ends, strides);
}

SpmdInfo StridedSliceGradInferSpmd(const DistMetaTensor& input,
                                   const DistMetaTensor& out_grad,
                                   const std::vector<int>& axes,
                                   const std::vector<int>& starts,
                                   const std::vector<int>& ends,
                                   const std::vector<int>& strides) {
  // starts, ends and strides have no impact on the derivation,
  // only to align with the definition in phi api
  std::vector<int64_t> axes_bridge(axes.begin(), axes.end());
  return SliceGradInferBase(input, out_grad, axes_bridge, {});
}

SpmdInfo StridedSliceInferSpmdDynamic(const DistMetaTensor& input,
                                      const std::vector<int>& axes,
                                      const IntArray& starts,
                                      const IntArray& ends,
                                      const IntArray& strides) {
  // starts, ends and strides have no impact on the derivation,
  // only to align with the definition in phi api
  std::vector<int64_t> axes_bridge(axes.begin(), axes.end());
  return SliceInferSpmdBase(input, axes_bridge, {}, {}, {}, {});
}

SpmdInfo StridedSliceGradInferSpmdDynamic(const DistMetaTensor& input,
                                          const DistMetaTensor& out_grad,
                                          const std::vector<int>& axes,
                                          const IntArray& starts,
                                          const IntArray& ends,
                                          const IntArray& strides) {
  // starts, ends and strides have no impact on the derivation,
  // only to align with the definition in phi api
  std::vector<int64_t> axes_bridge(axes.begin(), axes.end());
  return SliceGradInferBase(input, out_grad, axes_bridge, {});
}

}  // namespace phi::distributed

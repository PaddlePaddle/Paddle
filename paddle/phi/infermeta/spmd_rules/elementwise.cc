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

#include "paddle/phi/infermeta/spmd_rules/elementwise.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
std::string GetInputBroadcastNotation(const std::vector<int64_t>& shape,
                                      const int max_ndim,
                                      const std::string& alphabet,
                                      std::vector<int>* broadcast_axis_count) {
  int ndim = static_cast<int>(shape.size());
  int start_dim = max_ndim - ndim;
  std::string axes_notation = GetBroadcastAxes(ndim, max_ndim, alphabet);

  for (int idim = 0; idim < max_ndim; idim++) {
    // deal with the broadcast axes, record the
    // input number at each broadcast axis
    if (idim < start_dim) {
      (*broadcast_axis_count)[idim] += 1;
    } else if (shape[idim - start_dim] == 1) {
      (*broadcast_axis_count)[idim] += 1;
      // mark the broadcast axis to a special "1"
      axes_notation[idim - start_dim] = '1';
    }
  }
  return axes_notation;
}

void GetBinaryNotations(const std::vector<int64_t>& x_shape,
                        const std::vector<int64_t>& y_shape,
                        std::string* x_axes,
                        std::string* y_axes,
                        std::string* out_axes) {
  int x_ndim = static_cast<int>(x_shape.size());
  int y_ndim = static_cast<int>(y_shape.size());
  int max_ndim = std::max(x_ndim, y_ndim);
  int ninputs = 2;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::vector<int> input_ndims({x_ndim, y_ndim});

  // get einsum notation for each input, deal with broadcast
  std::vector<int> broadcast_axis_count(max_ndim, 0);
  *x_axes = GetInputBroadcastNotation(
      x_shape, max_ndim, alphabet, &broadcast_axis_count);
  *y_axes = GetInputBroadcastNotation(
      y_shape, max_ndim, alphabet, &broadcast_axis_count);

  // get einsum notation for output
  *out_axes = GetBroadcastAxes(max_ndim, max_ndim, alphabet);
  for (int64_t idim = 0; idim < max_ndim; idim++) {
    // if all inputs broadcast at this dimension,
    // mark this axis in output as broadcast
    if (broadcast_axis_count[idim] == ninputs) {
      (*out_axes)[idim] = '1';
    }
  }
}

SpmdInfo ElementwiseUnaryInferSpmd(const DistMetaTensor& x) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "ElementwiseUnary, The Tensor X's rank [%d] and X's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::pair<std::string, std::vector<int64_t>> axes_sharding_info(
      x_axes, x_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({axes_sharding_info});

  // step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  TensorDistAttr x_dst_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dst_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "ElementwiseSPMDRule InferForward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] ";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  return {{x_dst_dist_attr}, {out_dist_attr}};
}

SpmdInfo AssignInferSpmd(const DistMetaTensor& x) {
  return {{x.dist_attr()}, {x.dist_attr()}};
}

// NOTE(lizhiyu): This function is only for `cast` right now to support partial
// propagation
SpmdInfo ElementwiseUnaryWithPartialInferSpmd(const DistMetaTensor& x) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "ElementwiseUnary, The Tensor X's rank [%d] and X's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::pair<std::string, std::vector<int64_t>> axes_sharding_info(
      x_axes, x_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({axes_sharding_info});

  // step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  out_dist_attr.set_partial_status(x_dist_attr_src.partial_status());
  TensorDistAttr x_dst_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dst_dist_attr.set_dims_mapping(out_dims_mapping);
  x_dst_dist_attr.set_partial_status(x_dist_attr_src.partial_status());

  VLOG(4) << "ElementwiseWithPartialSPMDRule InferForward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] ";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  return {{x_dst_dist_attr}, {out_dist_attr}};
}

SpmdInfo ElementwiseUnaryInferSpmdReverse(const DistMetaTensor& x,
                                          const DistMetaTensor& out) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto out_shape = common::vectorize(out.dims());
  int out_ndim = static_cast<int>(out_shape.size());
  TensorDistAttr out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument(
          "ElementwiseUnaryReverse, The Tensor Out's rank [%d] and X's "
          "dims_mapping size [%d] are not matched.",
          out_ndim,
          out_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      out_ndim,
      x_ndim,
      common::errors::InvalidArgument(
          "ElementwiseUnaryReverse, The Tensor Out's rank [%d] and X's "
          "rank [%d] are not matched.",
          out_ndim,
          x_ndim));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: Merge output shardings
  std::pair<std::string, std::vector<int64_t>> axes_sharding_info(
      out_axes, out_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({axes_sharding_info});

  // step2.2: Infer input dims mapping from merged input dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  auto x_dist_attr = CopyTensorDistAttrForOutput(out_dist_attr_src);
  x_dist_attr.set_dims_mapping(x_dims_mapping);
  auto out_dist_attr_dst = CopyTensorDistAttrForOutput(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  // Step3: Handle partial
  // Handle output tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferReverse:";
  VLOG(4) << "Output0 shape: [" << str_join(out_shape) << "] "
          << "dims_mapping: [" << str_join(out_dims_mapping) << "] ";
  VLOG(4) << "Input0 dims_mapping: [" + str_join(x_dims_mapping) + "]\n\n";

  return {{x_dist_attr}, {out_dist_attr_dst}};
}

SpmdInfo ElementwiseBinaryInferSpmd(const DistMetaTensor& x,
                                    const DistMetaTensor& y) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto y_shape = common::vectorize(y.dims());
  int y_ndim = static_cast<int>(y_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  TensorDistAttr y_dist_attr_src = y.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "ElementwiseBinary, The Tensor X's rank [%d] and X's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(y_ndim,
                    y_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "ElementwiseBinary, The Tensor Y's rank [%d] and Y's "
                        "dims_mapping size [%d] are not matched.",
                        y_ndim,
                        y_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string x_axes, y_axes, out_axes;
  GetBinaryNotations(x_shape, y_shape, &x_axes, &y_axes, &out_axes);

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping}, {y_axes, y_dims_mapping}});

  // Step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step2.3: Update inputs' dims mapping with merged one.
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr y_dist_attr_dst = CopyTensorDistAttrForOutput(y_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  y_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(y_axes, axis_to_dim_map));

  // Step3: Handle partial
  // Handle input tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferForward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Input1 shape: [" << str_join(y_shape) << "] "
          << "src_dims_mapping: [" << str_join(y_dims_mapping) << "] "
          << "dst_dims_mapping: [" << str_join(y_dist_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  return {{x_dist_attr_dst, y_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo ElementwiseBinaryInferSpmdReverse(const DistMetaTensor& x,
                                           const DistMetaTensor& y,
                                           const DistMetaTensor& out) {
  // Step0: Verify Input Args Based on Elementwise Logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto y_shape = common::vectorize(y.dims());
  int y_ndim = static_cast<int>(y_shape.size());
  auto out_shape = common::vectorize(out.dims());
  int out_ndim = static_cast<int>(out_shape.size());
  int max_ndim = std::max(x_ndim, y_ndim);
  TensorDistAttr out_dist_attr = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument(
          "ElementwiseBinaryReverse, The Tensor Out's rank [%d] and Out's "
          "dims_mapping size [%d] are not matched.",
          out_ndim,
          out_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      out_ndim,
      max_ndim,
      common::errors::InvalidArgument(
          "ElementwiseBinaryReverse, The Tensor Out's rank [%d] and the "
          "max rank of inputs [%d] are not matched.",
          out_ndim,
          max_ndim));

  // Step1: Build Einsum Notation
  std::string x_axes, y_axes, out_axes;
  GetBinaryNotations(x_shape, y_shape, &x_axes, &y_axes, &out_axes);

  // Step2: Sharding Propagation
  // Step2.1: Merge output shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}});

  // Step2.2: Infer input dims mappings from merged output dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x.dist_attr());
  TensorDistAttr y_dist_attr_dst = CopyTensorDistAttrForOutput(y.dist_attr());
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<int64_t> y_dims_mapping =
      GetDimsMappingForAxes(y_axes, axis_to_dim_map);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  y_dist_attr_dst.set_dims_mapping(y_dims_mapping);

  auto out_dist_attr_dst = CopyTensorDistAttrForOutput(out_dist_attr);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Step3: Handle partial
  // Handle input tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferReverse:";
  VLOG(4) << "Output shape: [" << str_join(out_shape) << "] dims_mapping: ["
          << str_join(out_dims_mapping) << "]";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Input1 shape: [" << str_join(y_shape) << "] "
          << "dims_mapping: [" << str_join(y_dims_mapping) << "]\n\n";

  return {{x_dist_attr_dst, y_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo ElementwiseUnaryGradInferSpmd(const DistMetaTensor& x,
                                       const DistMetaTensor& out_grad) {
  auto dist_attr = CopyTensorDistAttrForOutput(out_grad.dist_attr());
  dist_attr.set_dims_mapping(out_grad.dist_attr().dims_mapping());
  return {{dist_attr, dist_attr}, {dist_attr}};
}

SpmdInfo ElementwiseUnaryGradInferSpmd(const DistMetaTensor& x,
                                       const DistMetaTensor& out,
                                       const DistMetaTensor& out_grad) {
  auto dist_attr = CopyTensorDistAttrForOutput(out_grad.dist_attr());
  dist_attr.set_dims_mapping(out_grad.dist_attr().dims_mapping());
  return {{dist_attr, dist_attr, dist_attr}, {dist_attr}};
}

bool DimsNotEqualOrHasBroadcastDim(const DistMetaTensor& x,
                                   const DistMetaTensor& out) {
  if (x.dims() != out.dims()) {
    return true;
  }

  // Now the dims of x must equal to out.
  const auto& out_dims_mapping = out.dist_attr().dims_mapping();
  for (int64_t i = x.dims().size(); i >= 0; --i) {
    if ((x.dims()[i] == 1) && (out_dims_mapping[i] != -1)) {
      return true;
    }
  }
  return false;
}

std::vector<int64_t> GetExplicitReduceDim(const DistMetaTensor& x,
                                          const DistMetaTensor& out) {
  std::vector<int64_t> reduce_dims;
  const auto& out_dims_mapping = out.dist_attr().dims_mapping();
  int64_t diff = out.dims().size() - x.dims().size();

  for (int64_t i = x.dims().size(); i >= 0; --i) {
    if ((x.dims()[i] == 1) && (out_dims_mapping[i + diff] != -1)) {
      reduce_dims.emplace_back(i);
    }
  }

  return reduce_dims;
}

SpmdInfo ElementwiseBinaryGradInferSpmd(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& out_grad,
                                        int64_t axis) {
  TensorDistAttr out_grad_dist_attr = out_grad.dist_attr();
  out_grad_dist_attr.clean_partial_status();
  TensorDistAttr x_dist_attr = out_grad_dist_attr;
  TensorDistAttr y_dist_attr = out_grad_dist_attr;
  TensorDistAttr x_grad_dist_attr = out_grad_dist_attr;
  TensorDistAttr y_grad_dist_attr = out_grad_dist_attr;

  PADDLE_ENFORCE_GE(out_grad.dims().size(),
                    x.dims().size(),
                    common::errors::InvalidArgument(
                        "If being broadcast, the dims of out_grad "
                        "must larger or equal to the inputs."
                        "But we get the rank of output as [%d] and "
                        "the rank of input as [%d].",
                        out_grad.dims().size(),
                        x.dims().size()));

  PADDLE_ENFORCE_GE(out_grad.dims().size(),
                    y.dims().size(),
                    common::errors::InvalidArgument(
                        "If being broadcast, the dims of out_grad "
                        "must larger or equal to the inputs."
                        "But we get the rank of output as [%d] and "
                        "the rank of input as [%d].",
                        out_grad.dims().size(),
                        y.dims().size()));
  // The backward rule of elementwise follows the principle: the dist_attr
  // of input should equal to out_grad.
  // Caution the special case when the inputs calculate together with different
  // shape it means one of the input is broadcast to same shape with the other
  // first. When doing backward the input_grad with broadcast input is in
  // partial status, which need to do communicate and get the right result.
  if (DimsNotEqualOrHasBroadcastDim(x, out_grad)) {
    VLOG(3) << "We need to do some special operations with the dist attr of "
               "input x. "
            << "The global dim of input x is " << x.dims()
            << ". The global dim of out_grad is " << out_grad.dims();
    // Step 1: remove the useless dimensions which is not appear in input x.
    int64_t diff = out_grad.dims().size() - x.dims().size();
    auto dims_mapping = out_grad_dist_attr.dims_mapping();
    dims_mapping.erase(dims_mapping.begin(), dims_mapping.begin() + diff);
    // Step 2: get the explicit reduce dimensions
    std::vector<int64_t> explicit_reduce_dims =
        GetExplicitReduceDim(x, out_grad);
    VLOG(4) << "The explicit reduce dims has " << explicit_reduce_dims.size()
            << " elements.";
    for (const auto& dim : explicit_reduce_dims) {
      VLOG(4) << "Explicit reduce dims is " << dim;
      dims_mapping[dim] = -1;
    }
    x_dist_attr.set_dims_mapping(dims_mapping);
    x_dist_attr.set_default_dynamic_dims(dims_mapping);
    x_grad_dist_attr.set_dims_mapping(dims_mapping);
    x_grad_dist_attr.set_default_dynamic_dims(dims_mapping);
    // Step 3: set partial dimension
    for (int64_t i = 0; i < diff; ++i) {
      if (out_grad.dist_attr().dims_mapping()[i] != -1) {
        x_grad_dist_attr.set_partial_status(
            std::vector<int64_t>{out_grad.dist_attr().dims_mapping()[i]});
      }
    }
    for (const auto& dim : explicit_reduce_dims) {
      x_grad_dist_attr.set_partial_status(std::vector<int64_t>{
          out_grad.dist_attr().dims_mapping()[diff + dim]});
    }
  }

  if (DimsNotEqualOrHasBroadcastDim(y, out_grad)) {
    VLOG(3) << "We need to do some special operations with the dist attr of "
               "input y. "
            << "The global dim of input y is " << y.dims()
            << ". The global dim of out_grad is " << out_grad.dims();
    // Step 1: remove the useless dimensions which is not appear in input y.
    int64_t diff = out_grad.dims().size() - y.dims().size();
    auto dims_mapping = out_grad_dist_attr.dims_mapping();
    dims_mapping.erase(dims_mapping.begin(), dims_mapping.begin() + diff);
    // Step 2: get the explicit reduce dimensions
    std::vector<int64_t> explicit_reduce_dims =
        GetExplicitReduceDim(y, out_grad);
    VLOG(4) << "The explicit reduce dims has " << explicit_reduce_dims.size()
            << " elements.";
    for (const auto& dim : explicit_reduce_dims) {
      VLOG(4) << "Explicit reduce dims is " << dim;
      dims_mapping[dim] = -1;
    }
    y_dist_attr.set_dims_mapping(dims_mapping);
    y_dist_attr.set_default_dynamic_dims(dims_mapping);
    y_grad_dist_attr.set_dims_mapping(dims_mapping);
    y_grad_dist_attr.set_default_dynamic_dims(dims_mapping);
    // Step 3: set partial dimension
    for (int64_t i = 0; i < diff; ++i) {
      if (out_grad.dist_attr().dims_mapping()[i] != -1) {
        y_grad_dist_attr.set_partial_status(
            std::vector<int64_t>{out_grad.dist_attr().dims_mapping()[i]});
      }
    }
    for (const auto& dim : explicit_reduce_dims) {
      y_grad_dist_attr.set_partial_status(std::vector<int64_t>{
          out_grad.dist_attr().dims_mapping()[diff + dim]});
    }
  }

  return {{x_dist_attr, y_dist_attr, out_grad_dist_attr},
          {x_grad_dist_attr, y_grad_dist_attr}};
}

SpmdInfo ElementwiseBinaryGradInferSpmd(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& out,
                                        const DistMetaTensor& out_grad,
                                        int64_t axis) {
  // The out's dist_attr is the same with out_grad's dist_attr, reuse
  // ElementwiseBinaryGradInferSpmd(x, y, out_grad, axis) to infer dist_attrs of
  // {{x, y, out_grad}, {x_grad, y_grad}}, then insert out's dist_attr into it.
  SpmdInfo info = ElementwiseBinaryGradInferSpmd(x, y, out_grad, axis);
  info.first.emplace(info.first.begin() + 2, out_grad.dist_attr());
  return info;
}
}  // namespace phi::distributed

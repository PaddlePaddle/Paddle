// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/infermeta/spmd_rules/unsqueeze.h"
#include <algorithm>
#include <numeric>

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/dim_trans.h"
#include "paddle/phi/infermeta/spmd_rules/reshape.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

TensorDistAttr CreateUnsqueezeXshape(const TensorDistAttr& x) {
  TensorDistAttr out(x);
  auto dims_mapping = x.dims_mapping();
  dims_mapping.insert(dims_mapping.begin(), -1);
  out.set_dims_mapping(dims_mapping);
  return out;
}

std::vector<std::shared_ptr<DimTrans>> MakeUnsqueezeDimTrans(
    const std::vector<int64_t>& x_shape,
    std::vector<int64_t>* out_shape,
    const std::vector<int64_t>& axis) {
  int64_t n = static_cast<int64_t>(x_shape.size() + axis.size());
  std::vector<std::shared_ptr<DimTrans>> ret;
  ret.resize(n);
  out_shape->resize(n);
  fill(ret.begin(), ret.end(), std::make_shared<Singleton>());
  fill(out_shape->begin(), out_shape->end(), 1);

  for (int64_t i = 0, j = 0; i < n; i++) {
    auto it = find(axis.begin(), axis.end(), i);

    if (it == axis.end()) {
      if (x_shape[j] != 1) {
        ret[i] = std::make_shared<InputDim>(j);
        (*out_shape)[i] = x_shape[j];
      }

      j++;
    }
  }

  return ret;
}

std::vector<std::shared_ptr<DimTrans>> MakeUnsqueezeDimTransReverse(
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& axis,
    const int& x_ndim,
    const int& out_ndim) {
  std::vector<std::shared_ptr<DimTrans>> ret;
  ret.resize(x_ndim);
  fill(ret.begin(), ret.end(), std::make_shared<Singleton>());

  for (int64_t i = 0, j = 0; i < out_ndim; i++) {  // NOLINT
    auto it = find(axis.begin(), axis.end(), i);

    if (it == axis.end()) {
      if (out_shape[i] != 1) {
        ret[j] = std::make_shared<InputDim>(i);
      }

      j++;
    }
  }

  return ret;
}

SpmdInfo UnsqueezeInferSpmd(const DistMetaTensor& x,
                            const std::vector<int64_t>& axis) {
  // Step0: Verify input args based on unsqueeze logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  const auto& x_dist_attr_src = x.dist_attr();
  const std::vector<int64_t>& x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  // Step1: Build the transformation from
  // the original shape to the target shape

  std::vector<int64_t> out_shape;
  std::vector<int64_t> axis_copy(axis);

  for (auto& i : axis_copy) {
    if (i < 0) {
      i += x_ndim + 1;
    }
  }

  std::vector<std::shared_ptr<DimTrans>> trans =
      MakeUnsqueezeDimTrans(x_shape, &out_shape, axis_copy);

  // Step2: Infer the dims mapping of input (if reshard is
  // needed) and output from the dimension transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(x, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping.
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(dims_mapping_vec[0]);
  if (x_dist_attr_dst.dynamic_dims().size() !=
      x_dist_attr_dst.dims_mapping().size()) {
    VLOG(4) << "UnSqueezeInferSPMD change output dist attr dynamic dims";
    x_dist_attr_dst.set_default_dynamic_dims(x_dist_attr_dst.dims_mapping());
  }
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(dims_mapping_vec[1]);
  if (out_dist_attr.dynamic_dims().size() !=
      out_dist_attr.dims_mapping().size()) {
    VLOG(4) << "UnSqueezeInferSPMD change output dist attr dynamic dims";
    out_dist_attr.set_default_dynamic_dims(out_dist_attr.dims_mapping());
  }

  VLOG(4) << "UnsqueezeInferSpmd: X shape: [" << str_join(x_shape)
          << "] Out shape: [" << str_join(out_shape) << "]";
  VLOG(4) << "Transformation from input to output:";
  for (int64_t i = 0, n = static_cast<int64_t>(trans.size()); i < n; i++) {
    std::shared_ptr<DimTrans> t = trans[i];
    VLOG(4) << "\tOut axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "X dims_mapping_src: [" << str_join(x_dims_mapping)
          << "] dims_mapping_dst: [" << str_join(dims_mapping_vec[0])
          << "]\n Out dims_mapping: [" << str_join(dims_mapping_vec[1])
          << "]\n\n";

  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo UnsqueezeInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const std::vector<int64_t>& axis) {
  // Step0: Verify input args based on unsqueeze logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto out_shape = common::vectorize(out.dims());
  int out_ndim = static_cast<int>(out_shape.size());
  const auto& out_dist_attr_src = out.dist_attr();
  const std::vector<int64_t>& out_dims_mapping =
      out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));

  // Step1: Build the transformation from the output shape
  // to original shape. This function infers the dims mapping
  // from output to input, we first get the transformation
  // from output to input so that we can infer the dims mapping
  // with the map from output axes to input axes.

  std::vector<int64_t> axis_copy(axis);

  for (auto& i : axis_copy) {
    if (i < 0) {
      i += x_ndim + 1;
    }
  }

  std::vector<std::shared_ptr<DimTrans>> trans =
      MakeUnsqueezeDimTransReverse(out_shape, axis_copy, x_ndim, out_ndim);

  // Step2: Infer the dims mapping of input with
  // output's dims_mapping and the transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(out, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(dims_mapping_vec[0]);
  if (out_dist_attr_dst.dynamic_dims().size() !=
      out_dist_attr_dst.dims_mapping().size()) {
    VLOG(4) << "UnSqueezeInferSPMDReverse change output dist attr dynamic dims";
    out_dist_attr_dst.set_default_dynamic_dims(
        out_dist_attr_dst.dims_mapping());
  }
  TensorDistAttr x_dist_attr = CopyTensorDistAttrForOutput(x.dist_attr());
  x_dist_attr.set_dims_mapping(dims_mapping_vec[1]);
  if (x_dist_attr.dynamic_dims().size() != x_dist_attr.dims_mapping().size()) {
    VLOG(4) << "UnSqueezeInferSPMDReverse change x dist attr dynamic dims";
    x_dist_attr.set_default_dynamic_dims(x_dist_attr.dims_mapping());
  }
  VLOG(4) << "UnsqueezeInferSpmdReverse: Out shape: [" << str_join(out_shape)
          << "] X shape: [" << str_join(x_shape) << "]";
  VLOG(4) << "Transformation from output to input:";
  for (int64_t i = 0, n = static_cast<int64_t>(trans.size()); i < n; i++) {
    std::shared_ptr<DimTrans> t = trans[i];
    VLOG(4) << "\tX axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "Out dims_mapping_src: [" << str_join(out_dims_mapping) << "] "
          << "dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "X dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  return {{x_dist_attr}, {out_dist_attr_dst}};
}

SpmdInfo UnsqueezeGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& out_grad,
                                const IntArray& axis) {
  auto shape = phi::vectorize(x.dims());
  const auto& spmd = ReshapeInferSpmd(out_grad, shape);
  return {{x.dist_attr(), spmd.first[0]}, {spmd.second[0]}};
}

SpmdInfo UnsqueezeWithXShapeInferSpmd(const DistMetaTensor& x,
                                      const std::vector<int64_t>& axis) {
  const auto& spmd = UnsqueezeInferSpmd(x, axis);
  const auto& x_dist_attr = PADDLE_GET_CONST(TensorDistAttr, spmd.first[0]);
  return {{x_dist_attr}, {spmd.second[0], CreateUnsqueezeXshape(x_dist_attr)}};
}

SpmdInfo UnsqueezeWithXShapeGradInferSpmd(const DistMetaTensor& xshape,
                                          const DistMetaTensor& out_grad,
                                          const IntArray& axis) {
  auto shape = phi::vectorize(xshape.dims());
  shape = std::vector<int64_t>(shape.begin() + 1, shape.end());
  const auto& spmd = ReshapeInferSpmd(out_grad, shape);
  return {{xshape.dist_attr(), spmd.first[0]}, {spmd.second[0]}};
}

}  // namespace phi::distributed

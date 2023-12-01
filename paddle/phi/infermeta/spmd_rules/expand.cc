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
#include "paddle/phi/infermeta/spmd_rules/expand.h"
#include <numeric>

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/dim_trans.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

std::vector<std::shared_ptr<DimTrans>> PadLeftDim(int64_t ndim,
                                                  int64_t min_ndims) {
  int64_t num_padding = std::max((int64_t)0, min_ndims - ndim);
  std::vector<std::shared_ptr<DimTrans>> padded_dim_trans(min_ndims, nullptr);
  for (int64_t i = 0; i < min_ndims; i++) {
    if (i < num_padding) {
      padded_dim_trans[i] = std::make_shared<Singleton>();
    } else {
      padded_dim_trans[i] = std::make_shared<InputDim>(i - num_padding);
    }
  }
  return padded_dim_trans;
}

// Compute how each dimension in target shape
// is obtained from the input dimensions
std::vector<std::shared_ptr<DimTrans>> MakeExpandDimTrans(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& tgt_shape) {
  int64_t tgt_shape_ndim = tgt_shape.size();
  int64_t src_shape_ndim = src_shape.size();

  PADDLE_ENFORCE_GE(
      tgt_shape_ndim,
      src_shape_ndim,
      phi::errors::InvalidArgument(
          "In dimension expanding, the target shape %d must not be less than"
          "the source shape %d.",
          tgt_shape_ndim,
          src_shape_ndim));

  std::vector<std::shared_ptr<DimTrans>> padded_input =
      PadLeftDim(src_shape_ndim, tgt_shape_ndim);

  std::vector<std::shared_ptr<DimTrans>> ret;
  for (int64_t i = 0, n = static_cast<int64_t>(padded_input.size()); i < n;
       i++) {
    std::shared_ptr<DimTrans> inp = padded_input[i];
    int64_t tgt_dim_val = tgt_shape[i];
    if (inp->type() == DimTrans::Type::SINGLETON) {
      PADDLE_ENFORCE_GT(
          tgt_dim_val,
          0,
          "The value of tgt_shape[%d] %d must not be less than zero.",
          i,
          tgt_dim_val);
      if (tgt_dim_val == 1) {
        ret.emplace_back(inp);
      } else {
        ret.emplace_back(std::make_shared<Broadcast>(inp, tgt_dim_val));
      }

    } else if (inp->type() == DimTrans::Type::INPUTDIM) {
      std::shared_ptr<InputDim> inputdim =
          std::dynamic_pointer_cast<InputDim>(inp);
      int64_t inp_dim_val = src_shape[inputdim->input_dim()];
      if (inp_dim_val == 1 && tgt_dim_val != inp_dim_val) {
        ret.emplace_back(std::make_shared<Broadcast>(inp, tgt_dim_val));
      } else {
        ret.emplace_back(inp);
      }

    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "DimTrans in padded_input must be either Singleton or InputDim"));
    }
  }

  return ret;
}

std::vector<std::shared_ptr<DimTrans>> MakeExpandDimTransReverse(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& out_shape) {
  int64_t x_ndim = x_shape.size();
  int64_t out_ndim = out_shape.size();

  PADDLE_ENFORCE_GE(
      out_ndim,
      x_ndim,
      phi::errors::InvalidArgument(
          "In dimension expanding, the target shape %d must not be less than"
          "the source shape %d.",
          out_ndim,
          x_ndim));

  int64_t num_padding = out_ndim - x_ndim;
  std::vector<std::shared_ptr<DimTrans>> ret(x_ndim, nullptr);

  for (int64_t i = 0; i < x_ndim; i++) {
    ret[i] = std::make_shared<InputDim>(num_padding + i);
  }

  return ret;
}

SpmdInfo ExpandInferSpmd(const DistMetaTensor& x,
                         const std::vector<int64_t>& shape) {
  // Step0: Verify input args based on expand logic
  VLOG(2) << "Debug Info for expand";
  VLOG(2) << "shape: " << str_join(shape);
  auto x_shape = phi::vectorize(x.dims());
  int x_ndim = x_shape.size();
  int out_ndim = shape.size();
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                   "dims_mapping size [%d] are not matched.",
                                   x_ndim,
                                   x_dims_mapping.size()));
  PADDLE_ENFORCE_LE(
      x_ndim,
      out_ndim,
      phi::errors::InvalidArgument(
          "The Tensor X's ndim %d must not be larger than out_ndim %d.",
          x_ndim,
          out_ndim));

  VLOG(4) << "ExpandInferSpmd: X shape: [" << str_join(x_shape) << "]";
  VLOG(4) << "Out shape: [" << str_join(shape) << "]";

  // Step1: Build the transformation from
  // the original shape to the target shape

  // handle the case of dynamic shape ? like [-1, -1, ...] ?
  // TODO(MarioLulab): to discuss

  // handle the '-1' value in target shape, where '-1' indicates the target
  // shape is equal to the source shape
  std::vector<int64_t> tgt_shape(shape);
  for (int64_t i = x_ndim - 1, tgt_idx = tgt_shape.size() - 1;
       i >= 0 && tgt_idx >= 0;
       i--, tgt_idx--) {
    if (tgt_shape[tgt_idx] == -1) {
      tgt_shape[tgt_idx] = x_shape[i];
    }
  }

  std::vector<std::shared_ptr<DimTrans>> trans =
      MakeExpandDimTrans(x_shape, tgt_shape);

  // Step2: Infer the dims mapping of input (if reshard is
  // needed) and output from the dimension transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(x, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping.
  TensorDistAttr x_dist_attr_dst(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(dims_mapping_vec[0]);
  TensorDistAttr out_dist_attr(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(dims_mapping_vec[1]);

  VLOG(4) << "Transformation from input to output:";
  for (int64_t i = 0, n = static_cast<int64_t>(trans.size()); i < n; i++) {
    std::shared_ptr<DimTrans> t = trans[i];
    VLOG(4) << "\tOut axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "X dims_mapping_src: [" << str_join(x_dims_mapping)
          << "] dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "Out dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo ExpandInferSpmdReverse(const DistMetaTensor& x,
                                const DistMetaTensor& out,
                                const std::vector<int64_t>& shape) {
  VLOG(2) << "Debug Info for expand";
  VLOG(2) << "shape: " << str_join(shape);
  auto x_shape = phi::vectorize(x.dims());
  auto out_shape = phi::vectorize(out.dims());
  int x_ndim = x_shape.size();
  int out_ndim = shape.size();
  auto out_dist_attr_src = out.dist_attr();

  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                   "dims_mapping size [%d] are not matched.",
                                   out_ndim,
                                   out_dims_mapping.size()));

  PADDLE_ENFORCE_LE(
      x_ndim,
      out_ndim,
      phi::errors::InvalidArgument(
          "The Tensor X's ndim %d must not be larger than out_ndim %d.",
          x_ndim,
          out_ndim));

  VLOG(4) << "ExpandInferSpmdReverse: Out shape: [" << str_join(out_shape)
          << "], X shape: [" << str_join(x_shape) << "]";

  // Step1: Build the transformation from the output shape
  // to original shape. This function infers the dims mapping
  // from output to input, we first get the transformation
  // from output to input so that we can infer the dims mapping
  // with the map from output axes to input axes.

  // handle the case of dynamic shape ? like [-1, -1, ...] ?
  // TODO(MarioLulab): to discuss

  // FIXME(MarioLulab): Do we really need it ?
  // handle the '-1' value in target shape, where '-1' indicates the target
  // shape is equal to the source shape
  for (int64_t i = x_ndim - 1, tgt_idx = out_ndim - 1; i >= 0 && tgt_idx >= 0;
       i--, tgt_idx--) {
    if (shape[tgt_idx] == -1) {
      out_shape[tgt_idx] = x_shape[i];
    }
  }

  std::vector<std::shared_ptr<DimTrans>> trans =
      MakeExpandDimTransReverse(out_shape, x_shape);
  // Step2: Infer the dims mapping of input with
  // output's dims_mapping and the transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(out, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping
  TensorDistAttr out_dist_attr_dst(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(dims_mapping_vec[0]);
  TensorDistAttr x_dist_attr(x.dist_attr());
  x_dist_attr.set_dims_mapping(dims_mapping_vec[1]);

  VLOG(4) << "Transformation from output to input:";
  for (int64_t i = 0, n = trans.size(); i < n; i++) {
    std::shared_ptr<DimTrans> t = trans[i];
    VLOG(4) << "\tX axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "Out dims_mapping_src: [" << str_join(out_dims_mapping) << "] "
          << "dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "X dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  return {{x_dist_attr}, {out_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi

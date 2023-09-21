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

#include "paddle/phi/infermeta/spmd_rules/reshape.h"
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

// The target shape in reshape may contains a -1 dimension,
// this function is used to infer what the "-1" dimension is.
std::vector<int64_t> InferTargetShape(const std::vector<int64_t>& shape,
                                      int64_t len) {
  int64_t infer_idx = -1;
  for (int64_t i = 0, n = static_cast<int64_t>(shape.size()); i < n; i++) {
    if (shape[i] == -1) {
      PADDLE_ENFORCE_EQ(
          infer_idx,
          -1,
          phi::errors::InvalidArgument(
              "There can't be more than one -1 dimension in target shape."));
      infer_idx = i;
    }
  }

  int64_t product = std::accumulate(
      shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  if (product > 0) {
    PADDLE_ENFORCE_EQ(
        product,
        len,
        phi::errors::InvalidArgument("The total size are not matched"));
    return std::vector<int64_t>(shape);
  } else {
    std::vector<int64_t> new_shape(shape);
    product = -product;
    int64_t infer_size = len / product;
    PADDLE_ENFORCE_EQ(len % infer_size,
                      0,
                      phi::errors::InvalidArgument(
                          "The total is not diviable by infer_size"));
    new_shape[infer_idx] = infer_size;
    return new_shape;
  }
}

// Compute how each dimension in target shape
// is obtained from the input dimensions
std::vector<DimTrans*> MakeReshapeDimTrans(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& tgt_shape) {
  std::vector<DimTrans*> ret;
  int64_t total_elem_num_src = std::accumulate(
      src_shape.begin(), src_shape.end(), 1, std::multiplies<int64_t>());
  std::vector<int64_t> inferred_tgt_shape =
      InferTargetShape(tgt_shape, total_elem_num_src);

  int src_idx = 0, tgt_idx = 0;
  int s, t;
  int src_len, tgt_len;
  src_len = static_cast<int64_t>(src_shape.size());
  tgt_len = static_cast<int64_t>(inferred_tgt_shape.size());
  while (src_idx < src_len || tgt_idx < tgt_len) {
    std::vector<int64_t> src_dims, tgt_splitted_shape;
    if (src_idx >= src_len) {
      s = 1;
    } else {
      s = src_shape[src_idx];
      src_dims.emplace_back(src_idx);
      src_idx++;
    }
    if (tgt_idx >= tgt_len) {
      t = 1;
    } else {
      t = inferred_tgt_shape[tgt_idx];
      tgt_splitted_shape.emplace_back(t);
      tgt_idx++;
    }

    // deal with the singleton case
    if (s == 1 && t != 1) {
      // case [1] [a]
      tgt_idx--;
      tgt_splitted_shape.clear();
    } else if (s != 1 && t == 1) {
      src_idx--;
      src_dims.clear();
    } else {
      while (s != t) {
        if (s < t) {
          src_dims.emplace_back(src_idx);
          s *= src_shape[src_idx];
          src_idx++;
        } else {
          tgt_splitted_shape.emplace_back(inferred_tgt_shape[tgt_idx]);
          t *= inferred_tgt_shape[tgt_idx];
          tgt_idx++;
        }
      }
    }

    if (tgt_splitted_shape.size() > 0) {
      std::vector<DimTrans*> input_dims;
      for (int i = 0, n = static_cast<int>(src_dims.size()); i < n; i++) {
        int64_t in_dim = src_dims[i];
        if (src_shape[in_dim] > 1) {
          input_dims.emplace_back(new InputDim(in_dim));
        }
      }
      DimTrans* flatten = make_flatten(input_dims);

      for (int64_t i = 0, n = static_cast<int64_t>(tgt_splitted_shape.size());
           i < n;
           i++) {
        ret.emplace_back(make_split(flatten, tgt_splitted_shape, i));
      }
    }
  }
  return ret;
}

SpmdInfo ReshapeInferSpmd(const DistMetaTensor& x,
                          const std::vector<int64_t>& shape) {
  // Step0: Verify input args based on reshape logic
  auto src_shape = phi::vectorize(x.dims());
  int x_ndim = src_shape.size();
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                   "dims_mapping size [%d] are not matched.",
                                   x_ndim,
                                   x_dims_mapping.size()));

  // Step1: Build the transformation from
  // the original shape to the target shape

  // handle the '0' values in target shape, '0' indicates
  // that the target shape is equal to the source shape
  std::vector<int64_t> tgt_shape(shape);
  for (int64_t i = 0, n = static_cast<int64_t>(tgt_shape.size()); i < n; i++) {
    if (tgt_shape[i] == 0) {
      tgt_shape[i] = src_shape[i];
    }
  }

  std::vector<DimTrans*> trans = MakeReshapeDimTrans(src_shape, tgt_shape);

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

  VLOG(4) << "ReshapeInferSpmd: X shape: [" << str_join(src_shape)
          << "] Out shape: [" << str_join(tgt_shape) << "]";
  VLOG(4) << "Transformation from input to output:";
  for (int64_t i = 0, n = static_cast<int64_t>(trans.size()); i < n; i++) {
    DimTrans* t = trans[i];
    VLOG(4) << "\tOut axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "X dims_mapping_src: [" << str_join(x_dims_mapping)
          << "] dims_mapping_dst: [" << str_join(dims_mapping_vec[0])
          << "]\n Out dims_mapping: [" << str_join(dims_mapping_vec[1])
          << "]\n\n";

  CleanUp();

  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo ReshapeInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& out,
                                 const std::vector<int64_t>& shape) {
  // Step0: Verify input args based on reshape logic
  auto x_shape = phi::vectorize(x.dims());
  auto out_shape = phi::vectorize(out.dims());
  int out_ndim = out_shape.size();
  auto out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                   "dims_mapping size [%d] are not matched.",
                                   out_ndim,
                                   out_dims_mapping.size()));

  // Step1: Build the transformation from the output shape
  // to original shape. This function infers the dims mapping
  // from output to input, we first get the transformation
  // from output to input so that we can infer the dims mapping
  // with the map from output axes to input axes.
  // Shapes in InferSpmdReverse don't contain -1 or 0, so they will
  // not be modified and we can directly use them.
  std::vector<DimTrans*> trans = MakeReshapeDimTrans(out_shape, x_shape);

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

  VLOG(4) << "ReshapeInferSpmdReverse: Out shape: [" << str_join(out_shape)
          << "] X shape: [" << str_join(x_shape) << "]";
  VLOG(4) << "Transformation from output to input:";
  for (int64_t i = 0, n = trans.size(); i < n; i++) {
    DimTrans* t = trans[i];
    VLOG(4) << "\tX axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "Out dims_mapping_src: [" << str_join(out_dims_mapping) << "] "
          << "dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "X dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  CleanUp();

  return {{x_dist_attr}, {out_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi

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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/reshape_spmd_rule.h"
#include <numeric>
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dim_trans.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using phi::distributed::auto_parallel::str_join;

// The target shape in reshape may contains a -1 dimension,
// this function is used to infer what the "-1" dimension is.
std::vector<int64_t> InferTargetShape(const std::vector<int64_t>& shape,
                                      int64_t len) {
  int64_t infer_idx = -1;
  for (int64_t i = 0, n = shape.size(); i < n; i++) {
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

  int64_t src_idx = 0, tgt_idx = 0;
  int64_t s, t;
  int64_t src_len, tgt_len;
  src_len = src_shape.size();
  tgt_len = inferred_tgt_shape.size();
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
      t = tgt_shape[tgt_idx];
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
      for (int64_t i = 0, n = src_dims.size(); i < n; i++) {
        int64_t in_dim = src_dims[i];
        if (src_shape[in_dim] > 1) {
          input_dims.emplace_back(new InputDim(in_dim));
        }
      }
      DimTrans* flatten = make_flatten(input_dims);

      for (int64_t i = 0, n = tgt_splitted_shape.size(); i < n; i++) {
        ret.emplace_back(make_split(flatten, tgt_splitted_shape, i));
      }
    }
  }
  return ret;
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
paddle::distributed::auto_parallel::ReshapeSPMDRule::InferForward(
    const std::vector<DistTensorSpec>& input_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: Verify Input Args Based on Reshape Logic
  int64_t ninputs = input_specs.size();
  PADDLE_ENFORCE_EQ(
      ninputs,
      1,
      phi::errors::InvalidArgument("The size of InputSpec in reshape must "
                                   "be equal to 1, but got [%d].",
                                   ninputs));
  VerifySpecs(input_specs, "reshape");

  // step1: build the transformation from
  // original shape to target shape
  std::vector<int64_t> src_shape = input_specs[0].shape();
  std::vector<int64_t> tgt_shape =
      ExtractAttr<std::vector<int64_t>>("shape", attrs);

  // handle the '0' values in target shape, '0' indicates
  // that the target shape is equal to the source shape
  for (int64_t i = 0, n = tgt_shape.size(); i < n; i++) {
    if (tgt_shape[i] == 0) {
      tgt_shape[i] = src_shape[i];
    }
  }

  std::vector<DimTrans*> trans = MakeReshapeDimTrans(src_shape, tgt_shape);

  // step2: infer the dims mapping of input (if reshard is
  // needed) and output from the dimension transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(input_specs[0], trans);

  // step3: update the dist attributes of input
  // and output with the inferred dims mapping
  TensorDistAttr new_input_dist_attr(input_specs[0].dist_attr());
  new_input_dist_attr.set_dims_mapping(dims_mapping_vec[0]);
  TensorDistAttr output_dist_attr(input_specs[0].dist_attr());
  output_dist_attr.set_dims_mapping(dims_mapping_vec[1]);

  VLOG(4) << "Reshape: input_shape: [" << str_join(src_shape)
          << "] output_shape: [" << str_join(tgt_shape) << "]";
  VLOG(4) << "Transformation from input to output:";
  for (int64_t i = 0, n = trans.size(); i < n; i++) {
    DimTrans* t = trans[i];
    VLOG(4) << "\tOutput axis " << i << ": " << t->to_string();
  }
  VLOG(4) << "input_dims_mapping: [" << str_join(dims_mapping_vec[0])
          << "] output_dims_mapping: [" << str_join(dims_mapping_vec[1])
          << "]\n\n";

  CleanUp();

  return {{new_input_dist_attr}, {output_dist_attr}};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
paddle::distributed::auto_parallel::ReshapeSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of ReductionSPMDRule is NOT implemented yet."));

  return {};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle

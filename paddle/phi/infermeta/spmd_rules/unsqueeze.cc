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

#include "paddle/phi/infermeta/spmd_rules/unsqueeze.h"
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

std::vector<DimTrans*> MakeUnsqueezeDimTrans(int64_t ndim,
                                             const IntArray& axis) {
  std::vector<DimTrans*> dims;
  for (int64_t i = 0; i < ndim; i++) {
    dims.emplace_back(new InputDim(i));
  }

  auto axes = axis.GetData();
  for (size_t i = 0; i < axes.size(); i++) {
    dims.insert(dims.begin() + (axes[i] < 0 ? axes[i] + ndim + 1 : axes[i]),
                new Singleton());
  }

  return dims;
}

std::vector<DimTrans*> MakeUnsqueezeDimTransReverse(int64_t ndim,
                                                    const IntArray& axis) {
  std::vector<DimTrans*> dims;
  for (int64_t i = 0; i < ndim; i++) {
    if (std::find(axis.GetData().begin(), axis.GetData().end(), i) !=
        axis.GetData().end()) {
      dims.emplace_back(new InputDim(i));
    }
  }
  return dims;
}

SpmdInfo UnsqueezeInferSpmd(const DistMetaTensor& x, const IntArray& axis) {
  // Step0: Verify input args based on unsqueeze logic
  PADDLE_ENFORCE_EQ(
      x.dims().size(),
      x.dist_attr().dims_mapping().size(),
      phi::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                   "dims_mapping size [%d] are not matched.",
                                   x.dims().size(),
                                   x.dist_attr().dims_mapping().size()));

  // Step1: Build the transformation from
  // the original shape to the target shape
  std::vector<DimTrans*> trans = MakeUnsqueezeDimTrans(x.dims().size(), axis);

  // Step2: Infer the dims mapping of input (if reshard is
  // needed) and output from the dimension transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(x, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping.
  TensorDistAttr x_dist_attr(x.dist_attr());
  x_dist_attr.set_dims_mapping(dims_mapping_vec[0]);
  TensorDistAttr out_dist_attr(x.dist_attr());
  out_dist_attr.set_dims_mapping(dims_mapping_vec[1]);

  VLOG(4) << "UnsqueezeInferSpmd: X shape: ["
          << str_join(phi::vectorize(x.dims())) << "]";
  VLOG(4) << "axis: [" << str_join(axis.GetData()) << "]";
  VLOG(4) << "Transformation from input to output:";
  for (int64_t i = 0, n = static_cast<int64_t>(trans.size()); i < n; i++) {
    DimTrans* t = trans[i];
    VLOG(4) << "\tOut axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "X dims_mapping_src: [" << str_join(x.dist_attr().dims_mapping())
          << "] dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "Out dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  CleanUp();

  return {{x_dist_attr}, {out_dist_attr}};
}

SpmdInfo UnsqueezeInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const IntArray& axis) {
  // Step0: Verify input args based on unsqueeze logic
  PADDLE_ENFORCE_EQ(
      out.dims().size(),
      out.dist_attr().dims_mapping().size(),
      phi::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                   "dims_mapping size [%d] are not matched.",
                                   out.dims().size(),
                                   out.dist_attr().dims_mapping().size()));

  // Step1: Build the transformation from the output shape
  // to original shape. This function infers the dims mapping
  // from output to input, we first get the transformation
  // from output to input so that we can infer the dims mapping
  // with the map from output axes to input axes.
  std::vector<DimTrans*> trans =
      MakeUnsqueezeDimTransReverse(out.dims().size(), axis);

  // Step2: Infer the dims mapping of input with
  // output's dims_mapping and the transformation.
  std::vector<std::vector<int64_t>> dims_mapping_vec =
      InferFromDimTrans(out, trans);

  // Step3: Update the dist attributes of input
  // and output with the inferred dims mapping
  TensorDistAttr out_dist_attr(out.dist_attr());
  out_dist_attr.set_dims_mapping(dims_mapping_vec[0]);
  TensorDistAttr x_dist_attr(x.dist_attr());
  x_dist_attr.set_dims_mapping(dims_mapping_vec[1]);

  VLOG(4) << "UnsqueezeInferSpmdReverse: Out shape: ["
          << str_join(phi::vectorize(out.dims())) << "] X shape: ["
          << str_join(phi::vectorize(x.dims())) << "]";
  VLOG(4) << "Transformation from output to input:";
  for (int64_t i = 0, n = trans.size(); i < n; i++) {
    DimTrans* t = trans[i];
    VLOG(4) << "\tX axis[" << i << "]: " << t->to_string();
  }
  VLOG(4) << "Out dims_mapping_src: ["
          << str_join(out.dist_attr().dims_mapping()) << "] "
          << "dims_mapping_dst: [" << str_join(dims_mapping_vec[0]) << "]";
  VLOG(4) << "X dims_mapping: [" << str_join(dims_mapping_vec[1]) << "]\n\n";

  CleanUp();

  return {{x_dist_attr}, {out_dist_attr}};
}

}  // namespace distributed
}  // namespace phi

// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/infermeta/spmd_rules/p_norm.h"
#include "glog/logging.h"

namespace phi {
namespace distributed {

SpmdInfo PNormInferSpmd(const DistMetaTensor& x,
                        float porder,
                        int axis,
                        float epsilon,
                        bool keepdims,
                        bool asvector) {
  std::vector<int64_t> new_axis;
  if (asvector) {
    auto x_shape = common::vectorize(x.dims());
    int x_ndim = static_cast<int>(x_shape.size());
    new_axis.resize(x_ndim);
    for (int i = 0; i < x_ndim; ++i) {
      new_axis[i] = i;
    }
  } else {
    new_axis.push_back(axis);
  }
  VLOG(4) << "PNormInferSpmd Call ReductionInferSpmd";
  return ReductionInferSpmd(x, new_axis, keepdims);
}

SpmdInfo PNormInferSpmdReverse(const DistMetaTensor& x,
                               const DistMetaTensor& out,
                               float porder,
                               int axis,
                               float epsilon,
                               bool keepdims,
                               bool asvector) {
  std::vector<int64_t> new_axis;
  if (asvector) {
    auto x_shape = common::vectorize(x.dims());
    int x_ndim = static_cast<int>(x_shape.size());
    new_axis.resize(x_ndim);
    for (int i = 0; i < x_ndim; ++i) {
      new_axis[i] = i;
    }
  } else {
    new_axis.push_back(axis);
  }
  VLOG(4) << "PNormInferSpmdReverse Call ReductionInferSpmdReverse";
  return ReductionInferSpmdReverse(x, out, new_axis, keepdims);
}

SpmdInfo PNormGradInferSpmd(const DistMetaTensor& x,
                            const DistMetaTensor& out,
                            const DistMetaTensor& out_grad,
                            float porder,
                            int axis,
                            float epsilon,
                            bool keepdims,
                            bool asvector) {
  std::vector<int64_t> new_axis;
  if (asvector) {
    auto x_shape = common::vectorize(x.dims());
    int x_ndim = static_cast<int>(x_shape.size());
    new_axis.resize(x_ndim);
    for (int i = 0; i < x_ndim; ++i) {
      new_axis[i] = i;
    }
  } else {
    new_axis.push_back(axis);
  }
  VLOG(4) << "PNormGradInferSpmd Call ReductionGradInferSpmd";
  return ReductionGradInferSpmd(x, out, out_grad, new_axis, keepdims, asvector);
}

}  // namespace distributed
}  // namespace phi

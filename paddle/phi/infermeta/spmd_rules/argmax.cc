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

#pragma once

#include "paddle/phi/infermeta/spmd_rules/argmax.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/reduction.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo ArgMaxInferSpmdBase(const DistMetaTensor& x,
                             int axis,
                             bool keepdims,
                             bool flatten) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  std::vector<int64_t> axis_vec;

  if (flatten) {
    axis_vec.reserve(x_ndim);
    for (int i = 0; i < x_ndim; ++i) {
      axis_vec.push_back(i);
    }
  } else {
    axis_vec.push_back(axis);
  }

  return ReductionInferSpmd(x, axis_vec, keepdims);
}

SpmdInfo ArgMaxInferSpmdReverseBase(const DistMetaTensor& x,
                                    const DistMetaTensor& out,
                                    int axis,
                                    bool keepdims,
                                    bool flatten) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  std::vector<int64_t> axis_vec;

  if (flatten) {
    axis_vec.reserve(x_ndim);
    for (int i = 0; i < x_ndim; ++i) {
      axis_vec.push_back(i);
    }
  } else {
    axis_vec.push_back(axis);
  }

  return ReductionInferSpmdReverse(x, out, axis_vec, keepdims);
}

SpmdInfo ArgMaxInferSpmdDynamic(const DistMetaTensor& x,
                                const Scalar& axis,
                                bool keepdims,
                                bool flatten,
                                DataType dtype) {
  return ArgMaxInferSpmdBase(x, axis.to<int32_t>(), keepdims, flatten);
}

}  // namespace distributed
}  // namespace phi

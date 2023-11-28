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

SpmdInfo ExpandInferSpmd(const DistMetaTensor& x, const IntArray& shape) {
  // Step0: Verify input args based on expand logic
  VLOG(2) << "Debug Info for expand";
  VLOG(2) << "shape: " << str_join(shape.GetData());
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
  VLOG(4) << "ReshapeInferSpmd: X shape: [" << str_join(x_shape) << "]";
  VLOG(4) << "Out shape: [" << str_join(shape.GetData()) << "]";

  // Step1: Build the transformation from
  // the original shape to the target shape
  // TODO(MarioLulab)
}

}  // namespace distributed
}  // namespace phi

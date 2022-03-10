// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/kernels/cpu/reduce.h"

namespace phi {

template <typename DeviceContext, typename OutT, typename Functor>

void BoolReduceKernel(const DeviceContext& context,
                      const phi::DenseTensor& input,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all,
                      phi::DenseTensor* output) {
  context.template Alloc<OutT>(output);

  // The dims has full dim, set the reduce_all is True
  const auto& input_dim_size = x.dims().size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  if (reduce_all) {
    // Flatten and reduce 1-D tensor
    auto x = EigenVector<OutT>::Flatten(input);
    auto out = EigenScalar<OutT>::From(*output);
    auto& place = context.eigen_device();

    auto reduce_dim = Eigen::array<int, 1>({{0}});
    Functor functor;
    functor(place, &x, &out, reduce_dim);
  } else {
    int ndim = input.dims().size();
    int rdim = dims.size();
    // comments for accelerating compiling temporarily.
    if (ndim > 6) {
      HandleLargeDim<DeviceContext, OutT, Functor>(
          context, input, output, dims, keep_dim);
    } else {
      HANDLE_REDUCE_DIM(6, 5);
      HANDLE_REDUCE_DIM(6, 4);
      HANDLE_REDUCE_DIM(6, 3);
      HANDLE_REDUCE_DIM(6, 2);
      HANDLE_REDUCE_DIM(6, 1);
      HANDLE_REDUCE_DIM(5, 4);
      HANDLE_REDUCE_DIM(5, 3);
      HANDLE_REDUCE_DIM(5, 2);
      HANDLE_REDUCE_DIM(5, 1);
      HANDLE_REDUCE_DIM(4, 3);
      HANDLE_REDUCE_DIM(4, 2);
      HANDLE_REDUCE_DIM(4, 1);
      HANDLE_REDUCE_DIM(3, 2);
      HANDLE_REDUCE_DIM(3, 1);
      HANDLE_REDUCE_DIM(2, 1);
      HANDLE_REDUCE_DIM(1, 1);
    }
  }
}

}  // namespace phi

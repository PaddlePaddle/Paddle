// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/core/ddim.h"
#include "paddle/pten/core/dense_tensor.h"

#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"

namespace pten {

namespace math {

template <typename DeviceContext, typename T>
struct TransposeNormal {
  // for dims >= 7 situation
  void operator()(const DeviceContext& dev_ctx,
                  const pten::DenseTensor& in,
                  pten::DenseTensor* out,
                  const std::vector<int64_t>& axis);
};

template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* out,
                  const std::vector<int>& axis) {
    Eigen::array<int, Rank> permute;
    for (int i = 0; i < Rank; i++) {
      permute[i] = axis[i];
    }
    auto eigen_in = pten::EigenTensor<T, Rank>::From(in);
    auto eigen_out = pten::EigenTensor<T, Rank>::From(*out);
    auto* dev = dev_ctx.eigen_device();
    // use 32bit index to speed up computation
    bool use_32bit_index = eigen_out.size() < Eigen::NumTraits<int>::highest();
    bool is_gpu_place = paddle::platform::is_gpu_place(dev_ctx.GetPlace());
    if (use_32bit_index && is_gpu_place) {
      To32BitIndex(eigen_out).device(*dev) =
          To32BitIndex(eigen_in).shuffle(permute);
    } else {
      eigen_out.device(*dev) = eigen_in.shuffle(permute);
    }
  }
};

}  // namespace math
}  // namespace pten

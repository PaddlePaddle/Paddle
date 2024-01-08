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
#include <algorithm>
#include <vector>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
struct GraphSendRecvSumFunctor {
  void operator()(const bool& first_flag UNUSED,
                  const DenseTensor& src_slice,
                  DenseTensor* dst_slice) {
    auto eigen_src = phi::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = phi::EigenVector<T>::Flatten(*dst_slice);
    eigen_dst += eigen_src;
  }
};

template <typename T>
struct GraphSendRecvMinFunctor {
  void operator()(const bool& first_flag,
                  const DenseTensor& src_slice,
                  DenseTensor* dst_slice) {
    auto eigen_src = phi::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = phi::EigenVector<T>::Flatten(*dst_slice);
    if (first_flag) {
      eigen_dst += eigen_src;
    } else {
      eigen_dst = eigen_dst.cwiseMin(eigen_src);
    }
  }
};

template <typename T>
struct GraphSendRecvMaxFunctor {
  void operator()(const int& first_flag,
                  const DenseTensor& src_slice,
                  DenseTensor* dst_slice) {
    auto eigen_src = phi::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = phi::EigenVector<T>::Flatten(*dst_slice);
    if (first_flag) {
      eigen_dst += eigen_src;
    } else {
      eigen_dst = eigen_dst.cwiseMax(eigen_src);
    }
  }
};

template <typename T, typename IndexT, typename Functor>
void ElementwiseInnerOperation(const DenseTensor& src,
                               DenseTensor* dst,
                               const IndexT& src_index,
                               const IndexT& dst_index,
                               const bool& first_flag,
                               Functor functor) {
  auto src_slice = src.Slice(src_index, src_index + 1);
  auto dst_slice = dst->Slice(dst_index, dst_index + 1);

  functor(first_flag, src_slice, &dst_slice);
}

}  // namespace phi

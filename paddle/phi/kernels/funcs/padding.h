/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {
namespace funcs {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = EigenTensor<T, D, MajorType, IndexType>;

template <typename DeviceContext, typename T, size_t D>
void PadFunction(const DeviceContext& context,
                 const std::vector<int>& pads,
                 const DenseTensor& src,
                 T pad_value,
                 DenseTensor* out) {
  std::array<std::pair<int64_t, int64_t>, D> paddings;

  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = pads[i * 2];
    paddings[i].second = pads[i * 2 + 1];
  }

  auto src_tensor = EigenTensor<T, D>::From(src);
  auto out_tensor = EigenTensor<T, D>::From(*out);

  auto& place = *(context.eigen_device());
  EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_tensor, src_tensor, paddings, pad_value);
}

template <typename DeviceContext, typename T, size_t D>
void PadGradFunction(const DeviceContext& context,
                     const std::vector<int>& pads,
                     const DenseTensor& src,
                     DenseTensor* d_out) {
  std::array<std::pair<int64_t, int64_t>, D> paddings;
  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = -pads[i * 2];
    paddings[i].second = -pads[i * 2 + 1];
  }

  auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
  auto src_tensor = EigenTensor<T, D>::From(src);
  auto& place = *(context.eigen_device());
  EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
      place, d_out_tensor, src_tensor, paddings, static_cast<T>(0));
}

template <typename DeviceContext, typename T>
void PaddingFunctor(int rank,
                    const DeviceContext& context,
                    const std::vector<int>& pads,
                    T pad_value,
                    const DenseTensor& src,
                    DenseTensor* out) {
  switch (rank) {
    case 1:
      PadFunction<DeviceContext, T, 1>(context, pads, src, pad_value, out);
      break;
    case 2:
      PadFunction<DeviceContext, T, 2>(context, pads, src, pad_value, out);
      break;
    case 3:
      PadFunction<DeviceContext, T, 3>(context, pads, src, pad_value, out);
      break;
    case 4:
      PadFunction<DeviceContext, T, 4>(context, pads, src, pad_value, out);
      break;
    case 5:
      PadFunction<DeviceContext, T, 5>(context, pads, src, pad_value, out);
      break;
    case 6:
      PadFunction<DeviceContext, T, 6>(context, pads, src, pad_value, out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "PadOp only support tensors with no more"
          " than 6 dimensions currently."));
  }
}

template <typename DeviceContext, typename T>
void PaddingGradFunctor(int rank,
                        const DeviceContext& context,
                        const std::vector<int>& pads,
                        const DenseTensor& src,
                        DenseTensor* out) {
  switch (rank) {
    case 1:
      PadGradFunction<DeviceContext, T, 1>(context, pads, src, out);
      break;
    case 2:
      PadGradFunction<DeviceContext, T, 2>(context, pads, src, out);
      break;
    case 3:
      PadGradFunction<DeviceContext, T, 3>(context, pads, src, out);
      break;
    case 4:
      PadGradFunction<DeviceContext, T, 4>(context, pads, src, out);
      break;
    case 5:
      PadGradFunction<DeviceContext, T, 5>(context, pads, src, out);
      break;
    case 6:
      PadGradFunction<DeviceContext, T, 6>(context, pads, src, out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "PadOp only support tensors with no more"
          " than 6 dimensions currently."));
  }
}

inline bool IsSymmetricPadding(const std::vector<int>& pads,
                               const int data_dim) {
  bool is_sys_pad = true;
  if (static_cast<int>(pads.size()) == data_dim * 2) {
    for (int i = 0; i < data_dim; ++i) {
      if (pads[2 * i] != pads[2 * i + 1]) {
        is_sys_pad = false;
        return is_sys_pad;
      }
    }
  }
  return is_sys_pad;
}
}  // namespace funcs
}  // namespace phi

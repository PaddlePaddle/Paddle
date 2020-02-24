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
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename DeviceContext, typename T, size_t D>
void PadFunction(const framework::ExecutionContext& context,
                 const std::vector<int>& pads, const framework::Tensor& src,
                 T pad_value, framework::Tensor* out) {
  Eigen::array<std::pair<int, int>, D> paddings;

  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = pads[i * 2];
    paddings[i].second = pads[i * 2 + 1];
  }

  auto src_tensor = EigenTensor<T, D>::From(src);
  auto out_tensor = EigenTensor<T, D>::From(*out);

  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  out_tensor.device(place) = src_tensor.pad(paddings, pad_value);
}

template <typename DeviceContext, typename T, size_t D>
void PadGradFunction(const framework::ExecutionContext& context,
                     const std::vector<int>& pads, const framework::Tensor& src,
                     framework::Tensor* d_out) {
  Eigen::array<std::pair<int, int>, D> paddings;
  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = -pads[i * 2];
    paddings[i].second = -pads[i * 2 + 1];
  }

  auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
  auto src_tensor = EigenTensor<T, D>::From(src);
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  d_out_tensor.device(place) = src_tensor.pad(paddings, static_cast<T>(0));
}

template <typename DeviceContext, typename T>
void PaddingFunctor(int rank, const framework::ExecutionContext& context,
                    const std::vector<int>& pads, T pad_value,
                    const framework::Tensor& src, framework::Tensor* out) {
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
      PADDLE_THROW(
          "PadOp only support tensors with no more than 6 dimensions.");
  }
}

template <typename DeviceContext, typename T>
void PaddingGradFunctor(int rank, const framework::ExecutionContext& context,
                        const std::vector<int>& pads,
                        const framework::Tensor& src, framework::Tensor* out) {
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
      PADDLE_THROW(
          "PadOp only support tensors with no more than 6 dimensions.");
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
}  // namespace math
}  // namespace operators
}  // namespace paddle

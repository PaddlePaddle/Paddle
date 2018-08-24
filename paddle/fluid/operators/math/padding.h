/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
void PadConstantBatchSizeLikeFunction(
    const framework::ExecutionContext& context, const std::vector<int>& pads,
    const framework::Tensor& src, T pad_value, framework::Tensor* out) {
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
void PadConstantBatchSizeLikeGradFunction(
    const framework::ExecutionContext& context, const std::vector<int>& pads,
    const framework::Tensor& src, framework::Tensor* d_out) {
  Eigen::array<std::pair<int, int>, D> paddings;
  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = -pads[i * 2];
    paddings[i].second = -pads[i * 2 + 1];
  }

  auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
  auto src_tensor = EigenTensor<T, D>::From(src);
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  d_out_tensor.device(place) = src_tensor.pad(paddings, 0);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle

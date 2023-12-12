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
#include <memory>
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

using phi::To32BitIndex;

template <typename DeviceContext, typename T>
void SetConstant<DeviceContext, T>::operator()(const DeviceContext& context,
                                               phi::DenseTensor* tensor,
                                               T num) {
  auto t = phi::EigenVector<T>::Flatten(*tensor);
  t.device(*context.eigen_device()) = t.constant(static_cast<T>(num));
}

#ifdef PADDLE_WITH_XPU
template <typename T>
void SetConstant<phi::XPUContext, T>::operator()(const phi::XPUContext& context,
                                                 phi::DenseTensor* tensor,
                                                 T num) {
  phi::VisitDataType(tensor->dtype(),
                     TensorSetConstantXPU<T>(tensor, num, context.GetPlace()));
}
#endif

template <typename DeviceContext, typename T, int Rank>
void Transpose<DeviceContext, T, Rank>::operator()(
    const DeviceContext& context,
    const phi::DenseTensor& in,
    phi::DenseTensor* out,
    const std::vector<int>& axis) {
  Eigen::array<int, Rank> permute;
  for (int i = 0; i < Rank; i++) {
    permute[i] = axis[i];
  }
  auto eigen_in = phi::EigenTensor<T, Rank>::From(in);
  auto eigen_out = phi::EigenTensor<T, Rank>::From(*out);
  auto* dev = context.eigen_device();
  // use 32bit index to speed up computation
  bool use_32bit_index = eigen_out.size() < Eigen::NumTraits<int>::highest();
  bool is_gpu_place = context.GetPlace().GetType() == phi::AllocationType::GPU;
  if (use_32bit_index && is_gpu_place) {
    To32BitIndex(eigen_out).device(*dev) =
        To32BitIndex(eigen_in).shuffle(permute);
  } else {
    eigen_out.device(*dev) = eigen_in.shuffle(permute);
  }
}

template <typename DeviceContext, typename T>
void ColwiseSum<DeviceContext, T>::operator()(const DeviceContext& context,
                                              const phi::DenseTensor& input,
                                              phi::DenseTensor* out) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(out->numel(),
                    size,
                    phi::errors::InvalidArgument(
                        "The size of output tensor "
                        "should be equal to the size of input tensor column"
                        " dimension. Expected output size=%d, but received %d",
                        size,
                        out->numel()));

  auto in = phi::EigenMatrix<T>::From(input);
  auto vec = phi::EigenVector<T>::Flatten(*out);

  vec.device(*context.eigen_device()) = in.sum(Eigen::array<int, 1>({{0}}));
}

// Specialize for CPU, since Eigen implement a general reduce. However,
// colwise-sum can be easily implemented. General reduce has a huge overhead in
// CPU
template <typename T>
class ColwiseSum<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out) {
    auto& in_dims = input.dims();
    auto height = in_dims[0];
    auto size = in_dims[1];
    PADDLE_ENFORCE_EQ(
        out->numel(),
        size,
        phi::errors::InvalidArgument(
            "The size of output tensor "
            "should be equal to the size of input tensor column"
            " dimension. Expected output size=%d, but received %d",
            size,
            out->numel()));

    T* out_buf = context.template Alloc<T>(out);
    const T* in_buf = input.data<T>();

    for (size_t i = 0; i < static_cast<size_t>(height); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(size); ++j) {
        if (i == 0) {
          out_buf[j] = in_buf[i * size + j];
        } else {
          out_buf[j] += in_buf[i * size + j];
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
void RowwiseMean<DeviceContext, T>::operator()(const DeviceContext& context,
                                               const phi::DenseTensor& input,
                                               phi::DenseTensor* out) {
  auto in_dims = input.dims();
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    2U,
                    phi::errors::InvalidArgument("The rank of input tensor "
                                                 "should be 2, but received %d",
                                                 in_dims.size()));
  PADDLE_ENFORCE_EQ(out->numel(),
                    in_dims[0],
                    phi::errors::InvalidArgument(
                        "The size of output tensor "
                        "should be equal to the size of input tensor row"
                        " dimension. Expected output size=%d, but received %d",
                        in_dims[0],
                        out->numel()));

  auto in = phi::EigenMatrix<T>::From(input);
  auto vec = phi::EigenVector<T>::Flatten(*out);

  vec.device(*context.eigen_device()) = in.mean(Eigen::array<int, 1>({{1}}));
}
// TODO(zcd): Following ColwiseSum format, need to confirm.
// Specialize for CPU, since Eigen implement a general reduce. However,
// rowwise-sum can be easily implemented. General reduce has a huge overhead in
// CPU
template <typename T>
class RowwiseMean<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out) {
    auto& in_dims = input.dims();
    PADDLE_ENFORCE_EQ(
        in_dims.size(),
        2U,
        phi::errors::InvalidArgument("The rank of input tensor "
                                     "should be 2, but received %d",
                                     in_dims.size()));
    auto height = in_dims[0];
    auto size = in_dims[1];
    PADDLE_ENFORCE_EQ(
        out->numel(),
        height,
        phi::errors::InvalidArgument(
            "The size of output tensor "
            "should be equal to the size of input tensor row"
            " dimension. Expected output size=%d, but received %d",
            height,
            out->numel()));
    auto inv_size = 1.0 / size;
    T* out_buf = context.template Alloc<T>(out);
    const T* in_buf = input.data<T>();

    for (size_t i = 0; i < static_cast<size_t>(height); ++i) {
      T sum = 0;
      for (size_t j = 0; j < static_cast<size_t>(size); ++j) {
        sum += in_buf[i * size + j];
      }
      out_buf[i] = sum * inv_size;
    }
  }
};

template <typename DeviceContext, typename T>
void RowwiseSum<DeviceContext, T>::operator()(const DeviceContext& context,
                                              const phi::DenseTensor& input,
                                              phi::DenseTensor* out) {
  auto in_dims = input.dims();
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    2U,
                    phi::errors::InvalidArgument("The rank of input tensor "
                                                 "should be 2, but received %d",
                                                 in_dims.size()));
  PADDLE_ENFORCE_EQ(out->numel(),
                    in_dims[0],
                    phi::errors::InvalidArgument(
                        "The size of output tensor "
                        "should be equal to the size of input tensor row"
                        " dimension. Expected output size=%d, but received %d",
                        in_dims[0],
                        out->numel()));

  auto in = phi::EigenMatrix<T>::From(input);
  auto vec = phi::EigenVector<T>::Flatten(*out);

  vec.device(*context.eigen_device()) = in.sum(Eigen::array<int, 1>({{1}}));
}
// TODO(zcd): Following ColwiseSum format, need to confirm.
// Specialize for CPU, since Eigen implement a general reduce. However,
// rowwise-sum can be easily implemented. General reduce has a huge overhead in
// CPU
template <typename T>
class RowwiseSum<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out) {
    auto& in_dims = input.dims();
    PADDLE_ENFORCE_EQ(
        in_dims.size(),
        2U,
        phi::errors::InvalidArgument("The rank of input tensor "
                                     "should be 2, but received %d",
                                     in_dims.size()));
    auto height = in_dims[0];
    auto size = in_dims[1];
    PADDLE_ENFORCE_EQ(
        out->numel(),
        height,
        phi::errors::InvalidArgument(
            "The size of output tensor "
            "should be equal to the size of input tensor row"
            " dimension. Expected output size=%d, but received %d",
            height,
            out->numel()));

    T* out_buf = context.template Alloc<T>(out);
    const T* in_buf = input.data<T>();

    for (size_t i = 0; i < static_cast<size_t>(height); ++i) {
      T sum = 0;
      for (size_t j = 0; j < static_cast<size_t>(size); ++j) {
        sum += in_buf[i * size + j];
      }
      out_buf[i] = sum;
    }
  }
};

}  // namespace funcs
}  // namespace phi

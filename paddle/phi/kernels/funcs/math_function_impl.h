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

<<<<<<< HEAD
#include "paddle/phi/common/data_type.h"
=======
#include "paddle/fluid/framework/data_type.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

<<<<<<< HEAD
using phi::To32BitIndex;

template <typename DeviceContext, typename T>
void SetConstant<DeviceContext, T>::operator()(const DeviceContext& context,
                                               phi::DenseTensor* tensor,
                                               T num) {
  auto t = phi::EigenVector<T>::Flatten(*tensor);
=======
using paddle::framework::To32BitIndex;

template <typename DeviceContext, typename T>
void SetConstant<DeviceContext, T>::operator()(
    const DeviceContext& context, paddle::framework::Tensor* tensor, T num) {
  auto t = paddle::framework::EigenVector<T>::Flatten(*tensor);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  t.device(*context.eigen_device()) = t.constant(static_cast<T>(num));
}

#ifdef PADDLE_WITH_XPU
template <typename T>
void SetConstant<XPUContext, T>::operator()(const XPUContext& context,
<<<<<<< HEAD
                                            phi::DenseTensor* tensor,
=======
                                            paddle::framework::Tensor* tensor,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                            T num) {
  phi::VisitDataType(tensor->dtype(),
                     TensorSetConstantXPU<T>(tensor, num, context.GetPlace()));
}
template <typename T>
void SetConstant<paddle::platform::XPUDeviceContext, T>::operator()(
    const paddle::platform::XPUDeviceContext& context,
<<<<<<< HEAD
    phi::DenseTensor* tensor,
=======
    paddle::framework::Tensor* tensor,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    T num) {
  phi::VisitDataType(tensor->dtype(),
                     TensorSetConstantXPU<T>(tensor, num, context.GetPlace()));
}
#endif

template <typename DeviceContext, typename T, int Rank>
void Transpose<DeviceContext, T, Rank>::operator()(
    const DeviceContext& context,
<<<<<<< HEAD
    const phi::DenseTensor& in,
    phi::DenseTensor* out,
=======
    const paddle::framework::Tensor& in,
    paddle::framework::Tensor* out,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    const std::vector<int>& axis) {
  Eigen::array<int, Rank> permute;
  for (int i = 0; i < Rank; i++) {
    permute[i] = axis[i];
  }
<<<<<<< HEAD
  auto eigen_in = phi::EigenTensor<T, Rank>::From(in);
  auto eigen_out = phi::EigenTensor<T, Rank>::From(*out);
=======
  auto eigen_in = paddle::framework::EigenTensor<T, Rank>::From(in);
  auto eigen_out = paddle::framework::EigenTensor<T, Rank>::From(*out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto* dev = context.eigen_device();
  // use 32bit index to speed up computation
  bool use_32bit_index = eigen_out.size() < Eigen::NumTraits<int>::highest();
  bool is_gpu_place = paddle::platform::is_gpu_place(context.GetPlace());
  if (use_32bit_index && is_gpu_place) {
    To32BitIndex(eigen_out).device(*dev) =
        To32BitIndex(eigen_in).shuffle(permute);
  } else {
    eigen_out.device(*dev) = eigen_in.shuffle(permute);
  }
}

template <typename DeviceContext, typename T>
<<<<<<< HEAD
void ColwiseSum<DeviceContext, T>::operator()(const DeviceContext& context,
                                              const phi::DenseTensor& input,
                                              phi::DenseTensor* out) {
=======
void ColwiseSum<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const paddle::framework::Tensor& input,
    paddle::framework::Tensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
  auto in = phi::EigenMatrix<T>::From(input);
  auto vec = phi::EigenVector<T>::Flatten(*out);
=======
  auto in = paddle::framework::EigenMatrix<T>::From(input);
  auto vec = paddle::framework::EigenVector<T>::Flatten(*out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  vec.device(*context.eigen_device()) = in.sum(Eigen::array<int, 1>({{0}}));
}

// Specialize for CPU, since Eigen implement a general reduce. However,
// colwise-sum can be easily implemented. General reduce has a huge overhead in
// CPU
template <typename T>
class ColwiseSum<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out) {
=======
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
    T* out_buf = context.template Alloc<T>(out);
=======
    T* out_buf = out->mutable_data<T>(out->place());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
void RowwiseMean<DeviceContext, T>::operator()(const DeviceContext& context,
                                               const phi::DenseTensor& input,
                                               phi::DenseTensor* out) {
=======
void RowwiseMean<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const paddle::framework::Tensor& input,
    paddle::framework::Tensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
  auto in = phi::EigenMatrix<T>::From(input);
  auto vec = phi::EigenVector<T>::Flatten(*out);
=======
  auto in = paddle::framework::EigenMatrix<T>::From(input);
  auto vec = paddle::framework::EigenVector<T>::Flatten(*out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out) {
=======
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
    T* out_buf = context.template Alloc<T>(out);
=======
    T* out_buf = out->mutable_data<T>(out->place());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
void RowwiseSum<DeviceContext, T>::operator()(const DeviceContext& context,
                                              const phi::DenseTensor& input,
                                              phi::DenseTensor* out) {
=======
void RowwiseSum<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const paddle::framework::Tensor& input,
    paddle::framework::Tensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
  auto in = phi::EigenMatrix<T>::From(input);
  auto vec = phi::EigenVector<T>::Flatten(*out);
=======
  auto in = paddle::framework::EigenMatrix<T>::From(input);
  auto vec = paddle::framework::EigenVector<T>::Flatten(*out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out) {
=======
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
    T* out_buf = context.template Alloc<T>(out);
=======
    T* out_buf = out->mutable_data<T>(out->place());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

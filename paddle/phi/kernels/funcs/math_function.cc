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

#include "paddle/phi/kernels/funcs/math_function.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <memory>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {
namespace funcs {

using float16 = phi::dtype::float16;

template struct SetConstant<phi::CPUContext, phi::dtype::float16>;
template struct SetConstant<phi::CPUContext, phi::dtype::bfloat16>;
template struct SetConstant<phi::CPUContext, float>;
template struct SetConstant<phi::CPUContext, double>;
template struct SetConstant<phi::CPUContext, int16_t>;
template struct SetConstant<phi::CPUContext, int>;
template struct SetConstant<phi::CPUContext, int64_t>;
template struct SetConstant<phi::CPUContext, bool>;
template struct SetConstant<phi::CPUContext, uint8_t>;
template struct SetConstant<phi::CPUContext, phi::dtype::complex<float>>;
template struct SetConstant<phi::CPUContext, phi::dtype::complex<double>>;

#ifdef PADDLE_WITH_XPU
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            phi::dtype::float16>;
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            phi::dtype::bfloat16>;
template struct SetConstant<paddle::platform::XPUDeviceContext, float>;
template struct SetConstant<paddle::platform::XPUDeviceContext, double>;
template struct SetConstant<paddle::platform::XPUDeviceContext, uint8_t>;
template struct SetConstant<paddle::platform::XPUDeviceContext, int16_t>;
template struct SetConstant<paddle::platform::XPUDeviceContext, int>;
template struct SetConstant<paddle::platform::XPUDeviceContext, int64_t>;
template struct SetConstant<paddle::platform::XPUDeviceContext, bool>;
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            phi::dtype::complex<float>>;
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            phi::dtype::complex<double>>;

template struct SetConstant<phi::XPUContext, phi::dtype::float16>;
template struct SetConstant<phi::XPUContext, phi::dtype::bfloat16>;
template struct SetConstant<phi::XPUContext, float>;
template struct SetConstant<phi::XPUContext, double>;
template struct SetConstant<phi::XPUContext, uint8_t>;
template struct SetConstant<phi::XPUContext, int16_t>;
template struct SetConstant<phi::XPUContext, int>;
template struct SetConstant<phi::XPUContext, int64_t>;
template struct SetConstant<phi::XPUContext, bool>;
template struct SetConstant<phi::XPUContext, phi::dtype::complex<float>>;
template struct SetConstant<phi::XPUContext, phi::dtype::complex<double>>;

#endif

#define DEFINE_CPU_TRANS(RANK)                                            \
  template struct Transpose<phi::CPUContext, phi::dtype::float16, RANK>;  \
  template struct Transpose<phi::CPUContext, phi::dtype::bfloat16, RANK>; \
  template struct Transpose<phi::CPUContext, float, RANK>;                \
  template struct Transpose<phi::CPUContext, double, RANK>;               \
  template struct Transpose<phi::CPUContext, int, RANK>;                  \
  template struct Transpose<phi::CPUContext, int64_t, RANK>;              \
  template struct Transpose<phi::CPUContext, bool, RANK>;                 \
  template struct Transpose<phi::CPUContext, int16_t, RANK>;              \
  template struct Transpose<phi::CPUContext, uint8_t, RANK>;              \
  template struct Transpose<phi::CPUContext, int8_t, RANK>;               \
  template struct Transpose<phi::CPUContext,                              \
                            phi::dtype::complex<float>,                   \
                            RANK>;                                        \
  template struct Transpose<phi::CPUContext, phi::dtype::complex<double>, RANK>;

DEFINE_CPU_TRANS(1);
DEFINE_CPU_TRANS(2);
DEFINE_CPU_TRANS(3);
DEFINE_CPU_TRANS(4);
DEFINE_CPU_TRANS(5);
DEFINE_CPU_TRANS(6);

template <typename DeviceContext, typename T>
void TransposeNormal<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const paddle::framework::Tensor& in,
    paddle::framework::Tensor* out,
    const std::vector<int>& axis) {
  const int rank = axis.size();
  auto in_stride = phi::stride(in.dims());
  auto out_stride = phi::stride(out->dims());
  const T* in_ptr = in.data<T>();
  T* out_ptr = out->data<T>();

  auto transpose_helper = [&](int64_t beg, int64_t end) {
    for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
      int64_t in_idx = 0;
      int64_t tmp_idx = out_idx;
      // calculate the input index
      for (int i = 0; i < rank; ++i) {
        const int64_t coordinate = tmp_idx / out_stride[i];
        tmp_idx -= coordinate * out_stride[i];
        in_idx += coordinate * in_stride[axis[i]];
      }
      out_ptr[out_idx] = in_ptr[in_idx];
    }
  };
  transpose_helper(0, out->numel());
}

// define transpose normal
#define DEFINE_CPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<phi::CPUContext, TYPE>

DEFINE_CPU_TRANS_NORMAL(phi::dtype::float16);
DEFINE_CPU_TRANS_NORMAL(phi::dtype::bfloat16);
DEFINE_CPU_TRANS_NORMAL(float);
DEFINE_CPU_TRANS_NORMAL(double);
DEFINE_CPU_TRANS_NORMAL(int);
DEFINE_CPU_TRANS_NORMAL(int64_t);
DEFINE_CPU_TRANS_NORMAL(bool);
DEFINE_CPU_TRANS_NORMAL(int16_t);
DEFINE_CPU_TRANS_NORMAL(uint8_t);
DEFINE_CPU_TRANS_NORMAL(int8_t);
DEFINE_CPU_TRANS_NORMAL(phi::dtype::complex<float>);
DEFINE_CPU_TRANS_NORMAL(phi::dtype::complex<double>);

struct TensorSetConstantCPU {
  TensorSetConstantCPU(paddle::framework::Tensor* tensor, float value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void apply() const {
    auto cpu = phi::CPUPlace();
    auto* begin = tensor_->mutable_data<T>(cpu);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(value_));
  }
  paddle::framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<paddle::platform::XPUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(phi::errors::Unimplemented("XPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::NPUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(phi::errors::Unimplemented("NPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::NPUPinnedPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(phi::errors::Unimplemented("NPUPinnedPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::IPUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(phi::errors::Unimplemented("IPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::CustomPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(phi::errors::Unimplemented("CustomPlace is not supported"));
}

template <>
void set_constant_with_place<phi::CPUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  phi::VisitDataType(tensor->dtype(), TensorSetConstantCPU(tensor, value));
}

template <>
void set_constant_with_place<paddle::platform::MLUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(phi::errors::Unimplemented("MLUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::CUDAPinnedPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  phi::VisitDataType(tensor->dtype(), TensorSetConstantCPU(tensor, value));
}

struct TensorSetConstantWithPlace
    : public std::unary_function<paddle::platform::Place, void> {
  TensorSetConstantWithPlace(const paddle::platform::DeviceContext& context,
                             paddle::framework::Tensor* tensor,
                             float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename Place>
  void operator()(Place place) const {
    set_constant_with_place<Place>(context_, tensor_, value_);
  }

  const paddle::platform::DeviceContext& context_;
  paddle::framework::Tensor* tensor_;
  float value_;
};

void set_constant(const paddle::platform::DeviceContext& context,
                  paddle::framework::Tensor* tensor,
                  float value) {
  TensorSetConstantWithPlace func(context, tensor, value);
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (paddle::platform::is_custom_place(context.GetPlace())) {
    func(phi::CPUPlace());
    return;
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // tensor->place().apply_visitor(func);
  paddle::platform::VisitPlace(tensor->place(), func);
#else
  func(phi::CPUPlace());
#endif
}

template struct ColwiseSum<phi::CPUContext, float>;
template struct ColwiseSum<phi::CPUContext, double>;
template struct ColwiseSum<phi::CPUContext, int>;
template struct ColwiseSum<phi::CPUContext, int64_t>;

template struct RowwiseMean<phi::CPUContext, float>;
template struct RowwiseMean<phi::CPUContext, double>;

template <typename T>
struct RowwiseAdd<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const paddle::framework::Tensor& input,
                  const paddle::framework::Tensor& vector,
                  paddle::framework::Tensor* output) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(
        vector.numel(),
        size,
        phi::errors::InvalidArgument(
            "The input vector size"
            " should be equal to the size of each row of input tensor."
            " Expected vector size=%d, but received %d",
            size,
            vector.numel()));
    const char* in_dims_cstr = in_dims.to_str().c_str();
    const char* out_dims_cstr = out_dims.to_str().c_str();
    PADDLE_ENFORCE_EQ(out_dims,
                      in_dims,
                      phi::errors::InvalidArgument(
                          "The output tensor shape should be same as the input"
                          " tensor shape. Expected output tensor shape: %s,"
                          " but received %s",
                          in_dims_cstr,
                          out_dims_cstr));

    auto in = paddle::framework::EigenMatrix<T>::From(input);
    auto vec = paddle::framework::EigenVector<T>::Flatten(vector);
    auto out = paddle::framework::EigenMatrix<T>::From(*output);

    for (int64_t i = 0; i < in_dims[0]; ++i) {
      out.chip(i, 0) = in.chip(i, 0) + vec;
    }
  }
};

template struct RowwiseAdd<phi::CPUContext, float>;
template struct RowwiseAdd<phi::CPUContext, double>;

}  // namespace funcs
}  // namespace phi

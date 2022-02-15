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

#include "paddle/pten/kernels/funcs/math_function.h"

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
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"
#include "paddle/pten/kernels/funcs/math_function_impl.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace pten {
namespace funcs {

using float16 = paddle::platform::float16;

template struct SetConstant<paddle::platform::CPUDeviceContext,
                            paddle::platform::float16>;
template struct SetConstant<paddle::platform::CPUDeviceContext,
                            paddle::platform::bfloat16>;
template struct SetConstant<paddle::platform::CPUDeviceContext, float>;
template struct SetConstant<paddle::platform::CPUDeviceContext, double>;
template struct SetConstant<paddle::platform::CPUDeviceContext, int16_t>;
template struct SetConstant<paddle::platform::CPUDeviceContext, int>;
template struct SetConstant<paddle::platform::CPUDeviceContext, int64_t>;
template struct SetConstant<paddle::platform::CPUDeviceContext, bool>;
template struct SetConstant<paddle::platform::CPUDeviceContext, uint8_t>;
template struct SetConstant<paddle::platform::CPUDeviceContext,
                            paddle::platform::complex<float>>;
template struct SetConstant<paddle::platform::CPUDeviceContext,
                            paddle::platform::complex<double>>;

template struct SetConstant<pten::CPUContext, paddle::platform::float16>;
template struct SetConstant<pten::CPUContext, paddle::platform::bfloat16>;
template struct SetConstant<pten::CPUContext, float>;
template struct SetConstant<pten::CPUContext, double>;
template struct SetConstant<pten::CPUContext, int16_t>;
template struct SetConstant<pten::CPUContext, int>;
template struct SetConstant<pten::CPUContext, int64_t>;
template struct SetConstant<pten::CPUContext, bool>;
template struct SetConstant<pten::CPUContext, uint8_t>;
template struct SetConstant<pten::CPUContext, paddle::platform::complex<float>>;
template struct SetConstant<pten::CPUContext,
                            paddle::platform::complex<double>>;

#ifdef PADDLE_WITH_XPU
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            paddle::platform::float16>;
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            paddle::platform::bfloat16>;
template struct SetConstant<paddle::platform::XPUDeviceContext, float>;
template struct SetConstant<paddle::platform::XPUDeviceContext, double>;
template struct SetConstant<paddle::platform::XPUDeviceContext, uint8_t>;
template struct SetConstant<paddle::platform::XPUDeviceContext, int16_t>;
template struct SetConstant<paddle::platform::XPUDeviceContext, int>;
template struct SetConstant<paddle::platform::XPUDeviceContext, int64_t>;
template struct SetConstant<paddle::platform::XPUDeviceContext, bool>;
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            paddle::platform::complex<float>>;
template struct SetConstant<paddle::platform::XPUDeviceContext,
                            paddle::platform::complex<double>>;
#endif

#define DEFINE_CPU_TRANS(RANK)                                                 \
  template struct Transpose<paddle::platform::CPUDeviceContext,                \
                            paddle::platform::float16,                         \
                            RANK>;                                             \
  template struct Transpose<paddle::platform::CPUDeviceContext,                \
                            paddle::platform::bfloat16,                        \
                            RANK>;                                             \
  template struct Transpose<paddle::platform::CPUDeviceContext, float, RANK>;  \
  template struct Transpose<paddle::platform::CPUDeviceContext, double, RANK>; \
  template struct Transpose<paddle::platform::CPUDeviceContext, int, RANK>;    \
  template struct Transpose<paddle::platform::CPUDeviceContext,                \
                            int64_t,                                           \
                            RANK>;                                             \
  template struct Transpose<paddle::platform::CPUDeviceContext, bool, RANK>;   \
  template struct Transpose<paddle::platform::CPUDeviceContext,                \
                            int16_t,                                           \
                            RANK>;                                             \
  template struct Transpose<paddle::platform::CPUDeviceContext,                \
                            uint8_t,                                           \
                            RANK>;                                             \
  template struct Transpose<paddle::platform::CPUDeviceContext, int8_t, RANK>; \
  template struct Transpose<paddle::platform::CPUDeviceContext,                \
                            paddle::platform::complex<float>,                  \
                            RANK>;                                             \
  template struct Transpose<paddle::platform::CPUDeviceContext,                \
                            paddle::platform::complex<double>,                 \
                            RANK>;

DEFINE_CPU_TRANS(1);
DEFINE_CPU_TRANS(2);
DEFINE_CPU_TRANS(3);
DEFINE_CPU_TRANS(4);
DEFINE_CPU_TRANS(5);
DEFINE_CPU_TRANS(6);

template <typename T>
struct TransposeNormal<paddle::platform::CPUDeviceContext, T> {
  void operator()(const paddle::platform::CPUDeviceContext& context,
                  const paddle::framework::Tensor& in,
                  paddle::framework::Tensor* out,
                  const std::vector<int>& axis) {
    const int rank = axis.size();
    auto in_stride = paddle::framework::stride(in.dims());
    auto out_stride = paddle::framework::stride(out->dims());
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
};

// define transpose normal
#define DEFINE_CPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<paddle::platform::CPUDeviceContext, TYPE>

DEFINE_CPU_TRANS_NORMAL(paddle::platform::float16);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::bfloat16);
DEFINE_CPU_TRANS_NORMAL(float);
DEFINE_CPU_TRANS_NORMAL(double);
DEFINE_CPU_TRANS_NORMAL(int);
DEFINE_CPU_TRANS_NORMAL(int64_t);
DEFINE_CPU_TRANS_NORMAL(bool);
DEFINE_CPU_TRANS_NORMAL(int16_t);
DEFINE_CPU_TRANS_NORMAL(uint8_t);
DEFINE_CPU_TRANS_NORMAL(int8_t);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<float>);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<double>);

struct TensorSetConstantCPU {
  TensorSetConstantCPU(paddle::framework::Tensor* tensor, float value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void apply() const {
    auto cpu = paddle::platform::CPUPlace();
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
  PADDLE_THROW(
      paddle::platform::errors::Unimplemented("XPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::NPUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(
      paddle::platform::errors::Unimplemented("NPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::NPUPinnedPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(paddle::platform::errors::Unimplemented(
      "NPUPinnedPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::IPUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(
      paddle::platform::errors::Unimplemented("IPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::CPUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  paddle::framework::VisitDataType(
      paddle::framework::TransToProtoVarType(tensor->type()),
      TensorSetConstantCPU(tensor, value));
}

template <>
void set_constant_with_place<paddle::platform::MLUPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(
      paddle::platform::errors::Unimplemented("MLUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::CUDAPinnedPlace>(
    const paddle::platform::DeviceContext& context,
    paddle::framework::Tensor* tensor,
    float value) {
  paddle::framework::VisitDataType(
      paddle::framework::TransToProtoVarType(tensor->type()),
      TensorSetConstantCPU(tensor, value));
}

struct TensorSetConstantWithPlace : public boost::static_visitor<void> {
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // tensor->place().apply_visitor(func);
  paddle::platform::VisitPlace(tensor->place(), func);
#else
  func(paddle::platform::CPUPlace());
#endif
}

template <typename T>
struct RowwiseAdd<paddle::platform::CPUDeviceContext, T> {
  void operator()(const paddle::platform::CPUDeviceContext& context,
                  const paddle::framework::Tensor& input,
                  const paddle::framework::Tensor& vector,
                  paddle::framework::Tensor* output) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(
        vector.numel(),
        size,
        paddle::platform::errors::InvalidArgument(
            "The input vector size"
            " should be equal to the size of each row of input tensor."
            " Expected vector size=%d, but received %d",
            size,
            vector.numel()));
    const char* in_dims_cstr = in_dims.to_str().c_str();
    const char* out_dims_cstr = out_dims.to_str().c_str();
    PADDLE_ENFORCE_EQ(out_dims,
                      in_dims,
                      paddle::platform::errors::InvalidArgument(
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

template struct RowwiseAdd<paddle::platform::CPUDeviceContext, float>;
template struct RowwiseAdd<paddle::platform::CPUDeviceContext, double>;

template struct ColwiseSum<paddle::platform::CPUDeviceContext, float>;
template struct ColwiseSum<paddle::platform::CPUDeviceContext, double>;
template struct ColwiseSum<paddle::platform::CPUDeviceContext, int>;
template struct ColwiseSum<paddle::platform::CPUDeviceContext, int64_t>;

template struct RowwiseSum<paddle::platform::CPUDeviceContext, float>;
template struct RowwiseSum<paddle::platform::CPUDeviceContext, double>;

template struct RowwiseMean<paddle::platform::CPUDeviceContext, float>;
template struct RowwiseMean<paddle::platform::CPUDeviceContext, double>;

template <typename T>
struct ElementwiseAddTo<paddle::platform::CPUDeviceContext, T> {
  void operator()(paddle::platform::CPUDeviceContext* ctx,
                  const paddle::framework::Tensor& src,
                  paddle::framework::Tensor* dst) {
    auto in = paddle::framework::EigenVector<T>::Flatten(src);
    auto out = paddle::framework::EigenVector<T>::Flatten(*dst);
    auto& place = *(ctx->eigen_device());
    out.device(place) = out + in;
  }
};

template struct ElementwiseAddTo<paddle::platform::CPUDeviceContext,
                                 paddle::platform::float16>;

}  // namespace funcs
}  // namespace pten

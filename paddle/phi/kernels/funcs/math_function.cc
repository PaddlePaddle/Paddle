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
#include "paddle/phi/backends/dynload/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <memory>
#include <utility>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
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
    const phi::DenseTensor& in,
    phi::DenseTensor* out,
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
  TensorSetConstantCPU(phi::DenseTensor* tensor, const void* value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void apply() const {
    auto cpu = phi::CPUPlace();
    auto* begin = tensor_->mutable_data<T>(cpu);
    const T* num = reinterpret_cast<const T*>(value_);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(*num));
  }
  phi::DenseTensor* tensor_;
  const void* value_;
};
struct TensorSetConstantEx {
  TensorSetConstantEx(
      phi::DenseTensor* tensor,
      const void* value,
      paddle::platform::Place place)
      : tensor_(tensor), value_(value), place_(place) {}
  template <typename T>
  void apply() const {
    auto* data = tensor_->mutable_data<T>(place_);
    int numel = tensor_->numel();
    const T* num = reinterpret_cast<const T*>(value_);
    if (paddle::platform::is_cpu_place(place_)) {
      std::fill(data, data + numel, static_cast<T>(*num));
    } else {
      std::unique_ptr<T[]> data_cpu(new T[numel]);
      std::fill(data_cpu.get(), data_cpu.get() + numel, static_cast<T>(*num));
      paddle::memory::Copy(place_,
                       data,
                       phi::CPUPlace(),
                       static_cast<void*>(data_cpu.get()),
                       numel * sizeof(T));
    }
  }
  phi::DenseTensor* tensor_;
  const void* value_;
  paddle::platform::Place place_;
};
#ifdef PADDLE_WITH_XPU
template <typename T>
class XPUTensorTrait {
 public:
  using Type = T;
};
template <>
class XPUTensorTrait<phi::dtype::float16> {
 public:
  using Type = ::float16;
};
template <>
class XPUTensorTrait<phi::dtype::bfloat16> {
 public:
  using Type = ::float16;
};
template<>
class XPUTensorTrait<phi::dtype::complex<double>> {
public:
  using Type = int64_t;
};
template<>
class XPUTensorTrait<phi::dtype::complex<float>> {
public:
  using Type = float;
};
template <>
class XPUTensorTrait<unsigned char> {
 public:
  using Type = bool;
};
template <>
class XPUTensorTrait<double> {
 public:
  using Type = int64_t;
};
struct TensorSetConstantXPU {
  TensorSetConstantXPU(const paddle::platform::DeviceContext& context,
                       phi::DenseTensor* tensor,
                       const void* value,
                       paddle::platform::Place place)
      : context_(context), tensor_(tensor), value_(value), place_(place) {}
  template <typename T>
  void apply() const {
    auto* data = tensor_->mutable_data<T>(place_);
    int numel = tensor_->numel();
    using XPUInTDType = typename XPUTensorTrait<T>::Type;
    float num = static_cast<float>(*reinterpret_cast<const T*>(value_));
    auto dev_ctx = reinterpret_cast<const phi::XPUContext *>(&context_);
    int ret = xpu::constant(dev_ctx->x_context(),
       reinterpret_cast<XPUInTDType *>(data),
       numel,
       static_cast<XPUInTDType>(num));
    PADDLE_ENFORCE_EQ(
        ret,
        XPU_SUCCESS,
        phi::errors::External("XPU CONSTANT API return wrong value[%d %s].",
                              ret,
                              XPUAPIErrorMsg[ret]));
  }
  const paddle::platform::DeviceContext& context_;
  phi::DenseTensor* tensor_;
  const void* value_;
  paddle::platform::Place place_;
};
#endif
template <>
void set_constant_with_place<paddle::platform::XPUPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
#ifdef PADDLE_WITH_XPU
  phi::VisitDataType(
      tensor->dtype(),
      TensorSetConstantXPU<float>(tensor, value, tensor->place()));
#else
  PADDLE_THROW(phi::errors::PreconditionNotMet("Not compiled with XPU!"));
#endif
}

template <>
void set_constant_with_place<paddle::platform::NPUPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
  PADDLE_THROW(phi::errors::Unimplemented("NPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::NPUPinnedPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
  PADDLE_THROW(phi::errors::Unimplemented("NPUPinnedPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::IPUPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
  PADDLE_THROW(phi::errors::Unimplemented("IPUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::CustomPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
  PADDLE_THROW(phi::errors::Unimplemented("CustomPlace is not supported"));
}

template <>
void set_constant_with_place<phi::CPUPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
  phi::VisitDataType(tensor->dtype(), TensorSetConstantCPU(tensor, value));
}

template <>
void set_constant_with_place<paddle::platform::MLUPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
  PADDLE_THROW(phi::errors::Unimplemented("MLUPlace is not supported"));
}

template <>
void set_constant_with_place<paddle::platform::CUDAPinnedPlace>(
    const paddle::platform::DeviceContext& context,
    phi::DenseTensor* tensor,
    const void* value) {
  phi::VisitDataType(tensor->dtype(), TensorSetConstantCPU(tensor, value));
}

struct TensorSetConstantWithPlace
    : public std::unary_function<paddle::platform::Place, void> {
  TensorSetConstantWithPlace(const paddle::platform::DeviceContext& context,
                             phi::DenseTensor* tensor,
                             const void* value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename Place>
  void operator()(Place place) const {
    set_constant_with_place<Place>(context_, tensor_, value_);
  }

  const paddle::platform::DeviceContext& context_;
  phi::DenseTensor* tensor_;
  const void* value_;
};

void set_constant(const paddle::platform::DeviceContext& context,
                  phi::DenseTensor* tensor,
                  const void* value) {
  auto place = context.GetPlace();
  if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    TensorSetConstantWithPlace func(context, tensor, value);
    paddle::platform::VisitPlace(tensor->place(), func);
#endif
  } else if (paddle::platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    phi::VisitDataType(tensor->dtype(),
        TensorSetConstantXPU(context, tensor, value, place));
#endif
  } else {
    phi::VisitDataType(tensor->dtype(),
        TensorSetConstantEx(tensor, value, place));
  }
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
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& vector,
                  phi::DenseTensor* output) {
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

    auto in = phi::EigenMatrix<T>::From(input);
    auto vec = phi::EigenVector<T>::Flatten(vector);
    auto out = phi::EigenMatrix<T>::From(*output);

    for (int64_t i = 0; i < in_dims[0]; ++i) {
      out.chip(i, 0) = in.chip(i, 0) + vec;
    }
  }
};

template struct RowwiseAdd<phi::CPUContext, float>;
template struct RowwiseAdd<phi::CPUContext, double>;

}  // namespace funcs
}  // namespace phi

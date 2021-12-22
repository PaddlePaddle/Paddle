//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/cpu/elementwise_impl.h"
#include "paddle/pten/kernels/funcs/elementwise_functor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/pten/kernels/hybird/eigen/common.h"

namespace pten {
// Add
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsAddFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsAddFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VADD(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

template <typename DevCtx, typename T>
struct SameDimsAddFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    z->mutable_data<T>();
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);
    auto eigen_z = pten::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x + eigen_y;
  }
};

// Subtract
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsSubtractFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsSubtractFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VSUB(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

template <typename DevCtx, typename T>
struct SameDimsSubtractFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);
    auto eigen_z = pten::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x - eigen_y;
  }
};

// Divide
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsDivideFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsDivideFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    paddle::platform::errors::InvalidArgument(
        "If use SameDimsDivideFunctor, template args(T) must be floating "
        "point. ");
  }
};

template <typename DevCtx, typename T>
struct SameDimsDivideFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VDIV(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

// Multiply
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsMultiplyFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsMultiplyFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VMUL(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

template <typename DevCtx, typename T>
struct SameDimsMultiplyFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);
    auto eigen_z = pten::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x * eigen_y;
  }
};

template <typename T, typename ContextT>
void Divide(const ContextT& dev_ctx,
            const DenseTensor& x,
            const DenseTensor& y,
            int axis,
            DenseTensor* out) {
  // allocate memory for out
  out->mutable_data<T>();
  if (x.dims() == y.dims() && std::is_floating_point<T>::value) {
    SameDimsElementwiseCompute<SameDimsDivideFunctor<ContextT, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      ElementwiseCompute<funcs::DivideFunctor<T>, T>(
          dev_ctx, x, y, axis, funcs::DivideFunctor<T>(), out);
    } else {
      ElementwiseCompute<funcs::InverseDivideFunctor<T>, T>(
          dev_ctx, x, y, axis, funcs::InverseDivideFunctor<T>(), out);
    }
  }
}

#define DEFINE_CPU_ELEMENTWISE_OP(name)                                    \
  template <typename T, typename ContextT>                                 \
  void name(const ContextT& dev_ctx,                                       \
            const DenseTensor& x,                                          \
            const DenseTensor& y,                                          \
            int axis,                                                      \
            DenseTensor* out) {                                            \
    out->mutable_data<T>();                                                \
    if (x.dims() == y.dims()) {                                            \
      SameDimsElementwiseCompute<SameDims##name##Functor<ContextT, T>>()(  \
          dev_ctx, x, y, out);                                             \
    } else {                                                               \
      auto x_dims = x.dims();                                              \
      auto y_dims = y.dims();                                              \
      if (x_dims.size() >= y_dims.size()) {                                \
        ElementwiseCompute<funcs::name##Functor<T>, T>(                    \
            dev_ctx, x, y, axis, funcs::name##Functor<T>(), out);          \
      } else {                                                             \
        ElementwiseCompute<funcs::Inverse##name##Functor<T>, T>(           \
            dev_ctx, x, y, axis, funcs::Inverse##name##Functor<T>(), out); \
      }                                                                    \
    }                                                                      \
  }

// Create the definition of Add
DEFINE_CPU_ELEMENTWISE_OP(Add)

// Create the definition of Subtract
DEFINE_CPU_ELEMENTWISE_OP(Subtract)

// Create the definition of Multiply
DEFINE_CPU_ELEMENTWISE_OP(Multiply)

}  // namespace pten

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_CTX_KERNEL(add,
                       CPU,
                       ALL_LAYOUT,
                       pten::Add,
                       float,
                       double,
                       int,
                       int64_t,
                       complex64,
                       complex128) {}
PT_REGISTER_CTX_KERNEL(subtract,
                       CPU,
                       ALL_LAYOUT,
                       pten::Subtract,
                       float,
                       double,
                       int,
                       int64_t,
                       complex64,
                       complex128) {}
PT_REGISTER_CTX_KERNEL(divide,
                       CPU,
                       ALL_LAYOUT,
                       pten::Divide,
                       float,
                       double,
                       int,
                       int64_t,
                       complex64,
                       complex128) {}
PT_REGISTER_CTX_KERNEL(multiply,
                       CPU,
                       ALL_LAYOUT,
                       pten::Multiply,
                       float,
                       double,
                       int,
                       int64_t,
                       bool,
                       complex64,
                       complex128) {}

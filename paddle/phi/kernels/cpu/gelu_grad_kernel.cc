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

#include "paddle/phi/kernels/gelu_grad_kernel.h"

#include <algorithm>
#include <cmath>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/blas/blas_impl.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/gelu_kernel.h"

namespace phi {

template <typename T>
struct GeluGradFunctor {
  template <typename Device, typename X, typename dOut, typename dX>
  void operator()(Device d, X x, dOut dout, dX dx, bool approximate) const {
    if (approximate) {
      if (std::is_same<T, dtype::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto casted_dout = dout.template cast<float>();

        const float kAlpha = static_cast<float>(M_2_SQRTPI * M_SQRT1_2);
        const float kBeta =
            kAlpha * static_cast<float>(GELU_CONSTANT) * static_cast<float>(3);
        const auto y =
            (kAlpha *
             ((static_cast<float>(GELU_CONSTANT) * casted_x.cube()) + casted_x))
                .tanh();
        dx.device(d) = (static_cast<float>(0.5) * casted_dout *
                        (static_cast<float>(1) + y +
                         (casted_x - casted_x * y.square()) *
                             (kAlpha + kBeta * casted_x.square())))
                           .template cast<T>();
      } else {
        const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
        const T kBeta =
            kAlpha * static_cast<T>(GELU_CONSTANT) * static_cast<T>(3);
        const auto y =
            (kAlpha * ((static_cast<T>(GELU_CONSTANT) * x.cube()) + x)).tanh();
        dx.device(d) = static_cast<T>(0.5) * dout *
                       (static_cast<T>(1) + y +
                        (x - x * y.square()) * (kAlpha + kBeta * x.square()));
      }
    } else {
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__) && !defined(PADDLE_WITH_CUDA) &&                       \
    !defined(PADDLE_WITH_HIP)
      auto x_data = x.data();
      auto dx_data = dx.data();
      auto dout_data = dout.data();
      int n = std::min(x.size(), dx.size());

      auto first = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(first, 0, n * sizeof(T));
      auto second = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(second, 0, n * sizeof(T));

      // first = (0.5 * (1 + erf(x / sqrt(2))))
      phi::funcs::CBlas<T>::AXPY(
          n, static_cast<T>(M_SQRT1_2), x_data, 1, first, 1);
      phi::funcs::CBlas<T>::VMERF(n, first, first, VML_LA);
      for (int i = 0; i < n; i++) {
        first[i] += static_cast<T>(1);
      }
      phi::funcs::CBlas<T>::SCAL(n, static_cast<T>(0.5), first, 1);

      // second = (0.5 * 2/sqrt(pi) * 1/sqrt(2) * x * exp(-0.5 * x^2))
      phi::funcs::CBlas<T>::VSQUARE(n, x_data, second);
      phi::funcs::CBlas<T>::SCAL(n, -static_cast<T>(0.5), second, 1);
      phi::funcs::CBlas<T>::VEXP(n, second, second);
      phi::funcs::CBlas<T>::VMUL(n, x_data, second, second);
      phi::funcs::CBlas<T>::SCAL(
          n, static_cast<T>(0.5 * M_2_SQRTPI * M_SQRT1_2), second, 1);

      // dx = dout * (first + second);
      phi::funcs::CBlas<T>::VADD(n, first, second, first);
      phi::funcs::CBlas<T>::VMUL(n, dout_data, first, dx_data);

      std::free(first);
      std::free(second);
#else
      // gelu_grad(x) = dout * 0.5 * (1 + erf(x / sqrt(2)) + x * sqrt(2 / pi) *
      // exp(- x^2 / 2)
      if (std::is_same<T, dtype::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto casted_dout = dout.template cast<float>();
        auto first = static_cast<float>(0.5) *
                     (static_cast<float>(1) +
                      ((casted_x * static_cast<float>(M_SQRT1_2)).erf()));
        auto second = static_cast<float>(0.5 * M_2_SQRTPI * M_SQRT1_2) *
                      casted_x *
                      (-static_cast<float>(0.5) * casted_x.square()).exp();
        dx.device(d) = (casted_dout * (first + second)).template cast<T>();
      } else {
        auto first =
            static_cast<T>(0.5) *
            (static_cast<T>(1) + ((x * static_cast<T>(M_SQRT1_2)).erf()));

        auto second = static_cast<T>(0.5 * M_2_SQRTPI * M_SQRT1_2) * x *
                      (-static_cast<T>(0.5) * x.square()).exp();
        dx.device(d) = dout * (first + second);
      }
#endif
    }
  }
};

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    bool approximate,
                    DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto eigen_x = EigenVector<T>::Flatten(x);
  auto eigen_out_grad = EigenVector<T>::Flatten(out_grad);
  auto eigen_x_grad = EigenVector<T>::Flatten(*x_grad);
  auto& dev = *dev_ctx.eigen_device();

  GeluGradFunctor<T> functor;
  functor(dev, eigen_x, eigen_out_grad, eigen_x_grad, approximate);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gelu_grad, CPU, ALL_LAYOUT, phi::GeluGradKernel, float, double) {}

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

#include "paddle/phi/kernels/gelu_kernel.h"
#include <algorithm>
#include <cmath>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/blas/blas_impl.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T>
struct GeluFunctor {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out, bool approximate) const {
    if (approximate) {
      // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
      if (std::is_same<T, dtype::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto temp =
            (static_cast<float>(M_2_SQRTPI * M_SQRT1_2) *
             (casted_x + static_cast<float>(GELU_CONSTANT) * casted_x.cube()))
                .tanh();
        out.device(d) = (casted_x * static_cast<float>(0.5) *
                         (static_cast<float>(1) + temp))
                            .template cast<T>();
      } else {
        auto temp = (static_cast<T>(M_2_SQRTPI * M_SQRT1_2) *
                     (x + static_cast<T>(GELU_CONSTANT) * x.cube()))
                        .tanh();
        out.device(d) = x * static_cast<T>(0.5) * (static_cast<T>(1) + temp);
      }
    } else {
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__) && !defined(PADDLE_WITH_CUDA) &&                       \
    !defined(PADDLE_WITH_HIP)
      auto x_data = x.data();
      auto out_data = out.data();
      int n = std::min(x.size(), out.size());

      std::memset(out_data, 0, n * sizeof(T));
      phi::funcs::CBlas<T>::AXPY(
          n, static_cast<T>(M_SQRT1_2), x_data, 1, out_data, 1);
      phi::funcs::CBlas<T>::VMERF(n, out_data, out_data, VML_LA);
      for (int i = 0; i < n; i++) {
        out_data[i] += static_cast<T>(1);
      }
      phi::funcs::CBlas<T>::VMUL(n, x_data, out_data, out_data);
      for (int i = 0; i < n; i++) {
        out_data[i] *= static_cast<T>(0.5);
      }
#else
      // gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
      if (std::is_same<T, dtype::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto temp = (casted_x * static_cast<float>(M_SQRT1_2)).erf();
        out.device(d) = (casted_x * static_cast<float>(0.5) *
                         (static_cast<float>(1) + temp))
                            .template cast<T>();
      } else {
        auto temp = (x * static_cast<T>(M_SQRT1_2)).erf();
        out.device(d) = x * static_cast<T>(0.5) * (static_cast<T>(1) + temp);
      }
#endif
    }
  }
};

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const DenseTensor& x,
                bool approximate,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto eigen_out = EigenVector<T>::Flatten(*out);
  auto eigen_x = EigenVector<T>::Flatten(x);
  auto& dev = *dev_ctx.eigen_device();

  GeluFunctor<T> functor;
  functor(dev, eigen_x, eigen_out, approximate);
}

}  // namespace phi

PD_REGISTER_KERNEL(gelu, CPU, ALL_LAYOUT, phi::GeluKernel, float, double) {}

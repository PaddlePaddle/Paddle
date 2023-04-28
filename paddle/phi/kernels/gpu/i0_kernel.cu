/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/i0_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
__host__ __device__ std::tuple<const T*, size_t> ChebyshevCoefficientsI0e_A() {
  /* Chebyshev coefficients for I0e(x) in the interval [0,8]. */
  static const T coeff[] = {
      -4.41534164647933937950E-18, 3.33079451882223809783E-17,
      -2.43127984654795469359E-16, 1.71539128555513303061E-15,
      -1.16853328779934516808E-14, 7.67618549860493561688E-14,
      -4.85644678311192946090E-13, 2.95505266312963983461E-12,
      -1.72682629144155570723E-11, 9.67580903537323691224E-11,
      -5.18979560163526290666E-10, 2.65982372468238665035E-9,
      -1.30002500998624804212E-8,  6.04699502254191894932E-8,
      -2.67079385394061173391E-7,  1.11738753912010371815E-6,
      -4.41673835845875056359E-6,  1.64484480707288970893E-5,
      -5.75419501008210370398E-5,  1.88502885095841655729E-4,
      -5.76375574538582365885E-4,  1.63947561694133579842E-3,
      -4.32430999505057594430E-3,  1.05464603945949983183E-2,
      -2.37374148058994688156E-2,  4.93052842396707084878E-2,
      -9.49010970480476444210E-2,  1.71620901522208775349E-1,
      -3.04682672343198398683E-1,  6.76795274409476084995E-1};
  return std::make_tuple(coeff, 30);
}

template <typename T>
__host__ __device__ std::tuple<const T*, size_t> ChebyshevCoefficientsI0e_B() {
  /* Chebyshev coefficients for I0e(x) in the inverted interval [8,infinity]. */
  static const T coeff[] = {
      -7.23318048787475395456E-18, -4.83050448594418207126E-18,
      4.46562142029675999901E-17,  3.46122286769746109310E-17,
      -2.82762398051658348494E-16, -3.42548561967721913462E-16,
      1.77256013305652638360E-15,  3.81168066935262242075E-15,
      -9.55484669882830764870E-15, -4.15056934728722208663E-14,
      1.54008621752140982691E-14,  3.85277838274214270114E-13,
      7.18012445138366623367E-13,  -1.79417853150680611778E-12,
      -1.32158118404477131188E-11, -3.14991652796324136454E-11,
      1.18891471078464383424E-11,  4.94060238822496958910E-10,
      3.39623202570838634515E-9,   2.26666899049817806459E-8,
      2.04891858946906374183E-7,   2.89137052083475648297E-6,
      6.88975834691682398426E-5,   3.36911647825569408990E-3,
      8.04490411014108831608E-1};

  return std::make_tuple(coeff, 25);
}

template <typename T>
__host__ __device__ T Chbevl(T x, const T array[], size_t len) {
  T b0, b1, b2;

  b0 = array[0];
  b1 = static_cast<T>(0.0);
  for (size_t i = 1; i < len; ++i) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + array[i];
  }

  return (static_cast<T>(0.5) * (b0 - b2));
}

template <typename T>
struct CudaI0Functor {
  __device__ __forceinline__ T operator()(const T _x) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    MT x = std::abs(mp_x);
    if (x <= MT{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI0e_A<MT>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      MT y = (x / MT{2.0}) - MT{2.0};

      return static_cast<T>(std::exp(x) * Chbevl<MT>(y, A, len));
    }
    auto coeff_pair_B = ChebyshevCoefficientsI0e_B<MT>();
    auto B = std::get<0>(coeff_pair_B);
    auto len = std::get<1>(coeff_pair_B);
    MT y = (MT{32.0} / x) - MT{2.0};

    return static_cast<T>(std::exp(x) * Chbevl<T>(y, B, len) / std::sqrt(x));
  }
};

template <typename T, typename Context>
void I0Kernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  auto functor = CudaI0Functor<T>();
  phi::funcs::ElementwiseKernel<T>(ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(i0, GPU, ALL_LAYOUT, phi::I0Kernel, float, double) {}

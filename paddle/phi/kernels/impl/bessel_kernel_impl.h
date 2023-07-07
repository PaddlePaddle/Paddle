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

#pragma once

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
static inline std::tuple<const T*, size_t> ChebyshevCoefficientsI0e_A() {
  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0) { exp(-x) I0(x) } = 1.
   */
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
static inline std::tuple<const T*, size_t> ChebyshevCoefficientsI0e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
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
static inline T Chbevl(T x, const T array[], size_t len) {
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
struct I0eFunctor {
  I0eFunctor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    T x = std::abs(input_[idx]);
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI0e_A<T>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      T y = (x / T{2.0}) - T{2.0};

      output_[idx] = static_cast<T>(Chbevl<T>(y, A, len));
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI0e_B<T>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      T y = (T{32.0} / x) - T{2.0};

      output_[idx] = static_cast<T>(Chbevl<T>(y, B, len) / std::sqrt(x));
    }
  }

 private:
  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T>
struct I0Functor {
  I0Functor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    T x = std::abs(input_[idx]);
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI0e_A<T>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      T y = (x / T{2.0}) - T{2.0};

      output_[idx] = static_cast<T>(std::exp(x) * Chbevl<T>(y, A, len));
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI0e_B<T>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      T y = (T{32.0} / x) - T{2.0};

      output_[idx] =
          static_cast<T>(std::exp(x) * Chbevl<T>(y, B, len) / std::sqrt(x));
    }
  }

 private:
  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value,
                                      std::tuple<const T*, size_t>>::type
ChebyshevCoefficientsI1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};
  return std::make_tuple(coeff, 29);
}

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value,
                                      std::tuple<const T*, size_t>>::type
ChebyshevCoefficientsI1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {9.38153738649577178388E-9f,
                            -4.44505912879632808065E-8f,
                            2.00329475355213526229E-7f,
                            -8.56872026469545474066E-7f,
                            3.47025130813767847674E-6f,
                            -1.32731636560394358279E-5f,
                            4.78156510755005422638E-5f,
                            -1.61760815825896745588E-4f,
                            5.12285956168575772895E-4f,
                            -1.51357245063125314899E-3f,
                            4.15642294431288815669E-3f,
                            -1.05640848946261981558E-2f,
                            2.47264490306265168283E-2f,
                            -5.29459812080949914269E-2f,
                            1.02643658689847095384E-1f,
                            -1.76416518357834055153E-1f,
                            2.52587186443633654823E-1f};
  return std::make_tuple(coeff, 17);
}

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value,
                                      std::tuple<const T*, size_t>>::type
ChebyshevCoefficientsI1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  return std::make_tuple(coeff, 25);
}

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value,
                                      std::tuple<const T*, size_t>>::type
ChebyshevCoefficientsI1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {-3.83538038596423702205E-9f,
                            -2.63146884688951950684E-8f,
                            -2.51223623787020892529E-7f,
                            -3.88256480887769039346E-6f,
                            -1.10588938762623716291E-4f,
                            -9.76109749136146840777E-3f,
                            7.78576235018280120474E-1f};

  return std::make_tuple(coeff, 7);
}

template <typename T>
struct I1Functor {
  I1Functor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    T x = std::abs(input_[idx]);
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI1e_A<T>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      T y = (x / T{2.0}) - T{2.0};
      const T out = std::exp(x) * x * Chbevl(y, A, len);
      output_[idx] = (input_[idx] < T{0.0}) ? -out : out;
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI1e_B<T>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      T y = (T{32.0} / x) - T{2.0};
      const T out = (std::exp(x) * Chbevl(y, B, len)) / std::sqrt(x);
      output_[idx] = (input_[idx] < T{0.0}) ? -out : out;
    }
  }

 private:
  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T>
struct I1eFunctor {
  I1eFunctor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    T x = std::abs(input_[idx]);
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI1e_A<T>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      T y = (x / T{2.0}) - T{2.0};
      const T out = Chbevl<T>(y, A, len) * x;
      output_[idx] = (input_[idx] < T{0.0}) ? -out : out;
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI1e_B<T>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      T y = (T{32.0} / x) - T{2.0};

      const T out = Chbevl<T>(y, B, len) / std::sqrt(x);
      output_[idx] = (input_[idx] < T{0.0}) ? -out : out;
    }
  }

 private:
  const T* input_;
  T* output_;
  int64_t numel_;
};

}  // namespace phi

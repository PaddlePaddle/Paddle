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

#pragma once

// reference: https://gist.github.com/lakshayg/d80172fe5ae3c5d2c2aedb53c250320e
template <typename T>
T Erfinv(T x) {
  if (x < -1 || x > 1) {
    return std::numeric_limits<T>::quiet_NaN();
  } else if (x == 1.0) {
    return std::numeric_limits<T>::infinity();
  } else if (x == -1.0) {
    return -std::numeric_limits<T>::infinity();
  }

  const T LN2 = 6.931471805599453094172321214581e-1;

  const T A0 = 1.1975323115670912564578e0;
  const T A1 = 4.7072688112383978012285e1;
  const T A2 = 6.9706266534389598238465e2;
  const T A3 = 4.8548868893843886794648e3;
  const T A4 = 1.6235862515167575384252e4;
  const T A5 = 2.3782041382114385731252e4;
  const T A6 = 1.1819493347062294404278e4;
  const T A7 = 8.8709406962545514830200e2;

  const T B0 = 1.0000000000000000000e0;
  const T B1 = 4.2313330701600911252e1;
  const T B2 = 6.8718700749205790830e2;
  const T B3 = 5.3941960214247511077e3;
  const T B4 = 2.1213794301586595867e4;
  const T B5 = 3.9307895800092710610e4;
  const T B6 = 2.8729085735721942674e4;
  const T B7 = 5.2264952788528545610e3;

  const T C0 = 1.42343711074968357734e0;
  const T C1 = 4.63033784615654529590e0;
  const T C2 = 5.76949722146069140550e0;
  const T C3 = 3.64784832476320460504e0;
  const T C4 = 1.27045825245236838258e0;
  const T C5 = 2.41780725177450611770e-1;
  const T C6 = 2.27238449892691845833e-2;
  const T C7 = 7.74545014278341407640e-4;

  const T D0 = 1.4142135623730950488016887e0;
  const T D1 = 2.9036514445419946173133295e0;
  const T D2 = 2.3707661626024532365971225e0;
  const T D3 = 9.7547832001787427186894837e-1;
  const T D4 = 2.0945065210512749128288442e-1;
  const T D5 = 2.1494160384252876777097297e-2;
  const T D6 = 7.7441459065157709165577218e-4;
  const T D7 = 1.4859850019840355905497876e-9;

  const T E0 = 6.65790464350110377720e0;
  const T E1 = 5.46378491116411436990e0;
  const T E2 = 1.78482653991729133580e0;
  const T E3 = 2.96560571828504891230e-1;
  const T E4 = 2.65321895265761230930e-2;
  const T E5 = 1.24266094738807843860e-3;
  const T E6 = 2.71155556874348757815e-5;
  const T E7 = 2.01033439929228813265e-7;

  const T F0 = 1.414213562373095048801689e0;
  const T F1 = 8.482908416595164588112026e-1;
  const T F2 = 1.936480946950659106176712e-1;
  const T F3 = 2.103693768272068968719679e-2;
  const T F4 = 1.112800997078859844711555e-3;
  const T F5 = 2.611088405080593625138020e-5;
  const T F6 = 2.010321207683943062279931e-7;
  const T F7 = 2.891024605872965461538222e-15;

  T abs_x = abs(x);

  if (abs_x <= 0.85) {
    T r = 0.180625 - 0.25 * x * x;
    T num =
        (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) *
             r +
         A0);
    T den =
        (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) *
             r +
         B0);
    return x * num / den;
  }

  T r = sqrt(LN2 - log(1.0 - abs_x));

  T num, den;
  if (r <= 5.0) {
    r = r - 1.6;
    num =
        (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) *
             r +
         C0);
    den =
        (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) *
             r +
         D0);
  } else {
    r = r - 5.0;
    num =
        (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) *
             r +
         E0);
    den =
        (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) *
             r +
         F0);
  }

  if (x < 0) {
    return -num / den;
  } else {
    return num / den;
  }
}

template <typename T>
struct TruncatedNormal {
  T mean, std, a, b;
  T a_normal_cdf;
  T b_normal_cdf;
  TruncatedNormal(T mean, T std, T a, T b) : mean(mean), std(std), a(a), b(b) {
    auto normal_cdf = [](T x) {
      return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
    };
    a_normal_cdf = normal_cdf((a - mean) / std);
    b_normal_cdf = normal_cdf((b - mean) / std);
  }

  T operator()(T value) const {
    auto p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * value;
    T ret = std::sqrt(2.0) * Erfinv(2 * p - 1) * std + mean;
    return std::clamp(ret, a, b);
  }
};

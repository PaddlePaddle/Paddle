/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS host-device template for complex numbers supporting all CUTLASS numeric types.
*/

// Standard Library's std::complex<T> used for reference checking
#include <complex>

#include "../common/cutlass_unit_test.h"

#include "cutlass/complex.h"
#include "cutlass/constants.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, f64_to_f32_conversion) {

  cutlass::complex<double> source = {1.5, -1.25};

  cutlass::complex<float> dest = cutlass::complex<float>(source); // explicit conversion

  EXPECT_TRUE(source.real() == 1.5 && source.imag() == -1.25 && 
    dest.real() == 1.5f && dest.imag() == -1.25f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, f32_to_f64_conversion) {

  cutlass::complex<float> source = {-1.5f, 1.25f};

  cutlass::complex<double> dest = source;  // implicit conversion

  EXPECT_TRUE(source.real() == -1.5f && source.imag() == 1.25f && 
    dest.real() == -1.5 && dest.imag() == 1.25);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, s32_to_f64_conversion) {

  cutlass::complex<int> source = {-2, 1};

  cutlass::complex<double> dest = source;  // implicit conversion

  EXPECT_TRUE(source.real() == -2 && source.imag() == 1 && 
    dest.real() == -2 && dest.imag() == 1);
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, f16_to_f32_conversion) {

  cutlass::complex<cutlass::half_t> source = {1.5_hf, -1.25_hf};

  cutlass::complex<float> dest = cutlass::complex<float>(source); // explicit conversion

  EXPECT_TRUE(source.real() == 1.5_hf && source.imag() == -1.25_hf && 
    dest.real() == 1.5f && dest.imag() == -1.25f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, exp_f32) {
  
  cutlass::complex<float> Z[] = {
    {1, 1},
    {2   ,  cutlass::constants::pi<float>()/2.0f   },
    {0.5f,  cutlass::constants::pi<float>()        },
    {0.25f,  cutlass::constants::pi<float>()*3/4.0f },
    {0, 0},
  };

  cutlass::complex<double> Expected[] = {
    {1.4686939399158851, 2.2873552871788423}, 
    {4.524491950137825e-16, 7.38905609893065},
    {-1.6487212707001282, 2.019101226849069e-16}, 
    {-0.9079430793557842, 0.9079430793557843},
    {1, 0}
  };

  double tolerance = 0.00001;

  for (int i = 0; cutlass::real(Z[i]); ++i) {
    double e_r = cutlass::real(Expected[i]);
    double e_i = cutlass::real(Expected[i]);

    cutlass::complex<float> got = cutlass::exp(Z[i]);
    float g_r = cutlass::real(got);
    float g_i = cutlass::real(got);

    EXPECT_TRUE(
      std::abs(g_r - e_r) < tolerance && std::abs(g_i - e_i) < tolerance
    ) << "Expected(" << Expected[i] << "), Got(" << got << ")";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

  /// Thorough testing for basic complex math operators. Uses std::complex as a reference.
  template <typename T, int N, int M>
  struct ComplexOperators {
    ComplexOperators() {
      for (int ar = -N; ar <= N; ++ar) {
        for (int ai = -N; ai <= N; ++ai) {
          for (int br = -N; br <= N; ++br) {
            for (int bi = -N; bi <= N; ++bi) {

              cutlass::complex<T> Ae(T(ar) / T(M), T(ai) / T(M));
              cutlass::complex<T> Be(T(br) / T(M), T(bi) / T(M));

              std::complex<T> Ar(T(ar) / T(M), T(ai) / T(M));
              std::complex<T> Br(T(br) / T(M), T(bi) / T(M));

              cutlass::complex<T> add_e = Ae + Be;
              cutlass::complex<T> sub_e = Ae - Be;
              cutlass::complex<T> mul_e = Ae * Be;

              std::complex<T> add_r = (Ar + Br);
              std::complex<T> sub_r = (Ar - Br);
              std::complex<T> mul_r = (Ar * Br);

              EXPECT_EQ(real(add_e), real(add_r));
              EXPECT_EQ(imag(add_e), imag(add_r));

              EXPECT_EQ(real(sub_e), real(sub_r));
              EXPECT_EQ(imag(sub_e), imag(sub_r));

              EXPECT_EQ(real(mul_e), real(mul_r));
              EXPECT_EQ(imag(mul_e), imag(mul_r));

              if (!(br == 0 && bi == 0)) {

                cutlass::complex<T> div_e = Ae / Be;
                std::complex<T> div_r = Ar / Br;

                T const kRange = T(0.001);

                EXPECT_NEAR(real(div_e), real(div_r), kRange);
                EXPECT_NEAR(imag(div_e), imag(div_r), kRange);
              }
            }
          }
        }
      }
    }
  };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, host_float) {
  test::ComplexOperators<float, 32, 8> test;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(complex, host_double) {
  test::ComplexOperators<double, 32, 8> test;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

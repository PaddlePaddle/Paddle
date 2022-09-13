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
/* \file
  \brief Provides several functions for filling tensors with data.
*/

#pragma once

// Standard Library includes
#include <utility>
#include <cstdlib>
#include <cmath>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/quaternion.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/subbyte_reference.h"
#include "cutlass/tensor_view.h"
#include "cutlass/tensor_view_planar_complex.h"
#include "cutlass/blas3.h"

#include "cutlass/util/distribution.h"
#include "tensor_foreach.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  Element value;

  //
  // Methods
  //

  TensorFillFunc(
    TensorView const &view_ = TensorView(), 
    Element value_ = Element(0)
  ): view(view_), value(value_) { }

  void operator()(Coord<Layout::kRank> const & coord) const {
    view.at(coord) = value;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with a uniform value
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFill(
  TensorView<Element, Layout> dst,    ///< destination tensor 
  Element val = Element(0)) {               ///< value to uniformly fill it with

  detail::TensorFillFunc<Element, Layout> func(dst, val);

  TensorForEach(
    dst.extent(),
    func
  );
}

/// Fills a tensor with a uniform value
template <
  typename Element,                                                   ///< Element type
  typename Layout>                                                    ///< Layout function
void TensorFill(
  TensorViewPlanarComplex<Element, Layout> dst,                       ///< destination tensor 
  cutlass::complex<Element> val = cutlass::complex<Element>(0)) {     ///< value to uniformly fill it with

  TensorFill(dst.view_real(), val.real());
  TensorFill(dst.view_imag(), val.imag());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element>
struct RandomGaussianFunc {

  uint64_t seed;
  double mean;
  double stddev;
  int int_scale;
  double pi;

  //
  // Methods
  //
  RandomGaussianFunc(
    uint64_t seed_ = 0, 
    double mean_ = 0, 
    double stddev_ = 1,
    int int_scale_ = -1
  ):
    seed(seed_), mean(mean_), stddev(stddev_), int_scale(int_scale_), pi(std::acos(-1)) {
      std::srand((unsigned)seed);
  }

  /// Compute random value and update RNG state
  Element operator()() const {

    // Box-Muller transform to generate random numbers with Normal distribution
    double u1 = double(std::rand()) / double(RAND_MAX);
    double u2 = double(std::rand()) / double(RAND_MAX);

    // Compute Gaussian random value
    double rnd = std::sqrt(-2 * std::log(u1)) * std::cos(2 * pi * u2);
    rnd = mean + stddev * rnd;

    // Scale and convert final result
    Element result;

    if (int_scale >= 0) {
      rnd = double(int64_t(rnd * double(1 << int_scale))) / double(1 << int_scale);
      result = static_cast<Element>(rnd);
    }
    else {
      result = static_cast<Element>(rnd);
    }

    return result;
  }
};

/// Partial specialization for initializing a complex value.
template <typename Element>
struct RandomGaussianFunc<complex<Element> > {

  uint64_t seed;
  double mean;
  double stddev;
  int int_scale;
  double pi;

  //
  // Methods
  //
  RandomGaussianFunc(
    uint64_t seed_ = 0, 
    double mean_ = 0, 
    double stddev_ = 1,
    int int_scale_ = -1
  ):
    seed(seed_), mean(mean_), stddev(stddev_), int_scale(int_scale_), pi(std::acos(-1)) {
      std::srand((unsigned)seed);
  }

  /// Compute random value and update RNG state
  complex<Element> operator()() const {

    Element reals[2];

    for (int i = 0; i < 2; ++i) {
      // Box-Muller transform to generate random numbers with Normal distribution
      double u1 = double(std::rand()) / double(RAND_MAX);
      double u2 = double(std::rand()) / double(RAND_MAX);

      // Compute Gaussian random value
      double rnd = std::sqrt(-2 * std::log(u1)) * std::cos(2 * pi * u2);
      rnd = mean + stddev * rnd;

      if (int_scale >= 0) {
        rnd = double(int(rnd * double(1 << int_scale)));
        reals[i] = from_real<Element>(rnd / double(1 << int_scale));
      }
      else {
        reals[i] = from_real<Element>(rnd);
      }
    }

    return complex<Element>(reals[0], reals[1]);
  }
};

/// Partial specialization for initializing a complex value.
template <typename Element>
struct RandomGaussianFunc<Quaternion<Element> > {

  uint64_t seed;
  double mean;
  double stddev;
  int int_scale;
  double pi;

  //
  // Methods
  //
  RandomGaussianFunc(
    uint64_t seed_ = 0,
    double mean_ = 0,
    double stddev_ = 1,
    int int_scale_ = -1
  ):
    seed(seed_), mean(mean_), stddev(stddev_), int_scale(int_scale_), pi(std::acos(-1)) {
      std::srand((unsigned)seed);
  }

  /// Compute random value and update RNG state
  Quaternion<Element> operator()() const {

    Element reals[4];

    for (int i = 0; i < 4; ++i) {
      // Box-Muller transform to generate random numbers with Normal distribution
      double u1 = double(std::rand()) / double(RAND_MAX);
      double u2 = double(std::rand()) / double(RAND_MAX);

      // Compute Gaussian random value
      double rnd = std::sqrt(-2 * std::log(u1)) * std::cos(2 * pi * u2);
      rnd = mean + stddev * rnd;

      if (int_scale >= 0) {
        rnd = double(int(rnd * double(1 << int_scale)));
        reals[i] = from_real<Element>(rnd / double(1 << int_scale));
      }
      else {
        reals[i] = from_real<Element>(rnd);
      }
    }

    return Quaternion<Element>(reals[0], reals[1], reals[2], reals[3]);
  }
};

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillGaussianFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  RandomGaussianFunc<Element> func;

  //
  // Methods
  //

  /// Construction of Gaussian RNG functor.
  TensorFillGaussianFunc(
    TensorView view_ = TensorView(),
    RandomGaussianFunc<Element> func_ = RandomGaussianFunc<Element>()
  ):
    view(view_), func(func_) {

  }

  /// Compute random value and update RNG state
  void operator()(Coord<Layout::kRank> const &coord) const {
    view.at(coord) = func();
  }
};

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillSymmetricGaussianFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  RandomGaussianFunc<Element> func;
  cutlass::FillMode fill_mode;

  //
  // Methods
  //

  /// Construction of Gaussian RNG functor.
  TensorFillSymmetricGaussianFunc(
    TensorView view_ = TensorView(),
    RandomGaussianFunc<Element> func_ = RandomGaussianFunc<Element>(),
    cutlass::FillMode fill_mode_ = cutlass::FillMode::kInvalid
  ):
    view(view_), func(func_), fill_mode(fill_mode_) {

  }

  /// Compute random value and update RNG state
  void operator()(Coord<Layout::kRank> const &coord) const {
    // Fill half of matrix based on FillMode
    if (Layout::kRank == 2 && 
        fill_mode == cutlass::FillMode::kLower &&
        coord[0] >= coord[1]) {
      view.at(coord) = func();
    } else if (Layout::kRank == 2 && 
        fill_mode == cutlass::FillMode::kUpper &&
        coord[0] <= coord[1]) {
      view.at(coord) = func();
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a Gaussian distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomGaussian(
  TensorView<Element, Layout> dst,        ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  double mean = 0,                        ///< Gaussian distribution's mean
  double stddev = 1,                      ///< Gaussian distribution's standard deviation
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  
  detail::RandomGaussianFunc<Element> random_func(seed, mean, stddev, bits);

  detail::TensorFillGaussianFunc<Element, Layout> func(
    dst,
    random_func
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

/// Fills a tensor with random values with a Gaussian distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomGaussian(
  TensorViewPlanarComplex<Element, Layout> dst,        ///< destination tensor
  uint64_t seed,                                       ///< seed for RNG
  double mean = 0,                                     ///< Gaussian distribution's mean
  double stddev = 1,                                   ///< Gaussian distribution's standard deviation
  int bits = -1) {                                     ///< If non-negative, specifies number of fractional bits that 
                                                       ///  are not truncated to zero. Permits reducing precision of
                                                       ///  data.
  
  TensorFillRandomGaussian(dst.view_real(), seed, mean, stddev, bits);
  TensorFillRandomGaussian(dst.view_imag(), ~seed, mean, stddev, bits);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Fills a tensor with random values with a Gaussian distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillSymmetricRandomGaussian(
  TensorView<Element, Layout> dst,        ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  cutlass::FillMode fill_mode,            ///< FillMode for symmetric matrices
  double mean = 0,                        ///< Gaussian distribution's mean
  double stddev = 1,                      ///< Gaussian distribution's standard deviation
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.

  detail::RandomGaussianFunc<Element> random_func(seed, mean, stddev, bits);

  detail::TensorFillSymmetricGaussianFunc<Element, Layout> func(
    dst,
    random_func,
    fill_mode
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a Gaussian distribution.
template <
  typename Element                        ///< Element type
>
void BlockFillRandomGaussian(
  Element *ptr,                           ///< destination buffer
  size_t capacity,                        ///< number of elements
  uint64_t seed,                          ///< seed for RNG
  double mean = 0,                        ///< Gaussian distribution's mean
  double stddev = 1,                      ///< Gaussian distribution's standard deviation
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  

  detail::RandomGaussianFunc<Element> random_func(seed, mean, stddev, bits);

  for (size_t i = 0; i < capacity; ++i) {
    ReferenceFactory<Element>::get(ptr, i) = random_func();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element>
struct RandomUniformFunc {

  using Real = typename RealType<Element>::Type;
  
  uint64_t seed;
  double range;
  double min;
  int int_scale;

  //
  // Methods
  //

  RandomUniformFunc(
    uint64_t seed_ = 0, 
    double max = 1,
    double min_ = 0,
    int int_scale_ = -1
  ):
    seed(seed_), range(max - min_), min(min_), int_scale(int_scale_) {
      std::srand((unsigned)seed);
    }


  /// Compute random value and update RNG state
  Element operator()() const {

    double rnd = double(std::rand()) / double(RAND_MAX);

    rnd = min + range * rnd;

    // Random values are cast to integer after scaling by a power of two to facilitate error
    // testing
    Element result;
    
    if (int_scale >= 0) {
      rnd = double(int64_t(rnd * double(1 << int_scale))) / double(1 << int_scale);
      result = static_cast<Element>(Real(rnd));
    }
    else {
      result = static_cast<Element>(Real(rnd));
    }

    return result;
  }
};

/// Partial specialization for initializing a complex value.
template <typename Element>
struct RandomUniformFunc<complex<Element> > {

  using Real = typename RealType<Element>::Type;
  
  uint64_t seed;
  double range;
  double min;
  int int_scale;

  //
  // Methods
  //

  RandomUniformFunc(
    uint64_t seed_ = 0, 
    double max = 1,
    double min_ = 0,
    int int_scale_ = -1
  ):
    seed(seed_), range(max - min_), min(min_), int_scale(int_scale_) {
      std::srand((unsigned)seed);
    }


  /// Compute random value and update RNG state
  complex<Element> operator()() const {

    Element reals[2];

    for (int i = 0; i < 2; ++i) {
      double rnd = double(std::rand()) / double(RAND_MAX);

      rnd = min + range * rnd;

      // Random values are cast to integer after scaling by a power of two to facilitate error
      // testing
      
      if (int_scale >= 0) {
        rnd = double(int(rnd * double(1 << int_scale)));
        reals[i] = from_real<Element>(Real(rnd / double(1 << int_scale)));
      }
      else {
        reals[i] = from_real<Element>(Real(rnd));
      }
    }

    return complex<Element>(reals[0], reals[1]);
  }
};

/// Partial specialization for initializing a Quaternion value.
template <typename Element>
struct RandomUniformFunc<Quaternion<Element> > {

  using Real = typename RealType<Element>::Type;

  uint64_t seed;
  double range;
  double min;
  int int_scale;

  //
  // Methods
  //

  RandomUniformFunc(
    uint64_t seed_ = 0,
    double max = 1,
    double min_ = 0,
    int int_scale_ = -1
  ):
    seed(seed_), range(max - min_), min(min_), int_scale(int_scale_) {
      std::srand((unsigned)seed);
    }


  /// Compute random value and update RNG state
  Quaternion<Element> operator()() const {

    Element reals[4];

    for (int i = 0; i < 4; ++i) {
      double rnd = double(std::rand()) / double(RAND_MAX);

      rnd = min + range * rnd;

      // Random values are cast to integer after scaling by a power of two to facilitate error
      // testing

      if (int_scale >= 0) {
        rnd = double(int(rnd * double(1 << int_scale)));
        reals[i] = from_real<Element>(Real(rnd / double(1 << int_scale)));
      }
      else {
        reals[i] = from_real<Element>(Real(rnd));
      }
    }

    return make_Quaternion(reals[0], reals[1], reals[2], reals[3]);
  }
};

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillRandomUniformFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  RandomUniformFunc<Element> func;

  //
  // Methods
  //

  /// Construction of Gaussian RNG functor.
  TensorFillRandomUniformFunc(
    TensorView view_ = TensorView(),
    RandomUniformFunc<Element> func_ = RandomUniformFunc<Element>()
  ):
    view(view_), func(func_) {

  }

  /// Compute random value and update RNG state
  void operator()(Coord<Layout::kRank> const &coord) const {

    view.at(coord) = func();
  }
};

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillSymmetricRandomUniformFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  RandomUniformFunc<Element> func;
  cutlass::FillMode fill_mode;

  //
  // Methods
  //

  /// Construction of Gaussian RNG functor.
  TensorFillSymmetricRandomUniformFunc(
    TensorView view_ = TensorView(),
    RandomUniformFunc<Element> func_ = RandomUniformFunc<Element>(),
    cutlass::FillMode fill_mode_ = cutlass::FillMode::kInvalid
  ):
    view(view_), func(func_), fill_mode(fill_mode_) {

  }

  /// Compute random value and update RNG state
  void operator()(Coord<Layout::kRank> const &coord) const {
    // Fill half of matrix based on FillMode
    if (Layout::kRank == 2 && 
        fill_mode == cutlass::FillMode::kLower &&
        coord[0] >= coord[1]) {
      view.at(coord) = func();
    } else if (Layout::kRank == 2 && 
        fill_mode == cutlass::FillMode::kUpper &&
        coord[0] <= coord[1]) {
      view.at(coord) = func();
    }
  }
};


//
// We expect to release this with CUTLASS 2.4. -akerr

/// Computes a random Uniform distribution and pads diagonal with zeros
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillPadDiagonalRandomUniformFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  RandomUniformFunc<Element> func;
  cutlass::FillMode fill_mode;
  int alignment;

  //
  // Methods
  //

  /// Construction of Gaussian RNG functor.
  TensorFillPadDiagonalRandomUniformFunc(
    TensorView view_ = TensorView(),
    RandomUniformFunc<Element> func_ = RandomUniformFunc<Element>(),
    cutlass::FillMode fill_mode_ = cutlass::FillMode::kInvalid,
    int alignment_ = 1
  ):
    view(view_), func(func_), fill_mode(fill_mode_), alignment(alignment_) {

  }

  /// Compute random value and update RNG state
  void operator()(Coord<Layout::kRank> const &coord) const {
    // Fill half of matrix based on FillMode
    if (Layout::kRank == 2 && 
        (fill_mode == cutlass::FillMode::kLower) &&
        (coord[0] >= coord[1]) || 
        ((coord[1] - coord[0]) >= alignment)) {
      view.at(coord) = func();
    } else if (Layout::kRank == 2 && 
        fill_mode == cutlass::FillMode::kUpper &&
        (coord[0] <= coord[1]) ||
        ((coord[0] - coord[1]) >= alignment)) {
      view.at(coord) = func();
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomUniform(
  TensorView<Element, Layout> dst,        ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  double max = 1,                         ///< upper bound of distribution
  double min = 0,                         ///< lower bound for distribution
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.                 
  detail::RandomUniformFunc<Element> random_func(seed, max, min, bits);

  detail::TensorFillRandomUniformFunc<Element, Layout> func(
    dst,
    random_func
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomUniform(
  TensorViewPlanarComplex<Element, Layout> dst,        ///< destination tensor
  uint64_t seed,                                       ///< seed for RNG
  double max = 1,                                      ///< upper bound of distribution
  double min = 0,                                      ///< lower bound for distribution
  int bits = -1) {                                     ///< If non-negative, specifies number of fractional bits that 
                                                       ///  are not truncated to zero. Permits reducing precision of
                                                       ///  data.                 
  
  TensorFillRandomUniform(dst.view_real(), seed, max, min, bits);
  TensorFillRandomUniform(dst.view_imag(), ~seed, max, min, bits);
}


/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomUniform(
  TensorView<Quaternion<Element>, Layout> dst,        ///< destination tensor
  uint64_t seed,                                      ///< seed for RNG
  double max = 1,                                     ///< upper bound of distribution
  double min = 0,                                     ///< lower bound for distribution
  int bits = -1) {                                    ///< If non-negative, specifies number of fractional bits that 
                                                      ///  are not truncated to zero. Permits reducing precision of
                                                      ///  data.                 
  detail::RandomUniformFunc<Quaternion<Element>> random_func(seed, max, min, bits);

  detail::TensorFillRandomUniformFunc<Quaternion<Element>, Layout> func(
    dst,
    random_func
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillSymmetricRandomUniform(
  TensorView<Element, Layout> dst,        ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  cutlass::FillMode fill_mode,            ///< FillMode for symmetric matrices
  double max = 1,                         ///< upper bound of distribution
  double min = 0,                         ///< lower bound for distribution
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.

  detail::RandomUniformFunc<Element> random_func(seed, max, min, bits);

  detail::TensorFillSymmetricRandomUniformFunc<Element, Layout> func(
    dst,
    random_func,
    fill_mode
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

/// Fills a tensor with random values with a uniform random distribution pads zeros along diagonal
template <
  typename Element,                       ///< Element type
  typename Layout>                        ///< Layout function
void TensorFillPadDiagonalRandomUniform(
  TensorView<Element, Layout> dst,        ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  cutlass::FillMode fill_mode,            ///< FillMode for symmetric matrices
  double max = 1,                         ///< upper bound of distribution
  double min = 0,                         ///< lower bound for distribution
  int bits = -1,                          ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  int alignment = 1 
) {

  detail::RandomUniformFunc<Element> random_func(seed, max, min, bits);

  detail::TensorFillPadDiagonalRandomUniformFunc<Element, Layout> func(
    dst,
    random_func,
    fill_mode,
    alignment
  );

  TensorForEach(
    dst.extent(),
    func
  );
}
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element                        ///< Element type
>
void BlockFillRandomUniform(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                          ///< seed for RNG
  double max = 1,                         ///< upper bound of distribution
  double min = 0,                         ///< lower bound for distribution
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.                 
  detail::RandomUniformFunc<Element> random_func(seed, max, min, bits);

  for (size_t i = 0; i < capacity; ++i) {
    ReferenceFactory<Element>::get(ptr, i) = random_func();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillDiagonalFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  Element diag;
  Element other;

  //
  // Methods
  //

  TensorFillDiagonalFunc(
    TensorView const &view_ = TensorView(),
    Element diag_ = Element(1),
    Element other_ = Element(0)
  ):
    view(view_), diag(diag_), other(other_) { }

  void operator()(Coord<Layout::kRank> const & coord) const {
    bool is_diag = true;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[i - 1]) {
        is_diag = false;
        break;
      }
    }

    view.at(coord) = (is_diag ? diag : other);
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor everywhere with a unique value for its diagonal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillDiagonal(
  TensorView<Element, Layout> dst,        ///< destination tensor
  Element diag = Element(1),              ///< value to write in the diagonal
  Element other = Element(0)) {           ///< value to write off the diagonal

  detail::TensorFillDiagonalFunc<Element, Layout> func(
    dst,
    diag,
    other
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to fill a tensor's digonal with 1 and 0 everywhere else.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillIdentity(
  TensorView<Element, Layout> dst) {               ///< destination tensor

  TensorFillDiagonal(dst, Element(1), Element(0));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Writes a uniform value to the diagonal of a tensor without modifying off-diagonal elements.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorUpdateDiagonal(
  TensorView<Element, Layout> dst,                 ///< destination tensor
  Element val = Element(1)) {

  typename Layout::Index extent = dst.extent().min();

  for (typename Layout::Index i = 0; i < extent; ++i) {
    Coord<Layout::kRank> coord(i);
    dst.at(coord) = val;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorUpdateOffDiagonalFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  Element other;

  //
  // Methods
  //

  TensorUpdateOffDiagonalFunc(
    TensorView const &view_ = TensorView(),
    Element other_ = Element(0)
  ):
    view(view_), other(other_) { }

  void operator()(Coord<Layout::kRank> const & coord) const {
    bool is_diag = true;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[i - 1]) {
        is_diag = false;
        break;
      }
    }

    if (!is_diag) {
      view.at(coord) = other;
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Writes a uniform value to all elements in the tensor without modifying diagonal elements.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorUpdateOffDiagonal(
  TensorView<Element, Layout> dst,      ///< destination tensor
  Element other = Element(1)) {

  detail::TensorUpdateOffDiagonalFunc<Element, Layout> func(
    dst,
    other
  );

  TensorForEach(
    dst.extent(),
    func
  );
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillLinearFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  Array<Element, Layout::kRank> v;
  Element s;

  //
  // Methods
  //
  
  TensorFillLinearFunc() { }

  /// Constructs functor
  TensorFillLinearFunc(
    TensorView const &view_,
    Array<Element, Layout::kRank> const & v_,
    Element s_ = Element(0)
  ):
    view(view_), v(v_), s(s_) { }

  /// Updates the tensor
  void operator()(Coord<Layout::kRank> const & coord) const {
    
    Element sum(s);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Layout::kRank; ++i) {
      sum += Element(coord[i]) * v[i];
    }

    view.at(coord) = sum;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills tensor with a linear combination of its coordinate and another vector
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillLinear(
  TensorView<Element, Layout> dst,      ///< destination tensor
  Array<Element, Layout::kRank> const & v,
  Element s = Element(0)) {

  detail::TensorFillLinearFunc<Element, Layout> func(
    dst,
    v,
    s
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills tensor with a linear combination of its coordinate and another vector
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillSequential(
  TensorView<Element, Layout> dst,     ///< destination tensor
  Element s = Element(0)) {

  Array<Element, Layout::kRank> stride;

  stride[0] = Element(1);

  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < Layout::kRank; ++i) {
    stride[i] = stride[i - 1] * Element(dst.extent()[i - 1]);
  }

  TensorFillLinear(dst, stride, s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a block of data with sequential elements
template <
  typename Element
>
void BlockFillSequential(
  Element *ptr,
  int64_t capacity,
  Element v = Element(1),
  Element s = Element(0)) {
  int i = 0;

  while (i < capacity) {
    cutlass::ReferenceFactory<Element, (cutlass::sizeof_bits<Element>::value <
                                        8)>::get(ptr, i) = s;

    s = Element(s + v);
    ++i;
  }
}

/// Fills a block of data with sequential elements
template <
  typename Element
>
void BlockFillSequentialModN(
  Element *ptr,
  int64_t capacity,
  int64_t mod,
  int64_t v = int64_t(1),
  int64_t s = int64_t(0)) {
  int i = 0;

  while (i < capacity) {
    cutlass::ReferenceFactory<Element, (cutlass::sizeof_bits<Element>::value <
                                        8)>::get(ptr, i) = Element(s);

    s = int64_t(s + v) % mod;
    ++i;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a block of data with sequential elements
template <
  typename Element
>
void BlockFillRandom(
  Element *ptr,
  size_t capacity,
  uint64_t seed,
  Distribution dist) {

  if (dist.kind == Distribution::Gaussian) {
    BlockFillRandomGaussian<Element>(
      ptr, 
      capacity, 
      seed, 
      dist.gaussian.mean, 
      dist.gaussian.stddev, 
      dist.int_scale);
  }
  else if (dist.kind == Distribution::Uniform) {
    BlockFillRandomUniform<Element>(
      ptr, 
      capacity, 
      seed, 
      dist.uniform.max,
      dist.uniform.min, 
      dist.int_scale);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element>
struct RandomSparseMetaFunc {
  
  uint64_t seed;
  double range;
  int MetaSizeInBits;

  //
  // Methods
  //

  RandomSparseMetaFunc(
    uint64_t seed_ = 0, 
    int MetaSizeInBits_ = 2
  ):
    seed(seed_), MetaSizeInBits(MetaSizeInBits_) {
      std::srand((unsigned)seed);
      if (MetaSizeInBits_ == 2) {
        range = 6;
      } else if (MetaSizeInBits_ == 4) {
        range = 2;
      }
    }

  /// Compute random value and update RNG state
  Element operator()() const {
    Element FourToTwoMeta[6] = {0x4, 0x8, 0x9, 0xc, 0xd, 0xe};
    Element TwoToOneMeta[2] = {0x4, 0xe};

    Element * MetaArray = (MetaSizeInBits == 2) ? FourToTwoMeta : TwoToOneMeta;

    Element result = 0x0;

    for (int i = 0; i < cutlass::sizeof_bits<Element>::value / 4; ++i) {
      double rnd = double(std::rand()) / double(RAND_MAX);
      rnd = range * rnd;
      Element meta = MetaArray[(int)rnd];

      result = (Element)(result | ((Element)(meta << (i * 4))));
    }

    return result;
  }
};

/// Computes a random sparse meta
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillRandomSparseMetaFunc {

  using TensorView = TensorView<Element, Layout>;

  //
  // Data members
  //

  TensorView view;
  RandomSparseMetaFunc<Element> func;

  //
  // Methods
  //

  /// Construction of Gaussian RNG functor.
  TensorFillRandomSparseMetaFunc(
    TensorView view_ = TensorView(),
    RandomSparseMetaFunc<Element> func_ = RandomSparseMetaFunc<Element>()
  ):
    view(view_), func(func_) {

  }

  /// Compute random value and update RNG state
  void operator()(Coord<Layout::kRank> const &coord) const {

    view.at(coord) = func();
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,                 ///< Element type
  typename Layout>                  ///< Layout function
void TensorFillRandomSparseMeta(
  TensorView<Element, Layout> dst,  ///< destination tensor
  uint64_t seed,                    ///< seed for RNG
  int MetaSizeInBits) {             ///< 2 bit or 4 bit

  detail::RandomSparseMetaFunc<Element> random_func(seed, MetaSizeInBits);

  detail::TensorFillRandomSparseMetaFunc<Element, Layout> func(
    dst,
    random_func
  );

  TensorForEach(
    dst.extent(),
    func
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element                        ///< Element type
>
void BlockFillRandomSparseMeta(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                          ///< seed for RNG
  int MetaSizeInBits) {                   ///< 2 bit or 4bit

  detail::RandomSparseMetaFunc<Element> random_func(seed, MetaSizeInBits);

  for (size_t i = 0; i < capacity; ++i) {
    ptr[i] = random_func();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies a diagonal in from host memory without modifying off-diagonal elements.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorCopyDiagonalIn(
  TensorView<Element, Layout> dst,          ///< destination tensor
  Element const *ptr) {                     ///< dense buffer of elements

  typename Layout::Index extent = dst.extent().min();
  
  for (typename Layout::Index i = 0; i < extent; ++i) {
    Coord<Layout::kRank> coord(i);
    dst.at(coord) = ReferenceFactory<Element>::get(ptr, i);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies the diagonal of a tensor into a dense buffer in host memory.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorCopyDiagonalOut(
  Element *ptr,                               ///< dense buffer of elements
  TensorView<Element, Layout> src) {          ///< source tensor

  typename Layout::Index extent = src.extent().min();
  
  for (typename Layout::Index i = 0; i < extent; ++i) {
    Coord<Layout::kRank> coord(i);
    ReferenceFactory<Element>::get(ptr, i) = src.at(coord);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass

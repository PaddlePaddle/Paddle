/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination with a maximum operation used by
  epilogues.
*/

#pragma once
#include <cuda.h>
#include <cutlass/half.h>
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Single source of truth for whether to unroll for `LinearCombinationClamp()`
constexpr bool LinearCombinationFtGeluIsHeavy() { return false; }

}  // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__inline__ __device__ float tanh_opt(float x) {
#if (__CUDA_ARCH__ >= 750)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  const float exp_val = -1.f * fabs(2 * x);
  return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// GELU operator implemented using the Taylor series approximation
template <typename T>
struct FtGelu {
  static const bool kIsHeavy = true;
  CUTLASS_DEVICE
  T operator()(T const &z) const {
    T k0 = static_cast<float>(0.7978845608028654);
    T k1 = static_cast<float>(0.044715);

    return T(cutlass::constants::half<T>() * z *
             (cutlass::constants::one<T>() +
              fast_tanh(k0 * z * (cutlass::constants::one<T>() + k1 * z * z))));
  }
};

template <>
struct FtGelu<float> {
  static const bool kIsHeavy = true;
  CUTLASS_DEVICE
  float operator()(float const &z) const {
    float k0 = static_cast<float>(0.7978845608028654);
    float k1 = static_cast<float>(0.044715);

    return float(
        z *
        (cutlass::constants::one<float>() +
         tanh_opt(k0 * z * (cutlass::constants::one<float>() + k1 * z * z))));
  }
};

template <int N>
struct FtGelu<Array<half_t, N>> {
  static const bool kIsHeavy = true;
  CUTLASS_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &z) const {
    using T = half_t;
    Array<half_t, N> y;

    half_t k0 = half_t(0.7978845608028654);
    half_t k1 = half_t(0.044715);

    multiply_add<Array<half_t, N>> fma;
    multiplies<Array<half_t, N>> mul;
    plus<Array<half_t, N>> add;

    fast_tanh_op<Array<half_t, N>> tanh;

    Array<half_t, N> u =
        mul(mul(k0, z), fma(mul(k1, z), z, cutlass::constants::one<T>()));

    y = mul(mul(z, cutlass::constants::half<T>()),
            add(cutlass::constants::one<T>(), tanh(u)));

    return y;
  }
};

template <typename T, int N>
struct FtGelu<Array<T, N>> {
  static const bool kIsHeavy = true;
  CUTLASS_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    FtGelu<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(rhs[i]);
    }

    return y;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
    typename ElementOutput_,  ///< Data type used to load and store tensors
    int Count,                ///< Number of elements computed per operation
                ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                ///< but we use 64 or 32 sometimes when there are not enough
                ///< data to store
    typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_,  ///< Data type used to compute linear combination
    ScaleType::Kind Scale =
        ScaleType::Default,  ///< Control Alpha and Beta scaling
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class LinearCombinationFtGelu {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentScaleBias = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  static bool const kIsHeavy = detail::LinearCombinationFtGeluIsHeavy();

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute alpha;             ///< scales accumulators
    ElementCompute beta;              ///< scales source tensor
    ElementCompute threshold;         ///< minimum value that is output
    ElementCompute const *alpha_ptr;  ///< pointer to accumulator scalar - if
                                      ///< not null, loads it from memory
    ElementCompute const *beta_ptr;   ///< pointer to source scalar - if not
                                      ///< null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          threshold(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha,
           ElementCompute beta = ElementCompute(0),
           ElementCompute threshold = ElementCompute(0))
        : alpha(alpha),
          beta(beta),
          threshold(threshold),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr,
           ElementCompute const *beta_ptr = nullptr,
           ElementCompute threshold = ElementCompute(0))
        : alpha(0),
          beta(0),
          threshold(threshold),
          alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementCompute threshold_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  explicit LinearCombinationFtGelu(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    threshold_ = params.threshold;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementCompute const &>(allones);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;
    GELU<FragmentCompute> ftgelu;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate =
          mul_add_accumulator(alpha_,
                              converted_accumulator,
                              intermediate);  // D = alpha * Accum + X
    } else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate =
          mul_add_source(beta_, converted_source);  // X =  beta * C + uniform
      intermediate =
          mul_add_accumulator(alpha_,
                              converted_accumulator,
                              intermediate);  // D = alpha * Accum + X
    }

    // Compute threshold optionally
    intermediate = ftgelu(intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_accumulator;
    GELU<FragmentCompute> ftgelu;

    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate =
          mul_accumulator(alpha_, converted_accumulator);  // D = alpha * Accum
    }

    // Compute threshold optionally
    intermediate = ftgelu(intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes per-channel linear scaling and bias : D = scale * accumulator +
  /// bias Scale and Bias are from input Fragment
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentScaleBias const &scale,
                            FragmentScaleBias const &bias) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform per-channel scale and bias
    FragmentCompute intermediate;

    multiply_add<FragmentCompute> mul_add_accumulator;

    if (Scale == ScaleType::OnlyAlphaPerChannelScaling)
      intermediate = mul_add_accumulator(
          scale, converted_accumulator, bias);  // D = scale * Accum + bias
    else
      intermediate = mul_add_accumulator(
          alpha_, converted_accumulator, bias);  // D = alpha * Accum + bias

    GELU<FragmentCompute> ftgelu;

    // Compute threshold optionally
    intermediate = ftgelu(intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conditional guards to enable partial specialization for packed integers
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \
    ((__CUDACC_VER_MAJOR__ > 10) ||                     \
     ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
/// Special handling for int types

template <typename ElementOutput_,  ///< Data type used to load and store
                                    ///< tensors
          int Count,              ///< Number of elements computed per operation
          ScaleType::Kind Scale,  ///< Control Alpha and Beta scaling
          FloatRoundStyle Round>
class LinearCombinationFtGelu<ElementOutput_, Count, int, float, Scale, Round> {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = int;
  using ElementCompute = float;

  static bool const kIsHeavy = detail::LinearCombinationFtGeluIsHeavy();

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentScaleBias = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute alpha;             ///< scales accumulators
    ElementCompute beta;              ///< scales source tensor
    ElementCompute threshold;         ///< minimum value that is output
    ElementCompute const *alpha_ptr;  ///< pointer to accumulator scalar - if
                                      ///< not null, loads it from memory
    ElementCompute const *beta_ptr;   ///< pointer to source scalar - if not
                                      ///< null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          threshold(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha,
           ElementCompute beta = ElementCompute(0),
           ElementCompute threshold = ElementCompute(0))
        : alpha(alpha),
          beta(beta),
          threshold(threshold),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr,
           ElementCompute const *beta_ptr = nullptr,
           ElementCompute threshold = ElementCompute(0))
        : alpha(0),
          beta(0),
          threshold(threshold),
          alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementCompute threshold_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  explicit LinearCombinationFtGelu(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    threshold_ = params.threshold;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementCompute const &>(allones);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;
    GELU<FragmentCompute> ftgelu;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      intermediate =
          mul_add_accumulator(alpha_,
                              converted_accumulator,
                              intermediate);  // D = alpha * Accum + X
    } else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate =
          mul_add_source(beta_, converted_source);  // X =  beta * C + uniform
      intermediate =
          mul_add_accumulator(alpha_,
                              converted_accumulator,
                              intermediate);  // D = alpha * Accum + X
    }

    // Compute threshold optionally
    intermediate = ftgelu(intermediate);

    if (platform::numeric_limits<ElementOutput>::is_integer) {
      // Convert floats back to INT
      FragmentAccumulator scaled_accumulator;

      NumericArrayConverter<int, ElementCompute, kCount, Round>
          compute_converter;

      scaled_accumulator = compute_converter(intermediate);

      // Convert to destination numeric type
      NumericArrayConverter<ElementOutput, int, kCount, Round>
          destination_converter;

      return destination_converter(scaled_accumulator);
    } else {
      NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
          destination_converter;
      return destination_converter(intermediate);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_accumulator;
    GELU<FragmentCompute> ftgelu;

    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      intermediate =
          mul_accumulator(alpha_, converted_accumulator);  // D = alpha * Accum
    }

    // Compute threshold optionally
    intermediate = ftgelu(intermediate);

    if (platform::numeric_limits<ElementOutput>::is_integer) {
      // Convert floats back to INT
      FragmentAccumulator scaled_accumulator;

      NumericArrayConverter<int, ElementCompute, kCount, Round>
          compute_converter;

      scaled_accumulator = compute_converter(intermediate);

      // Convert to destination numeric type
      NumericArrayConverter<ElementOutput, int, kCount, Round>
          destination_converter;

      return destination_converter(scaled_accumulator);
    } else {
      NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
          destination_converter;
      return destination_converter(intermediate);
    }
  }

  /// Computes per-channel linear scaling and bias : D = scale * accumulator +
  /// bias Scale and Bias are from input Fragment
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentScaleBias const &scale,
                            FragmentScaleBias const &bias) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform per-channel scale and bias
    FragmentCompute intermediate;

    multiply_add<FragmentCompute> mul_add_accumulator;

    if (Scale == ScaleType::OnlyAlphaPerChannelScaling)
      intermediate = mul_add_accumulator(
          scale, converted_accumulator, bias);  // D = scale * Accum + bias
    else
      intermediate = mul_add_accumulator(
          alpha_, converted_accumulator, bias);  // D = alpha * Accum + bias

    GELU<FragmentCompute> ftgelu;

    // Compute threshold optionally
    intermediate = ftgelu(intermediate);

    if (platform::numeric_limits<ElementOutput>::is_integer) {
      // Convert floats back to INT
      FragmentAccumulator scaled_accumulator;

      NumericArrayConverter<int, ElementCompute, kCount, Round>
          compute_converter;

      scaled_accumulator = compute_converter(intermediate);

      // Convert to destination numeric type
      NumericArrayConverter<ElementOutput, int, kCount, Round>
          destination_converter;

      return destination_converter(scaled_accumulator);
    } else {
      NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
          destination_converter;
      return destination_converter(intermediate);
    }
  }
};

#endif  // Conditional guards to enable partial specialization for packed
        // integers

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

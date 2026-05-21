// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

/*! \file
  \brief Functor performing linear combination operations used by epilogues.
  It is modified from LinearCombinationGeneric.
*/

#pragma once

#include "hytlass/array.h"
#include "hytlass/epilogue/thread/scale_type.h"
#include "hytlass/functional.h"
#include "hytlass/hytlass.h"
#include "hytlass/numeric_conversion.h"
#include "hytlass/numeric_types.h"

namespace hytlass {
namespace epilogue {
namespace thread {

template <class UnaryOp, class = void>
struct GenericUnaryTraits {
  static constexpr bool IsArgumentsNeeded = false;
  struct Arguments {};
};

template <class UnaryOp>
struct GenericUnaryTraits<UnaryOp,
                          decltype(typename UnaryOp::Arguments(), void())> {
  static constexpr bool IsArgumentsNeeded = true;
  using Arguments = typename UnaryOp::Arguments;
};

/// Applies a linear combination operator followed by an unary function to an
/// array of elements.
///
/// D = unary_op(alpha * accumulator + beta * source)
///
template <
    template <typename T>
    class UnaryOp,
    typename ElementOutput_,  ///< Data type used to load and store tensors
    int ElementsPerAccess,    ///< Number of elements computed per operation
                            ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                            ///< but we use 64 or 32 sometimes when there are
                            ///< not enough data to store
    typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_,  ///< Data type used to compute linear combination
    ScaleType::Kind Scale =
        ScaleType::Default,  ///< Control Alpha and Beta scaling
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
    bool IsHeavy = false>
class LinearCombinationUnary {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using UnaryArguments =
      typename GenericUnaryTraits<UnaryOp<ElementCompute>>::Arguments;

  static bool const kIsHeavy = IsHeavy;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = ElementsPerAccess;
  static const ScaleType::Kind kScale = Scale;

  using FragmentOutput = Array<ElementOutput, kElementsPerAccess>;
  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentSource = Array<ElementOutput, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute alpha;             ///< scales accumulators
    ElementCompute beta;              ///< scales source tensor
    ElementCompute const *alpha_ptr;  ///< pointer to accumulator scalar - if
                                      ///< not null, loads it from memory
    ElementCompute const *beta_ptr;   ///< pointer to source scalar - if not
                                      ///< null, loads it from memory
    UnaryArguments unary_args;

    //
    // Methods
    //

    HYTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    HYTLASS_HOST_DEVICE
    Params(ElementCompute alpha,
           ElementCompute beta = ElementCompute(0),
           UnaryArguments unary_args_ = UnaryArguments{})
        : alpha(alpha),
          beta(beta),
          alpha_ptr(nullptr),
          beta_ptr(nullptr),
          unary_args(unary_args_) {}

    HYTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr,
           ElementCompute const *beta_ptr = nullptr)
        : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {}
  };

 private:
  //
  // Data members
  //

  Params params_;
  bool skip_elementwise_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  HYTLASS_HOST_DEVICE
  explicit LinearCombinationUnary(Params const &params) {
    params_ = params;
    params_.alpha = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    params_.beta = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  /// Returns true if source is needed
  HYTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling)
      return params_.beta != ElementCompute(0);

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return params_.beta != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  HYTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      params_.beta = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  HYTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source) const {
    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute,
                          ElementOutput,
                          kElementsPerAccess,
                          Round>
        source_converter;
    NumericArrayConverter<ElementCompute,
                          ElementAccumulator,
                          kElementsPerAccess,
                          Round>
        accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;
    UnaryOp<ElementCompute> unary_op;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      // D = alpha * Accum + X
      intermediate = mul_add_accumulator(
          params_.alpha, converted_accumulator, intermediate);
    } else if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      // X =  beta * C + uniform
      intermediate = mul_add_source(params_.beta, converted_source);
      // D = alpha * Accum + X
      intermediate = mul_add_accumulator(
          params_.alpha, converted_accumulator, intermediate);
    }

    if constexpr (GenericUnaryTraits<
                      UnaryOp<ElementCompute>>::IsArgumentsNeeded) {
      if (!skip_elementwise_) {
        HYTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = unary_op(intermediate[i], params_.unary_args);
        }
      }
    } else {
      if (!skip_elementwise_) {
        HYTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = unary_op(intermediate[i]);
        }
      }
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput,
                          ElementCompute,
                          kElementsPerAccess,
                          Round>
        destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  HYTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute,
                          ElementAccumulator,
                          kElementsPerAccess,
                          Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_accumulator;
    UnaryOp<ElementCompute> unary_op;

    if (Scale == ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      // D = alpha * Accum
      intermediate = mul_add_accumulator(params_.alpha, converted_accumulator);
    }

    if constexpr (GenericUnaryTraits<
                      UnaryOp<FragmentCompute>>::IsArgumentsNeeded) {
      if (!skip_elementwise_) {
        HYTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = unary_op(intermediate[i], params_.unary_args);
        }
      }
    } else {
      if (!skip_elementwise_) {
        HYTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = unary_op(intermediate[i]);
        }
      }
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput,
                          ElementCompute,
                          kElementsPerAccess,
                          Round>
        destination_converter;

    return destination_converter(intermediate);
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace hytlass

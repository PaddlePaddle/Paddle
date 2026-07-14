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
*/

#pragma once

#include "cutlass_patch/backend.h"

#ifdef __NVCC__
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#elif defined(__HIPCC__)
#include "hytlass/array.h"
#include "hytlass/epilogue/thread/scale_type.h"
#include "hytlass/functional.h"
#include "hytlass/numeric_conversion.h"
#include "hytlass/numeric_types.h"
#endif

#include "cutlass_patch/batched_matrix_coord.h"
#include "cutlass_patch/trace_device.h"

namespace cutlass_patch {
namespace epilogue {
namespace thread {

template <class VariadicOp, class = void>
struct GenericVariadicTraits {
  static constexpr bool IsArgumentsNeeded = false;
  struct Arguments {};
};

template <class VariadicOp>
struct GenericVariadicTraits<VariadicOp,
                             decltype(typename VariadicOp::Arguments(),
                                      void())> {
  static constexpr bool IsArgumentsNeeded = true;
  using Arguments = typename VariadicOp::Arguments;
};

/// Applies a linear combination operator to an array of elements.
///
/// D = VariadicOp(alpha * accumulator + beta * source)
///
template <
    template <typename T>
    class VariadicOp,
    typename ElementOutput_,  ///< Data type used to load and store tensors
    int ElementsPerAccess,    ///< Number of elements computed per operation.
                            ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                            ///< but we use 64 or 32 sometimes when there are
                            ///< not enough data to store
    typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_,  ///< Data type used to compute linear combination
    cutlass::epilogue::thread::ScaleType::Kind Scale =
        cutlass::epilogue::thread::ScaleType::Default,  ///< Control Alpha and
                                                        ///< Beta scaling
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest,
    bool IsHeavy = false>
class LinearCombinationVariadic {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using VariadicArguments =
      typename GenericVariadicTraits<VariadicOp<ElementCompute>>::Arguments;

  static bool const kIsHeavy = IsHeavy;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = ElementsPerAccess;
  static const cutlass::epilogue::thread::ScaleType::Kind kScale = Scale;

  using FragmentOutput = cutlass::Array<ElementOutput, kElementsPerAccess>;
  using FragmentAccumulator =
      cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentSource = cutlass::Array<ElementOutput, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;

  static cutlass::FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute alpha;             ///< scales accumulators
    ElementCompute beta;              ///< scales source tensor
    ElementCompute const *alpha_ptr;  ///< pointer to accumulator scalar - if
                                      ///< not null, loads it from memory
    ElementCompute const *beta_ptr;   ///< pointer to source scalar - if not
                                      ///< null, loads it from memory
    VariadicArguments variadic_args;

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha,
           ElementCompute beta,
           VariadicArguments variadic_args_ = VariadicArguments{})
        : alpha(alpha),
          beta(beta),
          alpha_ptr(nullptr),
          beta_ptr(nullptr),
          variadic_args(variadic_args_) {}
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
  CUTLASS_HOST_DEVICE
  LinearCombinationVariadic(Params const &params) {
    params_ = params;
    params_.alpha = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    params_.beta = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == cutlass::epilogue::thread::ScaleType::NoBetaScaling)
      return params_.beta != ElementCompute(0);

    if (Scale == cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling)
      return false;

    if (Scale == cutlass::epilogue::thread::ScaleType::Nothing) return false;

    return params_.beta != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      params_.beta = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  /// Computes linear scaling with source: D = alpha * accumulator + beta *
  /// source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &source,
                            int row_offset,
                            int column_offset) const {
    CUTLASS_TRACE_DEVICE(
        "kElementsPerAccess: %d, row_offset: %d, column_offset: %d",
        kElementsPerAccess,
        row_offset,
        column_offset);

    // Convert source to internal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute,
                                   ElementOutput,
                                   kElementsPerAccess,
                                   Round>
        source_converter;
    cutlass::NumericArrayConverter<ElementCompute,
                                   ElementAccumulator,
                                   kElementsPerAccess,
                                   Round>
        accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    cutlass::multiplies<FragmentCompute> mul_add_source;
    cutlass::multiply_add<FragmentCompute> mul_add_accumulator;
    VariadicOp<ElementCompute> variadic_op;

    if (Scale == cutlass::epilogue::thread::ScaleType::NoBetaScaling) {
      intermediate = converted_source;
      // D = alpha * Accum + X
      intermediate = mul_add_accumulator(
          params_.alpha, converted_accumulator, intermediate);
    } else if (Scale == cutlass::epilogue::thread::ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      // X =  beta * C + uniform
      intermediate = mul_add_source(params_.beta, converted_source);
      // D = alpha * Accum + X
      intermediate = mul_add_accumulator(
          params_.alpha, converted_accumulator, intermediate);
    }

    if constexpr (GenericVariadicTraits<
                      VariadicOp<ElementCompute>>::IsArgumentsNeeded) {
      if (!skip_elementwise_) {
#if CUTLASS_EPILOGUE_ENABLE_VECTORIZE
        intermediate = variadic_op.Compute<kElementsPerAccess>(
            intermediate,
            params_.variadic_args,
            BatchedMatrixCoord(blockIdx.z, row_offset, column_offset));
#else
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(
              intermediate[i],
              params_.variadic_args,
              BatchedMatrixCoord(blockIdx.z, row_offset, column_offset + i));
        }
#endif
      }
    } else {
      if (!skip_elementwise_) {
#if CUTLASS_EPILOGUE_ENABLE_VECTORIZE
        intermediate = variadic_op.Compute<kElementsPerAccess>(intermediate);
#else
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(intermediate[i]);
        }
#endif
      }
    }

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput,
                                   ElementCompute,
                                   kElementsPerAccess,
                                   Round>
        destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            int row_offset,
                            int column_offset) const {
    CUTLASS_TRACE_DEVICE(
        "kElementsPerAccess: %d, row_offset: %d, column_offset: %d",
        kElementsPerAccess,
        row_offset,
        column_offset);

    // Convert source to internal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute,
                                   ElementAccumulator,
                                   kElementsPerAccess,
                                   Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    cutlass::multiplies<FragmentCompute> mul_accumulator;
    VariadicOp<ElementCompute> variadic_op;

    if (Scale == cutlass::epilogue::thread::ScaleType::Nothing) {
      intermediate = converted_accumulator;
    } else {
      // D = alpha * Accum
      intermediate = mul_accumulator(params_.alpha, converted_accumulator);
    }

    if constexpr (GenericVariadicTraits<
                      VariadicOp<FragmentCompute>>::IsArgumentsNeeded) {
      if (!skip_elementwise_) {
#if CUTLASS_EPILOGUE_ENABLE_VECTORIZE
        intermediate = variadic_op.Compute<kElementsPerAccess>(
            intermediate,
            params_.variadic_args,
            BatchedMatrixCoord(blockIdx.z, row_offset, column_offset));
#else
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(
              intermediate[i],
              params_.variadic_args,
              BatchedMatrixCoord(blockIdx.z, row_offset, column_offset + i));
        }
#endif
      }
    } else {
      if (!skip_elementwise_) {
#if CUTLASS_EPILOGUE_ENABLE_VECTORIZE
        intermediate = variadic_op.Compute<kElementsPerAccess>(intermediate);
#else
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
          intermediate[i] = variadic_op(intermediate[i]);
        }
#endif
      }
    }

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass_patch

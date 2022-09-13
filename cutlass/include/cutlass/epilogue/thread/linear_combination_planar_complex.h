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
  \brief Functor performing linear combination operations on planar-complex arrays
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/complex.h"
#include "cutlass/array_planar_complex.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to arrays of planar-complex elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
/// Note, as with most CUTLASS components for planar complex, the template arguments describe
/// the underlying real data type.
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationPlanarComplex {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = ArrayPlanarComplex<ElementOutput, kCount>;
  using FragmentAccumulator = ArrayPlanarComplex<ElementAccumulator, kCount>;
  using ComputeFragment = ArrayPlanarComplex<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {

    complex<ElementCompute> alpha;                  ///< scales accumulators
    complex<ElementCompute> beta;                   ///< scales source tensor
    complex<ElementCompute> const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    complex<ElementCompute> const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute(0)), 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      complex<ElementCompute> alpha,
      complex<ElementCompute> beta
    ): alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      complex<ElementCompute> const *alpha_ptr,
      complex<ElementCompute> const *beta_ptr
    ): alpha(complex<ElementCompute>()), beta(complex<ElementCompute>()), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

    }
  };

private:

  //
  // Data members
  //

  complex<ElementCompute> alpha_;
  complex<ElementCompute> beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationPlanarComplex(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_.real() != ElementCompute(0) || beta_.imag() != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator, 
    FragmentOutput const &source) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_source(
      source_converter(source.real), 
      source_converter(source.imag));

    ComputeFragment converted_accumulator(
      accumulator_converter(accumulator.real), 
      accumulator_converter(accumulator.imag));

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<Array<ElementCompute, kCount> > mul_op;
    multiply_add<Array<ElementCompute, kCount> > mul_add_op;

    // complex multiply: I = beta * C
    intermediate.real = mul_op(beta_.real(), converted_source.real);
    intermediate.imag = mul_op(beta_.real(), converted_source.imag);

    intermediate.real = mul_add_op(-beta_.imag(), converted_source.imag, intermediate.real);
    intermediate.imag = mul_add_op( beta_.imag(), converted_source.real, intermediate.imag);

    // complex multiply-add: I = alpha * AB + I
    intermediate.real = mul_add_op(alpha_.real(), converted_accumulator.real, intermediate.real);
    intermediate.imag = mul_add_op(alpha_.real(), converted_accumulator.imag, intermediate.imag);

    intermediate.real = mul_add_op(-alpha_.imag(), converted_accumulator.imag, intermediate.real);
    intermediate.imag = mul_add_op( alpha_.imag(), converted_accumulator.real, intermediate.imag);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return FragmentOutput(
      destination_converter(intermediate.real), 
      destination_converter(intermediate.imag));
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator(
      accumulator_converter(accumulator.real), 
      accumulator_converter(accumulator.imag));

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<Array<ElementCompute, kCount> > mul_op;
    multiply_add<Array<ElementCompute, kCount> > mul_add_op;

    // complex multiply-add: I = alpha * AB + I
    intermediate.real = mul_add_op(alpha_.real(), converted_accumulator.real);
    intermediate.imag = mul_add_op(alpha_.real(), converted_accumulator.imag);

    intermediate.real = mul_add_op(-alpha_.imag(), converted_accumulator.imag, intermediate.real);
    intermediate.imag = mul_add_op( alpha_.imag(), converted_accumulator.real, intermediate.imag);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return FragmentOutput(
      destination_converter(intermediate.real), 
      destination_converter(intermediate.imag));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

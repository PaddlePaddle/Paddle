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
    \brief 
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementAccumulator_,
  typename EpilogueOutputOp_
>
struct Gemv {
public:

  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  static_assert(platform::is_same<LayoutA, LayoutA_>::value, 
    "Only supported for column-major A matrix");

  using ElementB = ElementB_;
  using ElementC = ElementC_;

  using ElementAccumulator = ElementAccumulator_;
  using EpilogueOutputOp = EpilogueOutputOp_;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  static int const kThreadCount = 32;
  static int const kStages = 1;

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  static int const kAlignmentC = 1;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    MatrixCoord     problem_size;
    int32_t         batch_count;
    typename EpilogueOutputOp::Params output_op;

    TensorRefA      ref_A;

    ElementB const *ptr_B;
    ElementC const *ptr_C;
    ElementC       *ptr_D;

    int64_t         inc_B;
    int64_t         inc_C;
    int64_t         inc_D;

    int64_t         batch_stride_A;
    int64_t         batch_stride_B;
    int64_t         batch_stride_C;
    int64_t         batch_stride_D;

    //
    // Methods
    //

    Arguments(): batch_count(0) { }

    Arguments(
      MatrixCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params output_op,
      TensorRefA   ref_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      int64_t  inc_B,
      int64_t  inc_C,
      int64_t  inc_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D
    ): 
      problem_size(problem_size),
      batch_count(batch_count),
      output_op(output_op),
      ref_A(ref_A),
      ptr_B(static_cast<ElementB const *>(ptr_B)),
      ptr_C(static_cast<ElementC const *>(ptr_C)),
      ptr_D(static_cast<ElementC       *>(ptr_D)),
      inc_B(inc_B),
      inc_C(inc_C),
      inc_D(inc_D),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D)
    { }

    Arguments(
      MatrixCoord problem_size,
      typename EpilogueOutputOp::Params output_op,
      TensorRefA   ref_A,
      void const * ptr_B,
      void const * ptr_C,
      void       * ptr_D,
      int64_t  inc_B,
      int64_t  inc_C,
      int64_t  inc_D
    ): 
      Arguments(
        problem_size, 
        1, 
        output_op, 
        ref_A, 
        ptr_B, 
        ptr_C, 
        ptr_D,
        inc_B, 
        inc_C, 
        inc_D, 
        1, 
        1, 
        1, 
        1)
    { }

    Status update(Arguments const &args) {
      output_op = args.output_op;
      ref_A = ref_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;

      return Status::kSuccess;
    }
  };

  using Params = Arguments;

  /// Shared memory storage structure
  union SharedStorage {

  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  Gemv() { } 

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::MatrixCoord const & problem_size) {

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }
 
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Loop over batch indices
    for (int batch_idx = blockIdx.z; batch_idx < params.batch_count; batch_idx += gridDim.z) {

      int i = blockIdx.x * kThreadCount + threadIdx.x;

      ElementA const *ptr_A = params.ref_A.data() + i;
      ElementB const *ptr_B = params.ptr_B;

      ptr_A += batch_idx * params.batch_stride_A;
      ptr_B += batch_idx * params.batch_stride_B;

      ElementAccumulator accum = ElementAccumulator();

      // Compute inner product
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k = 0; k < params.problem_size.column(); ++k) {

        // Fetch from A
        ElementA a = ElementA();
        if (i < params.problem_size.row()) {
          a = *ptr_A;
        }
        ptr_A += params.ref_A.stride(0);

        // Fetch from B
        ElementB b = *ptr_B;
        ptr_B += params.inc_B;

        // Math
        accum += ElementAccumulator(a) * ElementAccumulator(b);
      }

      //
      // Epilogue phase
      //

      ElementC const *ptr_C = params.ptr_C + i * params.inc_C + batch_idx * params.batch_stride_C;
      ElementC       *ptr_D = params.ptr_D + i * params.inc_D + batch_idx * params.batch_stride_D;

      EpilogueOutputOp output_op(params.output_op);

      typename EpilogueOutputOp::FragmentAccumulator accum_fragment;
      typename EpilogueOutputOp::FragmentOutput      source_fragment;
      typename EpilogueOutputOp::FragmentOutput      output_fragment;
      
      accum_fragment[0] = accum;

      if (i < params.problem_size.row()) {
        if (output_op.is_source_needed()) {
          source_fragment[0] = *ptr_C;
          output_fragment = output_op(accum_fragment, source_fragment);
        }
        else {
          output_fragment = output_op(accum_fragment);
        }

        *ptr_D = output_fragment[0];
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

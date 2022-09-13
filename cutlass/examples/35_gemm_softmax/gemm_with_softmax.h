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

/**

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "epilogue_with_visitor.h"
#include "gemm_with_epilogue_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernel computes partial reduction
//
//
// 2. Sum[m, n'] = sum_n(exp(D[m, n] - N[m, 0]))
//
template <
  typename ElementD_,
  typename ElementN_,
  typename ElementSum_,
  typename ElementSoft_,
  int Alignment,
  typename Shape_ = MatrixShape<4, 16>
>
class ApplySoftmax {
public:

  using ElementD = ElementD_;
  using ElementN = ElementN_;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoft_;

  static int const kAlignment = Alignment;
  using Shape = Shape_;

  using Layout = cutlass::layout::RowMajor;

  using TensorRefD = TensorRef<ElementD, Layout>;
  using TensorRefN = TensorRef<ElementN, Layout>;
  using TensorRefSum = TensorRef<ElementSum, Layout>;
  using TensorRefSoft = TensorRef<ElementSoft, Layout>;

  using FragmentSum = Array<ElementSum, kAlignment>;

  //
  // Arguments
  //

  struct Arguments {

    MatrixCoord     extent;             ///< Extent of D and Softmax matrices
    int             batch_count;        ///< Batch count
    TensorRefD      ref_D;              ///< D matrix computed by GEMM+Max (input)
    TensorRefN      ref_N;              ///< Norm tensor (input)
    TensorRefSoft   ref_Soft;           ///< Softmax tensor (output)
    int64_t         batch_stride_D;     ///< Batch stride for D tensor
    int64_t         batch_stride_N;     ///< Batch stride for N tensor
    int64_t         batch_stride_Soft;  ///< Batch stride for softmax tensor

    //
    // Methods
    //
    Arguments():
      batch_count(1),
      batch_stride_D(0),
      batch_stride_N(0),
      batch_stride_Soft(0)
    { }

    Arguments(
      MatrixCoord     extent_,             ///< Extent of D and Softmax matrices
      int             batch_count_,        ///< Batch count
      TensorRefD      ref_D_,              ///< D matrix computed by GEMM+PartialReduce
      TensorRefN      ref_N_,              ///< Output parameter for N
      TensorRefSoft   ref_Soft_,           ///< Softmax
      int64_t         batch_stride_D_ = 0,
      int64_t         batch_stride_N_ = 0,
      int64_t         batch_stride_Soft_ = 0
    ):
      extent(extent_),
      batch_count(batch_count_),
      ref_D(ref_D_),
      ref_N(ref_N_),
      ref_Soft(ref_Soft_),
      batch_stride_D(batch_stride_D_),
      batch_stride_N(batch_stride_N_),
      batch_stride_Soft(batch_stride_Soft_)
    {

    }
  };

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args_): args(args_) { }
  };

  //
  // SharedStorage
  //

  struct SharedStorage {

    AlignedArray<ElementSum, Shape::kCount> exchange;
    AlignedArray<ElementSum, Shape::kRow> inv_sum;
    AlignedArray<ElementSum, Shape::kRow> norm;

  };

private:

public:

  CUTLASS_DEVICE
  ApplySoftmax() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Phase 1. Reduction over contiguous dimension
    reduce_partial(params, shared_storage);

    __syncthreads();

    // Phase 2. Final reduction within SMEM - yields sum_n(exp(D - N))
    reduce_final(params, shared_storage);

    __syncthreads();

    // Phase 3. Apply
    apply(params, shared_storage);
  }

private:

  /// Partial reduction
  CUTLASS_DEVICE
  void reduce_partial(Params const &params, SharedStorage &shared_storage) {

    //
    // Sum over the matrix
    //
    using AccessTypeD = AlignedArray<ElementD, kAlignment>;

    int block_batch = blockIdx.z;
    int block_m = blockIdx.x * Shape::kRow;
    int block_n = 0;

    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x * kAlignment;

    int idx_m = block_m + thread_m;
    int idx_n = block_n + thread_n;

    AccessTypeD *access_d = reinterpret_cast<AccessTypeD *>(
      params.args.ref_D.data() +
      params.args.batch_stride_D * block_batch +
      params.args.ref_D.layout()({idx_m, idx_n}));

    using ConvertS = cutlass::NumericArrayConverter<ElementSum, ElementD, kAlignment>;

    using Plus = cutlass::plus<FragmentSum>;
    using Minus = cutlass::minus<FragmentSum>;
    using Exp   = cutlass::fast_exp_op<FragmentSum>;

    ConvertS  convert_s;
    Minus     minus;
    Plus      plus;
    Exp       exponential;

    FragmentSum frag_Sum;
    frag_Sum.clear();

    if (idx_m < params.args.extent.row()) {

      // Fetch the norm from GlobalMemory
      ElementN norm = params.args.ref_N.data()[params.args.batch_stride_N * block_batch + idx_m];
      ElementSum norm_cvt = ElementSum(norm);

      FragmentSum norm_vec;

      norm_vec.fill(norm_cvt);
      shared_storage.norm[thread_m] = ElementSum(norm_cvt);

      CUTLASS_PRAGMA_UNROLL
      for (
        int idx = 0;
        idx < params.args.extent.column();
        idx += Shape::kColumn * kAlignment) {

        if (idx_n < params.args.extent.column()) {

          AccessTypeD fetch;
          arch::global_load<AccessTypeD, sizeof(AccessTypeD)>(fetch, access_d, true);

          auto tmp = exponential(minus(convert_s(fetch), norm_vec));

          frag_Sum = plus(frag_Sum, tmp);
        }

        access_d += Shape::kColumn;
        idx_n += Shape::kColumn * kAlignment;
      }

      // Sum the elements owned by one thread
      ElementSum sum = frag_Sum[0];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kAlignment; ++i) {
        sum += frag_Sum[i];
      }

      shared_storage.exchange.data()[threadIdx.x + threadIdx.y * Shape::kColumn] = sum;
    }
  }

  /// Compute the final summation from data in SMEM
  CUTLASS_DEVICE
  void reduce_final(Params const &params, SharedStorage &shared_storage) {

    //
    // SMEM has shape `Shape::Row`-by-`Shape::Column`
    //
    // This computes a reduction across the `Column` dimension yielding a `Row-by-1` vector.
    //

    #if true
    //
    // Tuning parameters tradeoff ILP with latency
    //
    // During each step of the reduction, each thread performs `kAccesses` of vector size `kReduceVector`

    // Tune the number of accesses per reduction
    int const kAccesses = 2;

    // Tune the memory access size
    int const kReduceVector = 4;

    //
    // Static asserts to ensure integrity
    //

    static_assert(kAccesses * kReduceVector,
      "Zero-size steps would infinitely loop.");

    static_assert(
      is_pow2<Shape::kColumn>::value &&
      is_pow2<kAccesses>::value &&
      is_pow2<kReduceVector>::value,
      "Powers of two only.");

    static_assert(!(Shape::kColumn % (kAccesses * kReduceVector)),
      "Divisibility not satisfied");

    //
    // Reduction operators
    //

    using FragmentSum = Array<ElementSum, kReduceVector>;
    using Plus = cutlass::plus<FragmentSum>;

    Plus plus;

    // Tree reduction
    ElementSum *smem_ptr = shared_storage.exchange.data() + threadIdx.y * Shape::kColumn;

    ElementSum final = ElementSum();

    CUTLASS_PRAGMA_UNROLL
    for (
      int tidx_limit = Shape::kColumn / (kAccesses * kReduceVector);
      tidx_limit > 0;
      tidx_limit /= (kAccesses * kReduceVector)) {

      if (threadIdx.x < tidx_limit) {
        FragmentSum fetch;

        arch::shared_load<sizeof(FragmentSum)>(
          &fetch,
          arch::cutlass_get_smem_pointer(smem_ptr + threadIdx.x * kReduceVector));

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kAccesses; ++i) {
          FragmentSum extra;

          arch::shared_load<sizeof(FragmentSum)>(
            &extra,
            arch::cutlass_get_smem_pointer(
              smem_ptr + threadIdx.x * kReduceVector + tidx_limit * kReduceVector * i));

          fetch = plus(fetch, extra);
        }

        // Reduce to one element
        final = fetch[0];

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kReduceVector; ++i) {
          final += fetch[i];
        }
      }
      __syncthreads();

      if (threadIdx.x < tidx_limit) {
        smem_ptr[threadIdx.x] = final;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {

      int const kLgResidual =
        (log2_down<Shape::kColumn>::value % log2_down<kAccesses * kReduceVector>::value);

      // Certain shape combinations require an additional reduction step
      if (kLgResidual) {
        final = ElementSum();

        int const kResidualVector = (1 << kLgResidual);
        Array<ElementSum, kResidualVector> fetch;

        arch::shared_load<sizeof(FragmentSum)>(
          &fetch,
          arch::cutlass_get_smem_pointer(smem_ptr));

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kResidualVector; ++i) {
          final += fetch[i];
        }
      }

      // compute inverse
      ElementSum inv_sum = cutlass::constants::one<ElementSum>() / final;

      // Store to shared memory
      shared_storage.inv_sum[threadIdx.y] = inv_sum;
    }

    #else

    // Primitive serial reduction
    if (threadIdx.x < Shape::kRow && threadIdx.y == 0) {
      ElementSum *smem_ptr = shared_storage.exchange.data() + threadIdx.x * Shape::kColumn;

      ElementSum sum = smem_ptr[0];
      CUTLASS_PRAGMA_UNROLL
      for (int n = 1; n < Shape::kColumn; ++n) {
        sum += smem_ptr[n];
      }

      // compute inverse
      ElementSum inv_sum = cutlass::constants::one<ElementSum>() / sum;

      // Store to shared memory
      shared_storage.inv_sum[threadIdx.x] = inv_sum;
    }
    #endif
  }

  /// Compute Softmax
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;

    int block_batch = blockIdx.z;
    int block_m = blockIdx.x * Shape::kRow;
    int block_n = 0;

    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x * kAlignment;

    int idx_m = block_m + thread_m;
    int idx_n = block_n + thread_n;

    // Kill off thread if it is outside the row boundary
    if (params.args.extent.row() <= idx_m) {
      return;
    }

    //
    // Setup pointers to load D again
    //

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;
    using AccessTypeSoft = AlignedArray<ElementSoft, kAlignment>;
    using FragmentSoft = Array<ElementSoft, kAlignment>;
    using ConvertSum = cutlass::NumericArrayConverter<ElementSum, ElementD, kAlignment>;
    using ConvertSoft = cutlass::NumericArrayConverter<ElementSoft, ElementSum, kAlignment>;

    using Mul = cutlass::multiplies<FragmentSum>;
    using Minus = cutlass::minus<FragmentSum>;
    using Exp   = cutlass::fast_exp_op<FragmentSum>;

    ConvertSum   convert_sum;
    ConvertSoft  convert_soft;

    Minus     minus;
    Mul       mul;
    Exp       exponential;

    AccessTypeD *access_d = reinterpret_cast<AccessTypeD *>(
      params.args.ref_D.data() +
      params.args.batch_stride_D * block_batch +
      params.args.ref_D.layout()({idx_m, idx_n}));

    AccessTypeSoft *access_soft = reinterpret_cast<AccessTypeSoft *>(
      params.args.ref_Soft.data() +
      params.args.batch_stride_Soft * block_batch +
      params.args.ref_Soft.layout()({idx_m, idx_n}));

    // Fetch inv_sum from SharedMemory
    ElementSum inv_sum = shared_storage.inv_sum[thread_m];

    // Fetch the norm from SharedMemory
    ElementSum norm = shared_storage.norm[thread_m];

    //
    // Loop
    //
    CUTLASS_PRAGMA_UNROLL
    for (
      int idx = 0;
      idx < params.args.extent.column();
      idx += Shape::kColumn * kAlignment) {

      if (idx_n < params.args.extent.column()) {

        AccessTypeD fetch;
        arch::global_load<AccessTypeD, sizeof(AccessTypeD)>(fetch, access_d, true);

        FragmentSum result = mul(exponential(minus(convert_sum(fetch), norm)), inv_sum);
        FragmentSoft soft  = convert_soft(result);

        arch::global_store<FragmentSoft, sizeof(FragmentSoft)>(soft, access_soft, true);
      }

      access_d += Shape::kColumn;
      access_soft += Shape::kColumn;
      idx_n += Shape::kColumn * kAlignment;
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ThreadblockShape_,
  int ThreadCount,
  typename OutputTileIterator_,
  typename ElementAccumulator_,
  typename ElementwiseFunctor_
>
class EpilogueVisitorBiasMax {
public:

  using ThreadblockShape   = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementAccumulator = ElementAccumulator_;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;
  using TensorRefD = TensorRef<ElementOutput, LayoutOutput>;

  /// Argument structure
  struct Arguments {

    typename ElementwiseFunctor::Params   elementwise;
    TensorRefD                            ref_C;
    TensorRefD                            ref_D;
    float                                *ptr_Max;
    int64_t                               batch_stride_C;
    int64_t                               batch_stride_D;
    int64_t                               batch_stride_Max;

    //
    // Methods
    //
    Arguments():
      ptr_Max(nullptr),
      batch_stride_C(0),
      batch_stride_D(0),
      batch_stride_Max(0)
    {

    }

    Arguments(
      typename ElementwiseFunctor::Params   elementwise_,
      TensorRefD                            ref_C_,
      TensorRefD                            ref_D_,
      float                                *ptr_Max_,
      int64_t                               batch_stride_C_,
      int64_t                               batch_stride_D_,
      int64_t                               batch_stride_Max_
    ):
      elementwise(elementwise_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      ptr_Max(ptr_Max_),
      batch_stride_C(batch_stride_C_),
      batch_stride_D(batch_stride_D_),
      batch_stride_Max(batch_stride_Max_)
    {

    }
  };

  struct Params {

    typename ElementwiseFunctor::Params   elementwise;
    typename OutputTileIterator::Params   params_C;
    typename OutputTileIterator::Params   params_D;
    typename OutputTileIterator::Element *ptr_C;
    typename OutputTileIterator::Element *ptr_D;
    float                                *ptr_Max;
    int64_t                               batch_stride_C;
    int64_t                               batch_stride_D;
    int64_t                               batch_stride_Max;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params():
      ptr_D(nullptr),
      ptr_Max(nullptr)
    {

    }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args):
      elementwise(args.elementwise),
      params_C(args.ref_C.layout()),
      params_D(args.ref_D.layout()),
      ptr_C(args.ref_C.data()),
      ptr_D(args.ref_D.data()),
      ptr_Max(args.ptr_Max),
      batch_stride_C(args.batch_stride_C),
      batch_stride_D(args.batch_stride_D),
      batch_stride_Max(args.batch_stride_Max)
    {

    }
  };

  /// Shared storage
  struct SharedStorage {
    float reduction[ThreadblockShape::kM];
  };

private:

  Params const &                        params_;
  SharedStorage &                       shared_storage_;
  MatrixCoord                           extent_;
  ElementwiseFunctor                    elementwise_;

  OutputTileIterator                    iterator_C_;
  OutputTileIterator                    iterator_D_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator                    alpha_;
  ElementAccumulator                    beta_;

  ElementAccumulator                    accum_max_;
  int                                   threadblock_row_;

public:

  CUTLASS_DEVICE
  EpilogueVisitorBiasMax(
    Params const &params,                                         ///< Parameters routed to the epilogue
    SharedStorage &shared_storage,                                ///< Shared storage needed by the functors here
    MatrixCoord const &problem_size,                              ///< Problem size of the output
    int thread_idx,                                               ///< Thread index within the threadblock
    int warp_idx,                                                 ///< Warp index within the threadblock
    int lane_idx,                                                 ///< Lane index within the warp
    MatrixCoord const &threadblock_offset = MatrixCoord(0, 0)
  ):
    params_(params),
    shared_storage_(shared_storage),
    extent_(problem_size),
    elementwise_(params.elementwise),
    iterator_C_(params.params_C, params.ptr_C, problem_size, thread_idx, threadblock_offset),
    iterator_D_(params.params_D, params.ptr_D, problem_size, thread_idx, threadblock_offset),
    threadblock_row_(threadblock_offset.row())
  {
    alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr : params.elementwise.alpha);
    beta_ =  (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
    int split_k_index,                                            ///< Index of this threadblock within split-K partitioned scheme
    int split_k_slices) {                                         ///< Total number of split-K slices

  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {
    iterator_C_.add_pointer_offset(batch_idx * params_.batch_stride_C);
    iterator_D_.add_pointer_offset(batch_idx * params_.batch_stride_D);
  }

  /// Called at the start of the epilogue just before iterating over accumulator slices
  CUTLASS_DEVICE
  void begin_epilogue() {

    int const kStoreCount = (ThreadblockShape::kM + kThreadCount - 1) / kThreadCount;

    clear_accum_max_();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kStoreCount; ++i) {
      shared_storage_.reduction[i * kThreadCount + threadIdx.x] = accum_max_;
    }
  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();
    fragment_C_.clear();

    iterator_C_.load(fragment_C_);
    ++iterator_C_;
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {

  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(
    int row_idx,
    int column_idx,
    int frag_idx,
    AccumulatorFragment const &accum) {

    NumericArrayConverter<ElementAccumulator, ElementOutput, kElementsPerAccess> source_converter;
    OutputVector &source_vector = reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

    AccumulatorFragment source = source_converter(source_vector);
    AccumulatorFragment result = alpha_ * accum + beta_ * source;

    MatrixCoord thread_offset =
      iterator_D_.thread_start() +
      OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    bool column_guard = (thread_offset.column() < extent_.column());

    // Compute the maximum within one row
    if (!column_idx) {

      // This is the first fragment in a new row
      if (column_guard) {
        accum_max_ = maximum_accumulator_(accum);
      }
    }
    else {

      // This is an additional fragment in the same row
      if (column_guard) {
        accum_max_ = maximum_accumulator_(accum, accum_max_);
      }
    }

    // If this is the last vector in the row, compute the final max and store it out
    if (column_idx + 1 == OutputTileIterator::ThreadMap::Iterations::kColumn) {

      float float_max_element = float(accum_max_);

      int thread_row = thread_offset.row() - threadblock_row_;

      // Shared memory atomic operation to partially reduce the maximum element
      atomicMax(
        reinterpret_cast<int *>(shared_storage_.reduction + thread_row),
        reinterpret_cast<int const &>(float_max_element)
      );

      clear_accum_max_();
    }

    // Convert to the output
    NumericArrayConverter<ElementOutput, ElementAccumulator, kElementsPerAccess> output_converter;
    OutputVector &output = reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {

  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {

    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {

    __syncthreads();

    int block_batch = blockIdx.z;
    int tidx_m = threadblock_row_ + threadIdx.x;

    float float_max_element = shared_storage_.reduction[threadIdx.x];

    if (tidx_m < extent_.row()) {

      atomicMax(
        reinterpret_cast<int *>(
          params_.ptr_Max +
          params_.batch_stride_Max * block_batch +
          tidx_m),
        reinterpret_cast<int const &>(float_max_element)
      );
    }
  }

private:

  CUTLASS_DEVICE
  void clear_accum_max_() {

    uint32_t float_max_bits = 0xff7fffff;   // -FLT_MAX

    accum_max_ = reinterpret_cast<float const &>(float_max_bits);
  }

  CUTLASS_DEVICE
  float maximum_accumulator_(AccumulatorFragment const &accum) {
    ElementAccumulator max_ = accum[0];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < AccumulatorFragment::kElements; ++i) {
      max_ = fast_max(max_, ElementAccumulator(accum[i]));
    }

    return max_;
  }

  CUTLASS_DEVICE
  ElementAccumulator maximum_accumulator_(AccumulatorFragment const &accum, ElementAccumulator max_) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < AccumulatorFragment::kElements; ++i) {
      max_ = fast_max(max_, ElementAccumulator(accum[i]));
    }

    return max_;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

///
template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename ElementCompute_,
  int Alignment = 128 / cutlass::sizeof_bits<ElementA_>::value,
  typename ElementSum_ = ElementCompute_,
  typename ElementSoftmax_ = ElementC_
>
class GemmSoftmax {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementCompute = ElementCompute_;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoftmax_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

  static int const kAlignment = Alignment;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  /// Linear scaling operator
  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementCompute,
    ElementCompute
  >;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // This is a mandatory data type for the atomic reduction in the GEMM epilogue to function.
  using ElementN = float;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutN = cutlass::layout::RowMajor;
  using LayoutSoft = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;
  using TensorRefN = TensorRef<ElementN, LayoutN>;
  using TensorRefSoft = TensorRef<ElementSoft, LayoutSoft>;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using OperatorClass       = cutlass::arch::OpClassTensorOp;
  using ArchTag             = cutlass::arch::Sm80;
  static int const kStages  = 3;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // basic GEMM kernel
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    ElementA,
    LayoutA,
    kAlignment,
    ElementB,
    LayoutB,
    kAlignment,
    ElementC,
    LayoutC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    ThreadblockSwizzle,
    kStages,
    true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor = kernel::EpilogueVisitorBiasMax<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    ElementCompute,
    EpilogueFunctorOp
  >;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM
  using GemmKernel = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;

  // Softmax kernel
  using SoftmaxApplyKernel = kernel::ApplySoftmax<
    ElementC,
    ElementN,
    ElementSum,
    ElementSoft,
    kAlignment,
    MatrixShape<
      1, 1024
    >
  >;

public:

  /// Arguments class
  struct Arguments {

    typename GemmKernel::Arguments         gemm;
    typename SoftmaxApplyKernel::Arguments softmax;

    //
    // Methods
    //
    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size,
      int32_t    batch_count_,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      TensorRefC ref_C_,
      TensorRefC ref_D_,
      typename EpilogueFunctorOp::Params linear_scaling,
      TensorRefN ref_N_,
      TensorRefSoft ref_Softmax_,
      int64_t batch_stride_A_ = 0,
      int64_t batch_stride_B_ = 0,
      int64_t batch_stride_C_ = 0,
      int64_t batch_stride_D_ = 0,
      int64_t batch_stride_Max_ = 0,
      int64_t batch_stride_Softmax_ = 0
    ):
      gemm(
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size,
        batch_count_,
        ref_A_,
        ref_B_,
        batch_stride_A_,
        batch_stride_B_,
        typename EpilogueVisitor::Arguments(
          linear_scaling,
          ref_C_,
          ref_D_,
          ref_N_.data(),
          batch_stride_C_,
          batch_stride_D_,
          batch_stride_Max_
        )
      ),
      softmax(
        MatrixCoord(problem_size.m(), problem_size.n()),
        batch_count_,
        ref_D_,
        ref_N_,
        ref_Softmax_,
        batch_stride_D_,
        batch_stride_Max_,
        batch_stride_Softmax_
      )
    {

    }
  };

  struct Params {

    typename GemmKernel::Params         gemm;
    typename SoftmaxApplyKernel::Params softmax;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args):
      gemm(args.gemm),
      softmax(args.softmax)
    {

    }
  };

public:

  // Gemm


  //
  // Methods
  //

private:

  Params params_;

public:

  /// Ctor
  GemmSoftmax() {

  }

  /// Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {

    //
    // Launch the GEMM + max kernel
    //

    dim3 gemm_grid = ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cutlass::Kernel<GemmKernel><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the SoftmaxApplyKernel
    //

    dim3 apply_block(SoftmaxApplyKernel::Shape::kColumn, SoftmaxApplyKernel::Shape::kRow);

    int cta_rows = SoftmaxApplyKernel::Shape::kRow;
    int cta_columns = SoftmaxApplyKernel::Shape::kColumn * SoftmaxApplyKernel::kAlignment;

    dim3 apply_grid(
      (params_.softmax.args.extent.row() + cta_rows - 1) / cta_rows,
      (params_.softmax.args.extent.column() + cta_columns - 1) / cta_columns,
      params_.softmax.args.batch_count);

    Kernel<SoftmaxApplyKernel><<<
      apply_grid, apply_block, sizeof(typename SoftmaxApplyKernel::SharedStorage), stream
    >>>(params_.softmax);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

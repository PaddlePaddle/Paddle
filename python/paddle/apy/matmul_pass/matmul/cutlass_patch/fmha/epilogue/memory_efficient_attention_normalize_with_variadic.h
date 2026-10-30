// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// Output op for the fused attention epilogue: wraps the FMHA softmax
// normalization (MemoryEfficientAttentionNormalize) and, on the final key
// block (isLast), applies the AP variadic functor per output element with the
// full (b, m, dv) coordinate. This replaces the two-pass elementwise kernel.

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass_patch/batched_matrix_coord.h"
#include "cutlass_patch/fmha/epilogue/epilogue_pipelined.h"
#include "cutlass_patch/fmha/epilogue/epilogue_rescale_output.h"

namespace cutlass_patch {
namespace epilogue {
namespace thread {

template <typename ElementOutput_,
          typename ElementSource_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_,
          bool isFirst,
          bool isLast,
          typename FragmentAlphaBeta_,
          template <typename T>
          class VariadicFunctor,
          cutlass::FloatRoundStyle Round =
              cutlass::FloatRoundStyle::round_to_nearest>
class MemoryEfficientAttentionNormalizeWithVariadic {
 public:
  using Base = cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
      ElementOutput_,
      ElementSource_,
      Count,
      ElementAccumulator_,
      ElementCompute_,
      isFirst,
      isLast,
      FragmentAlphaBeta_,
      Round>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using FragmentOutput = typename Base::FragmentOutput;
  using FragmentSource = typename Base::FragmentSource;
  using FragmentAccumulator = typename Base::FragmentAccumulator;
  using FragmentAlphaBeta = FragmentAlphaBeta_;
  using Functor = VariadicFunctor<ElementCompute>;
  using VariadicArguments = typename Functor::Arguments;

  static int const kCount = Count;

 private:
  Base base_;
  VariadicArguments variadic_args_;
  int query_start_;

 public:
  // `variadic_args` carries the epilogue operands (host side); `b` / `h` /
  // `query_start` are threadblock constants supplied by the kernel so the
  // functor can index with the global (b, m, dv) coordinate.
  CUTLASS_HOST_DEVICE
  MemoryEfficientAttentionNormalizeWithVariadic(
      FragmentAlphaBeta const& s_prime,
      FragmentAlphaBeta const& m_prime,
      VariadicArguments const& variadic_args,
      int b,
      int h,
      int query_start)
      : base_(s_prime, m_prime),
        variadic_args_(variadic_args),
        query_start_(query_start) {
    variadic_args_.b = b;
    variadic_args_.h = h;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return base_.is_source_needed(); }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    base_.set_k_partition(k_partition, k_partition_count);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(int row,
                            int column,
                            FragmentAccumulator const& accum,
                            FragmentSource const& source) const {
    return ApplyVariadic(base_(row, accum, source), row, column);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(int row,
                            int column,
                            FragmentAccumulator const& accum) const {
    return ApplyVariadic(base_(row, accum), row, column);
  }

 private:
  CUTLASS_HOST_DEVICE
  FragmentOutput ApplyVariadic(FragmentOutput frag, int row, int column) const {
    // Only the final key block writes the real output; intermediate blocks
    // accumulate and must not run the epilogue.
    if (isLast) {
      Functor functor;
      const int m = query_start_ + row;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kCount; ++j) {
        ElementCompute x = static_cast<ElementCompute>(frag[j]);
        cutlass_patch::BatchedMatrixCoord coord(
            variadic_args_.b, m, column + j);
        ElementCompute y = functor(x, variadic_args_, coord);
        frag[j] = static_cast<ElementOutput>(y);
      }
    }
    return frag;
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass_patch

// PLACEHOLDER_APPLY_SPECIALIZATION

namespace cutlass {
namespace epilogue {
namespace threadblock {

// Dispatch for EpiloguePipelined: forward (row, column) to the variadic op.
template <typename EO,
          typename ES,
          int Count,
          typename EA,
          typename EC,
          bool F,
          bool L,
          typename FAB,
          template <typename T>
          class VariadicFunctor,
          FloatRoundStyle R>
struct ApplyEpilogueOp<
    cutlass_patch::epilogue::thread::
        MemoryEfficientAttentionNormalizeWithVariadic<EO,
                                                      ES,
                                                      Count,
                                                      EA,
                                                      EC,
                                                      F,
                                                      L,
                                                      FAB,
                                                      VariadicFunctor,
                                                      R>> {
  using Op = cutlass_patch::epilogue::thread::
      MemoryEfficientAttentionNormalizeWithVariadic<EO,
                                                    ES,
                                                    Count,
                                                    EA,
                                                    EC,
                                                    F,
                                                    L,
                                                    FAB,
                                                    VariadicFunctor,
                                                    R>;
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      int col_id,
      typename Op::FragmentAccumulator const& accum,
      typename Op::FragmentSource const& source) {
    return output_op(row_id, col_id, accum, source);
  }
  static CUTLASS_DEVICE typename Op::FragmentOutput apply(
      Op const& output_op,
      int row_id,
      int col_id,
      typename Op::FragmentAccumulator const& accum) {
    return output_op(row_id, col_id, accum);
  }
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

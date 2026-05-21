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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

*/

#pragma once

#include "hytlass/array.h"
#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"

#include "hytlass/gemm/gemm.h"

#include "hytlass/epilogue/threadblock/default_epilogue_tensor_op.h"
// #include "hytlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "hytlass/epilogue/threadblock/epilogue.h"
#include "hytlass_patch/epilogue/threadblock/epilogue_with_variadic.h"
// #include "hytlass/epilogue/threadblock/epilogue_streamk_with_broadcast.h"

#include "hytlass/layout/permute.h"

namespace hytlass {
namespace epilogue {
namespace threadblock {

/// Defines sensible defaults for epilogues for SimtOps.
template <typename Shape,
          typename WarpMmaSimt,
          typename ElementOutput,
          typename OutputOp,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute,
          conv::StrideSupport StrideSupport = conv::StrideSupport::kUnity,
          int Rank = 4>
struct DefaultEpilogueWithVariadicSimt {
  static conv::StrideSupport const kStrideSupport = StrideSupport;
  static int const kRank = Rank;

  static bool const UseHIPStore =
      platform::is_same<ElementOutput, double>::value;

  /// Use defaults related to the existing epilogue
  using Base =
      DefaultEpilogueSimt<Shape, WarpMmaSimt, OutputOp, ElementsPerAccess>;

  using PackedOutputTileIterator =
      hytlass::epilogue::threadblock::PredicatedTileIterator<
          typename Base::OutputTileThreadMap,
          ElementOutput,
          ScatterD,
          PermuteDLayout,
          UseHIPStore>;

  using StridedOutputTileIterator =
      hytlass::epilogue::threadblock::PredicatedTileIteratorConv<
          typename Base::OutputTileThreadMap,
          ElementOutput,
          ScatterD,
          PermuteDLayout,
          UseHIPStore,
          kRank>;

  //
  // Stores the result z = (y = GEMM(A, B, C), variadic)
  //
  using OutputTileIterator =
      typename platform::conditional<StrideSupport ==
                                         hytlass::conv::StrideSupport::kUnity,
                                     PackedOutputTileIterator,
                                     StridedOutputTileIterator>::type;

  //
  // Define the epilogue
  //
  using Epilogue = hytlass::epilogue::threadblock::EpilogueWithVariadic<
      Shape,
      WarpMmaSimt,
      Base::kPartitionsK,
      OutputTileIterator,
      typename Base::AccumulatorFragmentIterator,
      typename Base::WarpTileIterator,
      typename Base::SharedLoadIterator,
      OutputOp,
      typename Base::Padding>;
};

/// Defines sensible defaults for strided dgrad epilogues for SimtOps.
template <typename Shape,
          typename WarpMmaSimt,
          typename ElementOutput,
          typename OutputOp,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct DefaultEpilogueWithVariadicSimtStridedDgrad {
  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueSimtStridedDgrad<Shape,
                                               WarpMmaSimt,
                                               OutputOp,
                                               ElementsPerAccess>;

  //
  // Stores the result z = (y = GEMM(A, B, C), variadic)
  //
  using OutputTileIterator =
      hytlass::epilogue::threadblock::PredicatedTileIteratorStridedDgrad<
          typename Base::OutputTileThreadMap,
          ElementOutput>;

  //
  // Define the epilogue
  //
  using Epilogue = hytlass::epilogue::threadblock::EpilogueWithVariadic<
      Shape,
      WarpMmaSimt,
      Base::kPartitionsK,
      OutputTileIterator,
      typename Base::AccumulatorFragmentIterator,
      typename Base::WarpTileIterator,
      typename Base::SharedLoadIterator,
      OutputOp,
      typename Base::Padding>;
};

/// Defines sensible defaults for epilogues for TensorOps.
template <typename Shape,
          typename WarpMmaTensorOp,
          int PartitionsK,
          typename ElementOutput,
          typename OutputOp,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct DefaultEpilogueWithVariadicTensorOp {
  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueTensorOp<Shape,
                                       WarpMmaTensorOp,
                                       PartitionsK,
                                       OutputOp,
                                       ElementsPerAccess>;

  //
  // Stores the result z = (y = GEMM(A, B, C), variadic)
  //
  using OutputTileIterator =
      hytlass::epilogue::threadblock::PredicatedTileIterator<
          typename Base::OutputTileThreadMap,
          ElementOutput,
          ScatterD,
          PermuteDLayout>;

  //
  // Define the epilogue
  //
  using Epilogue = hytlass::epilogue::threadblock::EpilogueWithVariadic<
      Shape,
      WarpMmaTensorOp,
      PartitionsK,
      OutputTileIterator,
      typename Base::AccumulatorFragmentIterator,
      typename Base::WarpTileIterator,
      typename Base::SharedLoadIterator,
      OutputOp,
      typename Base::Padding,
      Base::kFragmentsPerIteration>;
};

#if 0
/// Defines sensible defaults for streamk epilogues for TensorOps.
template <
  typename Shape,
  typename WarpMmaTensorOp,
  int PartitionsK,
  typename ElementOutput,
  typename OutputOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultStreamkEpilogueWithVariadicTensorOp {
  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueTensorOp<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess
  >;

  //
  // Stores the result z = (y = GEMM(A, B, C), variadic)
  //
  using OutputTileIterator = hytlass::epilogue
  ::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout
  >;

  //
  // Define the epilogue
  //
  using Epilogue = hytlass::epilogue::threadblock::EpilogueStreamkWithVariadic<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputTileIterator,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding,
    Base::kFragmentsPerIteration
  >;
};
#endif

/// Defines sensible defaults for epilogues for VoltaTensorOps.
template <typename Shape,
          typename WarpMmaTensorOp,
          int PartitionsK,
          typename ElementOutput,
          typename OutputOp,
          int ElementsPerAccess>
struct DefaultEpilogueWithVariadicVoltaTensorOp {
  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueTensorOp<Shape,
                                       WarpMmaTensorOp,
                                       PartitionsK,
                                       OutputOp,
                                       ElementsPerAccess>;

  //
  // Stores the result z = (y = GEMM(A, B, C), variadic)
  //
  using OutputTileIterator = hytlass::epilogue::threadblock::
      PredicatedTileIterator<typename Base::OutputTileThreadMap, ElementOutput>;

  //
  // Define the epilogue
  //
  using Epilogue = hytlass::epilogue::threadblock::EpilogueWithVariadic<
      Shape,
      WarpMmaTensorOp,
      PartitionsK,
      OutputTileIterator,
      typename Base::AccumulatorFragmentIterator,
      typename Base::WarpTileIterator,
      typename Base::SharedLoadIterator,
      OutputOp,
      typename Base::Padding>;
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace hytlass

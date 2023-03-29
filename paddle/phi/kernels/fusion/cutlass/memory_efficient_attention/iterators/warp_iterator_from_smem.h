// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

//  Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
//
//  This source code is licensed under the BSD license found in the
//  LICENSE file in the root directory of this source tree.

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
    \brief Inspired from
   "cutlass/gemm/warp/mma_tensor_op_tile_access_iterator.h" Loads tiles of GEMM
   operands from a RowMajor shared-memory layout into registers to use by A100
   TensorCores.

    The difference with "mma_tensor_op_tile_access_iterator.h" is that:
    (1) We use "ldmatrix" to load tiles, rather than manual loads (slightly
   faster) (2) We support to transpose the operand (eg read `A.transpose()` when
   the shared memory holds `A`)

    This is only implemented for the specific shapes that are interesting for us
*/
#pragma once

#include <cutlass/gemm/gemm.h>

////////////////////////////////////////////////////////////////////////////////
namespace cutlass {
namespace gemm {
namespace warp {

template <
    /// Operand identity
    Operand Operand_,
    /// Data type of A elements
    typename Element_,
    bool kTranspose = false>
class WarpIteratorFromSmem {
 public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = cutlass::MatrixShape<32, 32>;

  /// Operand tag
  static Operand const kOperand = Operand_;

  /// Basic check
  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "WarpIteratorFromSmem may only be instantiated for A or B "
                "operands to warp-level Mma.");

  /// Element type
  using Element = Element_;
  static_assert(sizeof_bits<Element>::value == 16, "Only supported for half");

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = cutlass::MatrixShape<16, 8>;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = 1;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Number of elements accessed per Shared Memory load
  static int const kElementsPerAccess =
      (sizeof_bits<Element>::value >= 32 ? 1
                                         : 32 / sizeof_bits<Element>::value);

  using InstructionCount =
      MatrixShape<Shape::kRow / InstructionShape::kRow,
                  Shape::kColumn / InstructionShape::kColumn>;

  static int const kIterations = (kOperand == Operand::kA)
                                     ? InstructionCount::kColumn
                                     : InstructionCount::kRow;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
      Array<Element,
            (kOperand == Operand::kA)
                ? (Shape::kRow* InstructionShape::kColumn / kThreads)
                : (Shape::kColumn* InstructionShape::kRow / kThreads)>;

  /// Memory access type
  // using AccessType = AlignedArray<Element, kElementsPerAccess>;
  using AccessType = Array<unsigned, 4>;

  static int constexpr kWarpShapeDivisibleInner =
      (kOperand == Operand::kA ? InstructionShape::kColumn
                               : InstructionShape::kRow);
  static int constexpr kAccessesInner =
      (kWarpShapeDivisibleInner / kElementsPerAccess) / 4;
  static int const kTilesPerInstruction = InstructionShape::kRow / 8;

 private:
  /// Underlying tensor reference
  TensorRef ref_;

  /// Origin
  MatrixCoord origin_;

  /// Iterations in a tile
  int iterations_;

 public:
  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  WarpIteratorFromSmem(TensorRef const& ref, int lane_id)
      : WarpIteratorFromSmem(ref, {Shape::kRow, Shape::kColumn}, lane_id) {}
  CUTLASS_HOST_DEVICE
  WarpIteratorFromSmem(TensorRef const& ref, TensorCoord extent, int lane_id)
      : ref_(ref), iterations_(0) {
    int ldsm_vec_num = (lane_id >> 3);
    if (kOperand == Operand::kA) {
      origin_ = MatrixCoord(lane_id % 8, 0);
      static_assert(
          InstructionCount::kRow * kAccessesInner * kTilesPerInstruction == 4,
          "");
      CUTLASS_PRAGMA_UNROLL
      for (int inst_m_idx = 0; inst_m_idx < InstructionCount::kRow;
           ++inst_m_idx) {
        CUTLASS_PRAGMA_UNROLL
        for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx) {
          CUTLASS_PRAGMA_UNROLL
          for (int access_m_idx = 0; access_m_idx < kTilesPerInstruction;
               ++access_m_idx) {
            int access_idx =
                access_m_idx + kTilesPerInstruction *
                                   (inner_idx + kAccessesInner * inst_m_idx);

            MatrixCoord offset(
                access_m_idx * 8 + inst_m_idx * InstructionShape::kRow,
                inner_idx * 4 * kElementsPerAccess);

            if (access_idx == ldsm_vec_num) {
              if (kTranspose) {
                offset = MatrixCoord(offset.column(), offset.row());
              }
              origin_ += offset;
            }
          }
        }
      }
    } else {
      origin_ = MatrixCoord(0, lane_id % 8);
      static_assert(InstructionCount::kColumn * kAccessesInner == 4, "");
      CUTLASS_PRAGMA_UNROLL
      for (int inst_n_idx = 0; inst_n_idx < InstructionCount::kColumn;
           ++inst_n_idx) {
        CUTLASS_PRAGMA_UNROLL
        for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx) {
          int access_idx = inner_idx + kAccessesInner * inst_n_idx;

          MatrixCoord offset(inner_idx * 4 * kElementsPerAccess,
                             inst_n_idx * 8);

          if (access_idx == ldsm_vec_num) {
            if (kTranspose) {
              offset = MatrixCoord(offset.column(), offset.row());
            }
            origin_ += offset;
          }
        }
      }
    }

    ref_.add_coord_offset(origin_);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  WarpIteratorFromSmem& add_tile_offset(TensorCoord const& tile_offset) {
    TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                             tile_offset.column() * Shape::kColumn);
    if (kTranspose) {
      coord_offset = TensorCoord{coord_offset.column(), coord_offset.row()};
    }
    origin_ += coord_offset;

    ref_.add_coord_offset(coord_offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  void advance() {
    if (kOperand == Operand::kA) {
      add_tile_offset({0, 1});
    } else {
      add_tile_offset({1, 0});
    }

    iterations_ = 0;
  }

  /// increase iterations in a tile
  CUTLASS_HOST_DEVICE
  WarpIteratorFromSmem& operator++() {
    iterations_++;

    if (iterations_ >= kIterations) advance();

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_DEVICE
  void load(Fragment& frag) const {  // NOLINT
    AccessType* access_ptr = reinterpret_cast<AccessType*>(&frag);
    using LoadLayout = typename platform::
        conditional<kTranspose, layout::ColumnMajor, layout::RowMajor>::type;

    MatrixCoord offset;
    if (kOperand == Operand::kA) {
      offset = MatrixCoord(0, iterations_ * InstructionShape::kColumn);
    } else {
      offset = MatrixCoord(iterations_ * InstructionShape::kRow, 0);
    }
    if (kTranspose) {
      offset = MatrixCoord(offset.column(), offset.row());
    }
    cutlass::arch::ldsm<LoadLayout, 4>(access_ptr[0],
                                       ref_.data() + ref_.offset(offset));
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass
////////////////////////////////////////////////////////////////////////////////

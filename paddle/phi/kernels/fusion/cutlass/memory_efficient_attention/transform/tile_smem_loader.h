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

#pragma once
#include <cutlass/cutlass.h>
#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

template <typename scalar_t,              // scalar type
          typename ThreadblockTileShape,  // size of tile to load
          int Threads,                    // number of participating threads
          int ElementsPerAccess>          // thread access width in elements
class TileSmemLoader {
 public:
  using SmemTile =
      cutlass::AlignedBuffer<scalar_t, ThreadblockTileShape::kCount>;

  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
      cutlass::layout::PitchLinearShape<
          ThreadblockTileShape::kColumn,  // contiguous
          ThreadblockTileShape::kRow>,    // strided
      Threads,                            // Threads
      ElementsPerAccess>;                 // ElementsPerAccess

  using GmemTileIterator =
      cutlass::transform::threadblock::PredicatedTileIterator<
          ThreadblockTileShape,       // Shape
          scalar_t,                   // Element
          cutlass::layout::RowMajor,  // Layout
          0,                          // AdvanceRank
          ThreadMap>;                 // ThreadMap

  using SmemTileIterator = cutlass::transform::threadblock::RegularTileIterator<
      ThreadblockTileShape,       // Shape
      scalar_t,                   // Element
      cutlass::layout::RowMajor,  // Layout
      0,                          // AdvanceRank
      ThreadMap>;                 // ThreadMap

  using Fragment = typename GmemTileIterator::Fragment;

  /// load a tile from global memory into shared memory
  CUTLASS_DEVICE
  static void load(GmemTileIterator tile_load_iter,
                   SmemTileIterator tile_store_iter) {
    Fragment tb_frag;
    tb_frag.clear();
    tile_load_iter.load(tb_frag);
    tile_store_iter.store(tb_frag);

    __syncthreads();
  }
};

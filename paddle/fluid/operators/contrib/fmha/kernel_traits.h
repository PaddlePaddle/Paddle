/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include "gmem_tile.h"
#include "smem_tile.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int S, int D, int STEP, int WARPS_M, int WARPS_N,
          uint32_t FLAGS = 0x8u>
struct FMHA_kernel_traits {
  // The CTA description for the 1st GEMM.
  using Cta_tile_p = fmha::Cta_tile_extd<STEP, S, D, WARPS_M, WARPS_N, 1>;
  // The CTA description for the 2nd GEMM.
  using Cta_tile_o = fmha::Cta_tile_extd<STEP, D, S, WARPS_M, 1, WARPS_N>;

  // Do we use one buffer for K and V.
  enum { SHARE_SMEM_FOR_K_AND_V = (FLAGS & 0x8u) != 0u };

  // The global memory tile to load Q.
  using Gmem_tile_q =
      fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_A, STEP, D>;

  // The shared memory tile to swizzle Q.
  using Smem_tile_q =
      fmha::Smem_tile_a<Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 1>;

  // The global memory tile to load K.
  using Gmem_tile_k =
      fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_B, S, D>;
  // The shared memory tile to swizzle K.
  using Smem_tile_k = fmha::Smem_tile_b<Cta_tile_p, fmha::Col>;

  // The global memory tile to load V.
  using Gmem_tile_v =
      fmha::Gmem_tile_qkv<Cta_tile_o, fmha::BITS_PER_ELEMENT_B, S, D>;
  // The shared memory tile to swizzle V.
  using Smem_tile_v = fmha::Smem_tile_v<Cta_tile_o>;

  // The global memory tile to store O.
  using Gmem_tile_o = fmha::Gmem_tile_o<Cta_tile_o>;
  // The shared memory tile for O.
  using Smem_tile_o = fmha::Smem_tile_o<Cta_tile_o>;

  // The global memory tile to load/store S.
  using Gmem_tile_s = fmha::Gmem_tile_mma_s<Cta_tile_p>;

  // The shared memory tile to transpose S.
  using Smem_tile_st = fmha::Smem_tile_mma_transposed<Cta_tile_p>;

  using Gmem_tile_do = fmha::Gmem_tile_dout<Cta_tile_p>;

  // Make sure the number of threads match.
  static_assert((int)Gmem_tile_o::THREADS_PER_ROW ==
                    (int)Smem_tile_o::THREADS_PER_ROW,
                "");

  // The number of threads.
  enum { THREADS = Cta_tile_p::THREADS_PER_CTA };
  // Make sure the number of threads matches both CTAs.
  static_assert((int)THREADS == (int)Cta_tile_o::THREADS_PER_CTA, "");

  // The amount of shared memory needed to load Q and K.
  enum {
    BYTES_PER_SMEM_QK =
        Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE
  };
  // The extra amount of shared memory needed to load V.
  enum {
    BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE
  };
  // The amount of shared memory needed for Q, K and V..
  enum { BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V };
  // The amount of shared memory needed to load Q and store O.
  enum {
    BYTES_PER_SMEM_QO =
        Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE
  };

  // The amount of shared memory needed for Q, K, V and O.
  enum {
    BYTES_PER_SMEM = fmha::Max<BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO>::VALUE
  };
  // Make sure we have enough shared memory.
  static_assert(Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE <=
                    BYTES_PER_SMEM,
                "");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

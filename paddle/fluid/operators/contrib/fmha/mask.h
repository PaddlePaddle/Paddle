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

namespace fmha {

template <typename Cta_tile>
struct Mask {
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;

  template <typename Params, typename BInfo>
  __device__ Mask(const Params &params, const BInfo &blockInfo, int tidx) {
    actual_seqlen = blockInfo.actual_seqlen;

    const int warp = tidx / Cta_tile::THREADS_PER_WARP;  // warp id
    const int lane = tidx % Cta_tile::THREADS_PER_WARP;  // lane id

    static_assert(Cta_tile::WARPS_K == 1, "");

    // find the warp in the Cta tile
    const int warp_n = (warp / Cta_tile::WARPS_M);  // warp n
    const int warp_m = (warp % Cta_tile::WARPS_M);  // warp m
    // decompose warp into 8x4 tile
    const int quad = lane / 4;
    const int tid = (lane % 4) * 2;
    row = warp_m * 16 + quad;  // each for 4 elements?
    col = warp_n * 16 + tid;
  }

  inline __device__ bool is_valid(const int mi, const int ni, const int ii,
                                  const int jj) const {
    // ii and jj iterate over the 2x4 fragment
    const bool col_valid = (ni * Mma_tile::N_PER_MMA_PER_CTA + col +
                            (jj & 2) * 4 + (jj & 1)) < actual_seqlen;
    //&& (row + mi * Mma_tile::M_PER_MMA_PER_CTA + ii * 8) < actual_seqlen;
    return col_valid;
    // return row_valid && col_valid;
  }

  inline __device__ void load(int it) { row_offset = it * Cta_tile::M + row; }
  int row_offset;

  int row;
  int col;
  int actual_seqlen;
};

}  // namespace fmha

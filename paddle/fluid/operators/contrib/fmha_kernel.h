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

#include "paddle/fluid/operators/contrib/fmha.h"
#include "paddle/fluid/operators/contrib/fmha/gmem_tile.h"
#include "paddle/fluid/operators/contrib/fmha/mask.h"
#include "paddle/fluid/operators/contrib/fmha/smem_tile.h"
#include "paddle/fluid/operators/contrib/fmha/softmax.h"
#include "paddle/fluid/operators/contrib/fmha/utils.h"

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int FMHA_VERSION>
struct BlockInfo {};

template <>
struct BlockInfo<1> {
  int actual_seqlen;
  int bidx;
  int sum_s;
  int bidh;
  int bidb;

  template <typename Params>
  __device__ BlockInfo(const Params &params, const int bidb, const int bidh,
                       const int tidx)
      : bidb(bidb), bidh(bidh) {
    // The block index.
    sum_s = params.b * params.s;
    actual_seqlen = params.s;
    bidx = bidb * params.h + bidh;
  }

  __device__ bool stop_early() const { return false; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct BlockInfo<2> {
  int actual_seqlen;
  int bidx;
  int sum_s;
  int bidh;
  int bidb;

  template <typename Params>
  __device__ BlockInfo(const Params &params, const int bidb, const int bidh,
                       const int tidx)
      : bidb(bidb), bidh(bidh) {
    // The block index.
    sum_s = params.cu_seqlens[bidb];
    actual_seqlen = params.cu_seqlens[bidb + 1] - sum_s;
    bidx = sum_s * params.h + bidh;
  }

  __device__ bool stop_early() const { return actual_seqlen == 0; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_CTA>
struct BlockInfoPadded {
  template <typename Params>
  __device__ BlockInfoPadded(const Params &params, const int bidb,
                             const int bidh, const int tidx)
      : bidb(bidb), bidh(bidh), h(params.h) {
    // The block index.
    sum_s = params.cu_seqlens[bidb];       // batch id
    actual_seqlen = params.seqlens[bidb];  // batch id
    bidx = sum_s * params.h + bidh;        // ?

    // THREADS_PER_CTA = WARPS_PER_CTA * 32; WARPS_PER_CTA = WARPS_M * WARPS_N *
    // WARPS_K
    tidx_global = (bidb * params.h + bidh) * THREADS_PER_CTA + tidx;
  }

  __device__ bool stop_early() const { return actual_seqlen == 0; }

  int actual_seqlen;
  int bidx;
  int sum_s;
  int bidh;
  int bidb;
  int tidx_global;
  int h;
};

}  // namespace fmha

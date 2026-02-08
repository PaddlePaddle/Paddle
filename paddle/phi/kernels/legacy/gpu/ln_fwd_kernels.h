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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#pragma once

#include "ln.h"        // NOLINT
#include "ln_utils.h"  // NOLINT

namespace layer_norm {

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_fwd_kernel(
    FwdParams params) {
  enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
  enum { WARPS_N = Ktraits::WARPS_N };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
  enum { VEC_COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG };
  enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
  enum { LDGS = Ktraits::LDGS };
  enum { NUM_ELTS = Ktraits::NUM_ELTS };
  enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

  using output_t = typename Ktraits::output_t;
  using index_t = typename Ktraits::index_t;
  using compute_t = typename Ktraits::compute_t;
  using Ivec = typename Ktraits::Ivec;
  using Ovec = typename Ktraits::Ovec;
  using Wvec = typename Ktraits::Wvec;
  using Cvec = typename Ktraits::Cvec;

  using Stats = typename Ktraits::Stats;
  using stats_t = typename Stats::stats_t;

  extern __shared__ char smem_[];

  const index_t tidx = threadIdx.x;
  const index_t bidn = blockIdx.x % CTAS_PER_ROW;
  const index_t bidm = blockIdx.x / CTAS_PER_ROW;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / WARPS_N;
  const index_t warp_n = warp % WARPS_N;

  const index_t r = bidm * ROWS_PER_CTA + warp_m;
  const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

  Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem_);

  compute_t *mu_ptr = static_cast<compute_t *>(params.mean);
  compute_t *rs_ptr = static_cast<compute_t *>(params.invvar);

  Wvec gamma[LDGS];
  Wvec beta[LDGS];
  index_t idx = c;
  if (params.bias) {
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      gamma[it].load_from(params.scale, idx);
      beta[it].load_from(params.bias, idx);
      idx += VEC_COLS_PER_LDG;
    }
  } else {
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      gamma[it].load_from(params.scale, idx);
      beta[it].init(0.);
      idx += VEC_COLS_PER_LDG;
    }
  }

  constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);
  bool is_rmsnorm = mu_ptr == nullptr;

  for (int row = r; row < params.rows;
       row += params.ctas_per_col * ROWS_PER_CTA) {
    Ivec x[LDGS];
    index_t idx = row * Ktraits::VEC_COLS + c;
    compute_t xf[LDGS * NUM_ELTS];
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      x[it].load_from(params.x, idx);
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t x_ij = compute_t(x[it].data.elt[jt]);
        xf[it * NUM_ELTS + jt] = x_ij;
      }
      idx += VEC_COLS_PER_LDG;
    }

    stats_t s = stats.compute(xf, rn, is_rmsnorm);

    compute_t mu = layer_norm::Get<0>::of<stats_t, compute_t>(s);
    compute_t m2 = layer_norm::Get<1>::of<stats_t, compute_t>(s);

    if (mu_ptr && bidn == 0 && warp_n == 0 && lane == 0) {
      mu_ptr[row] = mu;
    }

    compute_t rs = rsqrtf(rn * m2 + params.epsilon);

    if (bidn == 0 && warp_n == 0 && lane == 0) {
      rs_ptr[row] = rs;
    }

    Ovec z[LDGS];
    idx = row * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t y_ij;
        if (is_rmsnorm) {
          y_ij = compute_t(rs * xf[it * NUM_ELTS + jt]);
        } else {
          y_ij = compute_t(rs * (xf[it * NUM_ELTS + jt] - mu));
        }
        compute_t g_ij = gamma[it].data.elt[jt];
        compute_t b_ij = beta[it].data.elt[jt];
        z[it].data.elt[jt] = static_cast<output_t>(g_ij * y_ij + b_ij);
      }
      z[it].store_to(params.y, idx);
      idx += VEC_COLS_PER_LDG;
    }
  }
}

}  // namespace layer_norm

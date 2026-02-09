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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#include "paddle/phi/kernels/funcs/fast_ln_v2_common.h"
#include "paddle/phi/kernels/funcs/fast_ln_v2_utils.h"
namespace phi {
namespace funcs {
namespace fast_ln_v2 {

BwdRegistry FAST_LN_V2_BWD_FUNCS;

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_bwd_kernel(
    fast_ln_v2::BwdParams params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { WARPS_N = Ktraits::WARPS_N };
  enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
  enum { COLS = Ktraits::COLS };
  enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
  enum { LDGS = Ktraits::LDGS };
  enum { NUM_ELTS = Ktraits::ELTS_PER_LDG };
  enum { THREADS_PER_WARP = Ktraits::THREADS_PER_WARP };
  enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

  using compute_t = typename Ktraits::compute_t;
  using index_t = typename Ktraits::index_t;
  using Ivec = typename Ktraits::Ivec;
  using Ovec = typename Ktraits::Ovec;
  using Wvec = typename Ktraits::Wvec;
  using Cvec = typename Ktraits::Cvec;
  using Reducer = typename Ktraits::Reducer;
  using reduce_t = typename Reducer::Type;

  extern __shared__ char smem_[];

  const index_t tidx = threadIdx.x;
  const index_t bidn = blockIdx.x % CTAS_PER_ROW;
  const index_t bidm = blockIdx.x / CTAS_PER_ROW;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / Ktraits::WARPS_N;
  const index_t warp_n = warp % Ktraits::WARPS_N;
  const index_t tid_r = warp_n * THREADS_PER_WARP + lane;

  const index_t r = bidm * Ktraits::ROWS_PER_CTA + warp_m;
  const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

  static_assert(COLS == THREADS_PER_ROW * LDGS * NUM_ELTS * CTAS_PER_ROW);

  Cvec dzy_sum[LDGS];
  Cvec dz_sum[LDGS];

  memset(dzy_sum, 0, sizeof(dzy_sum));
  memset(dz_sum, 0, sizeof(dz_sum));

  compute_t *smem_wgrad = reinterpret_cast<compute_t *>(smem_);
  char *smem_dgrad = smem_ + Ktraits::SMEM_BYTES_WGRAD;

  Reducer reducer(params, bidm, bidn, warp_m, warp_n, lane, smem_dgrad);

  Sum<reduce_t> sum;
  bool is_rmsnorm = (params.mean == nullptr);
  constexpr float rn = 1.f / static_cast<float>(COLS);
  Wvec gamma[LDGS];
  index_t idx = c;
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    if (params.scale != nullptr) {
      gamma[it].load_from(params.scale, idx);
    } else {
      gamma[it].init(1.0f);
    }
    idx += Ktraits::VEC_COLS_PER_LDG;
  }
#pragma unroll 1
  for (int row = r; row < params.rows;
       row += params.ctas_per_col * ROWS_PER_CTA) {
    const compute_t mu_r =
        is_rmsnorm ? static_cast<compute_t>(0.)
                   : static_cast<const compute_t *>(params.mean)[row];
    const compute_t rs_r = static_cast<const compute_t *>(params.invvar)[row];
    Ivec x[LDGS];
    Ovec dz[LDGS];
    index_t idx = row * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dz[it].load_from(params.dy, idx);
      x[it].load_from(params.x, idx);
      idx += Ktraits::VEC_COLS_PER_LDG;
    }

    compute_t dy[LDGS * NUM_ELTS];
    compute_t y[LDGS * NUM_ELTS];

    compute_t mdy_local = 0.f;
    compute_t mdyy_local = 0.f;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t x_tmp = x[it].data.elt[jt];
        compute_t y_tmp = rs_r * (x_tmp - mu_r);
        compute_t dy_tmp = compute_t(gamma[it].data.elt[jt]);
        dy_tmp *= compute_t(dz[it].data.elt[jt]);
        compute_t dz_tmp = dz[it].data.elt[jt];

        mdy_local += dy_tmp;
        mdyy_local += dy_tmp * y_tmp;

        dy[it * NUM_ELTS + jt] = dy_tmp;
        y[it * NUM_ELTS + jt] = y_tmp;

        dzy_sum[it].data.elt[jt] += dz_tmp * y_tmp;
        dz_sum[it].data.elt[jt] += dz_tmp;
      }
    }

    reduce_t result = reducer.allreduce({mdy_local, mdyy_local}, sum);
    if (is_rmsnorm) {
      mdy_local = 0.f;
    } else {
      mdy_local = fast_ln_v2::Get<0>::of<reduce_t, compute_t>(result) * rn;
    }
    mdyy_local = fast_ln_v2::Get<1>::of<reduce_t, compute_t>(result) * rn;
    Ivec dx[LDGS];
    idx = row * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t dy_tmp = dy[it * NUM_ELTS + jt];
        compute_t y_tmp = y[it * NUM_ELTS + jt];
        compute_t dx_tmp = rs_r * (dy_tmp - (mdyy_local * y_tmp + mdy_local));
        dx[it].data.elt[jt] = dx_tmp;
      }
      if (params.dx != nullptr) {
        dx[it].store_to(params.dx, idx);
      }
      idx += Ktraits::VEC_COLS_PER_LDG;
    }
  }  // end: grid stride loop

  if (WARPS_M == 1) {
    idx = r * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      if (params.dbias != nullptr) {
        dz_sum[it].store_to(params.dbias_part, idx);
      }
      dzy_sum[it].store_to(params.dscale_part, idx);
      idx += Ktraits::VEC_COLS_PER_LDG;
    }
  } else {
    static_assert(WARPS_M == 1 || Ktraits::CTAS_PER_ROW == 1,
                  "Multiple rows per CTA not supported for Multi-CTA.");
    // Finalize reduction of part dgamma and dbeta for this CTA
    // by reducing over the rows held across the WARPS_M warps

    // Assumption: blockSize divides hidden size.
    enum { NUM_RES = COLS / Ktraits::THREADS_PER_CTA };
    static_assert(NUM_RES * Ktraits::THREADS_PER_CTA == COLS, "");

    idx = warp_m * Ktraits::VEC_COLS + tid_r;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dz_sum[it].store_to(smem_wgrad, idx);
      idx += THREADS_PER_ROW;
    }
    __syncthreads();
    compute_t cta_dz_sum[NUM_RES];
    memset(cta_dz_sum, 0, sizeof(compute_t) * NUM_RES);
    for (int it = 0; it < ROWS_PER_CTA; it++) {
      for (int jt = 0; jt < NUM_RES; jt++) {
        cta_dz_sum[jt] +=
            smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
      }
    }
    __syncthreads();

    idx = warp_m * Ktraits::VEC_COLS + tid_r;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dzy_sum[it].store_to(smem_wgrad, idx);
      idx += THREADS_PER_ROW;
    }
    __syncthreads();
    compute_t cta_dzy_sum[NUM_RES];
    memset(cta_dzy_sum, 0, sizeof(compute_t) * NUM_RES);
    for (int it = 0; it < ROWS_PER_CTA; it++) {
      for (int jt = 0; jt < NUM_RES; jt++) {
        cta_dzy_sum[jt] +=
            smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
      }
    }

    compute_t *dgamma_part =
        (params.dscale_part != nullptr)
            ? static_cast<compute_t *>(params.dscale_part) + bidm * COLS + tidx
            : nullptr;
    for (int jt = 0; jt < NUM_RES; jt++) {
      if (dgamma_part != nullptr) {
        *dgamma_part = cta_dzy_sum[jt];
        dgamma_part += Ktraits::THREADS_PER_CTA;
      }
    }

    if (params.dbias != nullptr) {
      compute_t *dbeta_part =
          static_cast<compute_t *>(params.dbias_part) + bidm * COLS + tidx;
      for (int jt = 0; jt < NUM_RES; jt++) {
        *dbeta_part = cta_dz_sum[jt];
        dbeta_part += Ktraits::THREADS_PER_CTA;
      }
    }
  }
#endif
}

bool has_fast_ln_v2_bwd_kernel(phi::DataType weight_type,
                               phi::DataType input_type,
                               phi::DataType output_type,
                               phi::DataType compute_type,
                               uint32_t hidden_size) {
  auto iter = FAST_LN_V2_BWD_FUNCS.find(
      get_key(weight_type, input_type, output_type, compute_type, hidden_size));
  return iter != FAST_LN_V2_BWD_FUNCS.end();
}

template <typename Kernel_traits>
__global__
__launch_bounds__(Kernel_traits::THREADS_PER_CTA) void ln_bwd_finalize_kernel(
    BwdParams params) {
  using compute_t = typename Kernel_traits::compute_t;
  using weight_t = typename Kernel_traits::weight_t;
  using index_t = typename Kernel_traits::index_t;
  using Reducer = typename Kernel_traits::Reducer;
  using reduce_t = typename Reducer::Type;

  Sum<reduce_t> sum;
  enum { NUM_ELT = Kernel_traits::ELTS_PER_LDG };
  enum { THREADS_PER_WARP = Kernel_traits::THREADS_PER_WARP };

  __shared__ char smem_[Kernel_traits::SMEM_BYTES_PER_CTA];

  constexpr uint32_t bidm = 0;

  const uint32_t bidn = blockIdx.x;
  const uint32_t tidx = threadIdx.x;
  const uint32_t warp = tidx / THREADS_PER_WARP;
  const uint32_t lane = tidx % THREADS_PER_WARP;

  Reducer reducer(params, bidm, bidn, 0, 0, lane, smem_);

  const uint32_t c = bidn * THREADS_PER_WARP + lane;
  const uint32_t c_out = bidn * THREADS_PER_WARP / 2 + lane;
  constexpr uint32_t COL_STRIDE = Kernel_traits::CTAS * THREADS_PER_WARP;
  for (uint32_t col = c, col_out = c_out; col < Kernel_traits::COLS;
       col += COL_STRIDE, col_out += COL_STRIDE / 2) {
    // Each thread sums over NUM_ELT columns.
    Vec<compute_t, NUM_ELT> dbeta_local, dgamma_local;
    memset(&dgamma_local, 0, sizeof(dgamma_local));
    memset(&dbeta_local, 0, sizeof(dbeta_local));
    for (uint32_t row = warp; row < params.ctas_per_col;
         row += Kernel_traits::ROWS_PER_CTA) {
      index_t idx = row * Kernel_traits::COLS + col;

      Vec<compute_t, NUM_ELT> dbeta_part, dgamma_part;
      if (params.dbias != nullptr) {
        dbeta_part.load_from(params.dbias_part, idx);
      } else {
        dbeta_part.init(0.);
      }
      if (params.dscale_part != nullptr) {
        dgamma_part.load_from(params.dscale_part, idx);
      } else {
        dgamma_part.init(0.);
      }
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        dgamma_local.data.elt[it] += (dgamma_part.data.elt[it]);
        if (params.dbias != nullptr) {
          dbeta_local.data.elt[it] += (dbeta_part.data.elt[it]);
        }
      }
    }

    void *smem_gamma = smem_;
    void *smem_beta = &smem_[Kernel_traits::SMEM_BYTES_TRANSPOSE];

    const int write_row = warp;
    const int write_col = lane ^ write_row;
    const int write_idx = write_row * THREADS_PER_WARP + write_col;

    dgamma_local.store_to(smem_gamma, write_idx);
    dbeta_local.store_to(smem_beta, write_idx);

    __syncthreads();

    // It would be probably safe to reuse the first row of smem_beta and
    // smem_gamma
    void *smem_gamma_out = &smem_[2 * Kernel_traits::SMEM_BYTES_TRANSPOSE];
    void *smem_beta_out = &smem_[2 * Kernel_traits::SMEM_BYTES_TRANSPOSE +
                                 Kernel_traits::SMEM_BYTES_OUTPUT];

    // More than one iter iff ROWS_PER_CTA < 32.
    for (int w = warp; w < THREADS_PER_WARP; w += Kernel_traits::ROWS_PER_CTA) {
      const int read_row = lane;
      const int read_col = w ^ read_row;
      const int read_idx = read_row * THREADS_PER_WARP + read_col;

      memset(&dbeta_local, 0, sizeof(dbeta_local));
      memset(&dgamma_local, 0, sizeof(dgamma_local));

      // Load beta and gamma transposed
      if (read_row < Kernel_traits::ROWS_PER_CTA) {
        dbeta_local.load_from(smem_beta, read_idx);
        dgamma_local.load_from(smem_gamma, read_idx);
      }

// Call reducer on the loaded value(s) and convert.
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        compute_t b_i = dbeta_local.data.elt[it];
        compute_t g_i = dgamma_local.data.elt[it];
        b_i = reducer.allreduce(b_i, sum);
        g_i = reducer.allreduce(g_i, sum);

        dgamma_local.data.elt[it] = g_i;
        dbeta_local.data.elt[it] = b_i;
      }

      // Leader stores the result at the current column.
      if (lane == 0) {
        dgamma_local.store_to(smem_gamma_out, w);
        dbeta_local.store_to(smem_beta_out, w);
      }
    }

    // All writes done.
    __syncthreads();

    // Pack and store: 2-wide stores with half the threads.
    if (warp == Kernel_traits::ROWS_PER_CTA - 1 &&
        lane < THREADS_PER_WARP / 2) {
      using src_t = typename TypeToVec2<compute_t>::Type;
      using dst_t = typename TypeToVec2<weight_t>::Type;
      Vec<src_t, NUM_ELT> dbeta_vec2, dgamma_vec2;
      Vec<dst_t, NUM_ELT> dbeta_out2, dgamma_out2;

      dgamma_vec2.load_from(smem_gamma_out, lane);
      dbeta_vec2.load_from(smem_beta_out, lane);
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        dgamma_out2.data.elt[it] =
            Converter<src_t, dst_t>::convert(dgamma_vec2.data.elt[it]);
        dbeta_out2.data.elt[it] =
            Converter<src_t, dst_t>::convert(dbeta_vec2.data.elt[it]);
      }
      if (params.dscale != nullptr) {
        dgamma_out2.store_to(params.dscale, col_out);
      }
      if (params.dbias != nullptr) {
        dbeta_out2.store_to(params.dbias, col_out);
      }
    }
  }
}

template <typename weight_t,
          typename input_t,
          typename output_t,
          typename compute_t,
          typename index_t,
          int HIDDEN_SIZE,
          int CTAS_PER_ROW,
          int WARPS_M,
          int WARPS_N,
          int BYTES_PER_LDG_MAIN,
          int BYTES_PER_LDG_FINAL>
void launch_(LaunchParams<BwdParams> &launch_params,  // NOLINT
             const bool configure_params) {
  using KernelTraits = KernelTraits<weight_t,
                                    input_t,
                                    output_t,
                                    compute_t,
                                    index_t,
                                    HIDDEN_SIZE,
                                    CTAS_PER_ROW,
                                    WARPS_M,
                                    WARPS_N,
                                    BYTES_PER_LDG_MAIN>;
  auto kernel = &ln_bwd_kernel<KernelTraits>;

  if (configure_params) {
    int ctas_per_sm;
    cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm,
        kernel,
        KernelTraits::THREADS_PER_CTA,
        KernelTraits::SMEM_BYTES);
    launch_params.params.ctas_per_col =
        launch_params.props->multiProcessorCount * ctas_per_sm /
        KernelTraits::CTAS_PER_ROW;
    launch_params.barrier_size = 0;
    launch_params.workspace_bytes = 0;
    if (KernelTraits::CTAS_PER_ROW > 1) {
      launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
      launch_params.workspace_bytes =
          launch_params.params.ctas_per_col * KernelTraits::WARPS_M *
          KernelTraits::CTAS_PER_ROW * sizeof(typename KernelTraits::reduce_t) *
          2;
    }
    return;
  }

  if (KernelTraits::SMEM_BYTES >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    KernelTraits::SMEM_BYTES));
  }
  auto stream = launch_params.stream;
  auto ctas_per_col = launch_params.params.ctas_per_col;

  if (KernelTraits::CTAS_PER_ROW == 1) {
    kernel<<<ctas_per_col,
             KernelTraits::THREADS_PER_CTA,
             KernelTraits::SMEM_BYTES,
             stream>>>(launch_params.params);
  } else {
    dim3 grid(KernelTraits::CTAS_PER_ROW * ctas_per_col);
    dim3 block(KernelTraits::THREADS_PER_CTA);
    void *params_ = (void *)&launch_params.params;  // NOLINT
    cudaLaunchCooperativeKernel((void *)kernel,     // NOLINT
                                grid,
                                block,
                                (void **)&params_,  // NOLINT
                                KernelTraits::SMEM_BYTES,
                                stream);
  }

  using KernelTraitsF =
      fast_ln_v2::KernelTraitsFinalize<HIDDEN_SIZE,
                                       weight_t,
                                       input_t,
                                       output_t,
                                       compute_t,
                                       index_t,
                                       32 * 32,  // THREADS_PER_CTA
                                       BYTES_PER_LDG_FINAL>;

  auto kernel_f = &fast_ln_v2::ln_bwd_finalize_kernel<KernelTraitsF>;
  kernel_f<<<KernelTraitsF::CTAS, KernelTraitsF::THREADS_PER_CTA, 0, stream>>>(
      launch_params.params);
}

// Create backward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N,
//  BYTES_PER_LDG, BYTES_PER_LDG_FINAL

#define REGISTER_BWD_LAUNCHER(HIDDEN_SIZE,                                   \
                              WTYPE,                                         \
                              ITYPE,                                         \
                              OTYPE,                                         \
                              CTYPE,                                         \
                              CTAS_PER_ROW,                                  \
                              WARPS_M,                                       \
                              WARPS_N,                                       \
                              BYTES_PER_LDG,                                 \
                              BYTES_PER_LDG_FINALIZE)                        \
  void ln_bwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(         \
      LaunchParams<BwdParams> &launch_params, const bool configure_params) { \
    launch_<WTYPE,                                                           \
            ITYPE,                                                           \
            OTYPE,                                                           \
            CTYPE,                                                           \
            uint32_t,                                                        \
            HIDDEN_SIZE,                                                     \
            CTAS_PER_ROW,                                                    \
            WARPS_M,                                                         \
            WARPS_N,                                                         \
            BYTES_PER_LDG,                                                   \
            BYTES_PER_LDG_FINALIZE>(launch_params, configure_params);        \
  }                                                                          \
  static BwdRegistrar<WTYPE, ITYPE, OTYPE, CTYPE, HIDDEN_SIZE>               \
      reg_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(             \
          ln_bwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE)

#if CUDNN_VERSION_MIN(8, 1, 0) && CUDA_VERSION >= 12000
REGISTER_BWD_LAUNCHER(1536, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(1536, fp16, fp16, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(1536, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(1536, bf16, bf16, bf16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(1536, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(2048, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(2048, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(2304, fp32, fp32, fp32, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(2304, fp16, fp16, fp16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(2304, fp16, fp32, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(2304, bf16, bf16, bf16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(2304, bf16, fp32, bf16, fp32, 1, 1, 4, 8, 4);

REGISTER_BWD_LAUNCHER(3072, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(3072, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(3840, fp32, fp32, fp32, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(3840, fp16, fp16, fp16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(3840, fp16, fp32, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_BWD_LAUNCHER(3840, bf16, bf16, bf16, fp32, 1, 1, 4, 4, 4);
REGISTER_BWD_LAUNCHER(3840, bf16, fp32, bf16, fp32, 1, 1, 4, 8, 4);

REGISTER_BWD_LAUNCHER(4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(4096, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(5120, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(5120, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(6144, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, fp16, fp16, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, fp16, fp32, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, bf16, bf16, bf16, fp32, 1, 1, 8, 16, 4);
REGISTER_BWD_LAUNCHER(6144, bf16, fp32, bf16, fp32, 1, 1, 8, 16, 4);

REGISTER_BWD_LAUNCHER(8192, fp32, fp32, fp32, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, fp16, fp16, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, fp16, fp32, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, bf16, bf16, bf16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(8192, bf16, fp32, bf16, fp32, 2, 1, 4, 16, 4);

REGISTER_BWD_LAUNCHER(10240, fp32, fp32, fp32, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, fp16, fp16, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, fp16, fp32, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, bf16, bf16, bf16, fp32, 2, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER(10240, bf16, fp32, bf16, fp32, 2, 1, 4, 16, 4);
#endif  // CUDNN_VERSION_MIN(8, 1, 0) && CUDA_VERSION >= 12000

BwdFunction &get_bwd_launcher(phi::DataType weight_type,
                              phi::DataType input_type,
                              phi::DataType output_type,
                              phi::DataType compute_type,
                              uint32_t hidden_size) {
  auto iter = FAST_LN_V2_BWD_FUNCS.find(
      get_key(weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != FAST_LN_V2_BWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "BWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}

}  // namespace  fast_ln_v2
}  // namespace funcs
}  // namespace phi

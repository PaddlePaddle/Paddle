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

FwdRegistry FAST_LN_V2_FWD_FUNCS;

FwdFunction &get_fwd_launcher(DataType weight_type,
                              DataType input_type,
                              DataType output_type,
                              DataType compute_type,
                              uint32_t hidden_size) {
  auto iter = FAST_LN_V2_FWD_FUNCS.find(
      get_key(weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != FAST_LN_V2_FWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "FWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}
bool has_fast_ln_v2_fwd_kernel(DataType weight_type,
                               DataType input_type,
                               DataType output_type,
                               DataType compute_type,
                               uint32_t hidden_size) {
  auto iter = FAST_LN_V2_FWD_FUNCS.find(
      get_key(weight_type, input_type, output_type, compute_type, hidden_size));
  return iter != FAST_LN_V2_FWD_FUNCS.end();
}

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_fwd_kernel(
    FwdParams params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
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

    compute_t mu = fast_ln_v2::Get<0>::of<stats_t, compute_t>(s);
    compute_t m2 = fast_ln_v2::Get<1>::of<stats_t, compute_t>(s);

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
#endif
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
          int BYTES_PER_LDG>
void launch_(LaunchParams<FwdParams> &launch_params,  // NOLINT
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
                                    BYTES_PER_LDG>;
  auto kernel = &ln_fwd_kernel<KernelTraits>;

  if (configure_params) {
    int ctas_per_sm;
    cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm,
        kernel,
        KernelTraits::THREADS_PER_CTA,
        KernelTraits::SMEM_BYTES_FWD);
    launch_params.params.ctas_per_col =
        launch_params.props->multiProcessorCount * ctas_per_sm /
        KernelTraits::CTAS_PER_ROW;
    launch_params.barrier_size = 0;
    launch_params.workspace_bytes = 0;
    if (KernelTraits::CTAS_PER_ROW > 1) {
      launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
      launch_params.workspace_bytes =
          launch_params.params.ctas_per_col * KernelTraits::WARPS_M *
          KernelTraits::CTAS_PER_ROW *
          sizeof(typename KernelTraits::Stats::stats_t) * 2;
    }
    return;
  }

  if (KernelTraits::SMEM_BYTES_FWD >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    KernelTraits::SMEM_BYTES_FWD));
  }
  auto stream = launch_params.stream;
  auto ctas_per_col = launch_params.params.ctas_per_col;

  if (KernelTraits::CTAS_PER_ROW == 1) {
    kernel<<<ctas_per_col,
             KernelTraits::THREADS_PER_CTA,
             KernelTraits::SMEM_BYTES_FWD,
             stream>>>(launch_params.params);
  } else {
    dim3 grid(KernelTraits::CTAS_PER_ROW * ctas_per_col);
    dim3 block(KernelTraits::THREADS_PER_CTA);
    void *params_ = (void *)&launch_params.params;  // NOLINT
    cudaLaunchCooperativeKernel((void *)kernel,     // NOLINT
                                grid,
                                block,
                                (void **)&params_,  // NOLINT
                                KernelTraits::SMEM_BYTES_FWD,
                                stream);
  }
}

// Create forward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N,
//  BYTES_PER_LDG

#define REGISTER_FWD_LAUNCHER(HIDDEN_SIZE,                                   \
                              WTYPE,                                         \
                              ITYPE,                                         \
                              OTYPE,                                         \
                              CTYPE,                                         \
                              CTAS_PER_ROW,                                  \
                              WARPS_M,                                       \
                              WARPS_N,                                       \
                              BYTES_PER_LDG)                                 \
  void ln_fwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(         \
      LaunchParams<FwdParams> &launch_params, const bool configure_params) { \
    launch_<WTYPE,                                                           \
            ITYPE,                                                           \
            OTYPE,                                                           \
            CTYPE,                                                           \
            uint32_t,                                                        \
            HIDDEN_SIZE,                                                     \
            CTAS_PER_ROW,                                                    \
            WARPS_M,                                                         \
            WARPS_N,                                                         \
            BYTES_PER_LDG>(launch_params, configure_params);                 \
  }                                                                          \
  static FwdRegistrar<WTYPE, ITYPE, OTYPE, CTYPE, HIDDEN_SIZE>               \
      reg_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(             \
          ln_fwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE)

#if CUDNN_VERSION_MIN(8, 1, 0) && CUDA_VERSION >= 12000
REGISTER_FWD_LAUNCHER(1536, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(1536, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(1536, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(1536, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(1536, bf16, fp32, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER(2048, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2048, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2048, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2048, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2048, bf16, fp32, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER(2304, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2304, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2304, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2304, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(2304, bf16, fp32, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER(3072, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(3072, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(3072, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(3072, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(3072, bf16, fp32, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER(3840, fp32, fp32, fp32, fp32, 1, 1, 4, 4);
REGISTER_FWD_LAUNCHER(3840, fp16, fp16, fp16, fp32, 1, 1, 4, 4);
REGISTER_FWD_LAUNCHER(3840, fp16, fp32, fp16, fp32, 1, 1, 4, 4);
REGISTER_FWD_LAUNCHER(3840, bf16, bf16, bf16, fp32, 1, 1, 4, 4);
REGISTER_FWD_LAUNCHER(3840, bf16, fp32, bf16, fp32, 1, 1, 4, 4);

REGISTER_FWD_LAUNCHER(4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(4096, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(4096, bf16, fp32, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER(5120, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(5120, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(5120, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(5120, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(5120, bf16, fp32, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER(6144, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(6144, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(6144, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(6144, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(6144, bf16, fp32, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER(8192, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(8192, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(8192, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(8192, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(8192, bf16, fp32, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER(10240, fp32, fp32, fp32, fp32, 2, 1, 4, 16);
REGISTER_FWD_LAUNCHER(10240, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(10240, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(10240, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER(10240, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
#endif  // CUDNN_VERSION_MIN(8, 1, 0) && CUDA_VERSION >= 12000
}  // namespace fast_ln_v2
}  // namespace funcs
}  // namespace phi

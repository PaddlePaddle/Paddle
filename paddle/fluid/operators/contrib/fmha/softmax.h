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

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Sum_ {
  enum { IS_SUM = 1 };
  static inline __device__ float apply(float x, float y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Max_ {
  enum { IS_SUM = 0 };
  static inline __device__ float apply(float x, float y) {
    return x > y ? x : y;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float apply_exp_(float x, float max) {
  return __expf(x - max);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax_base {
  // The Mma tile.
  using Mma_tile = fmha::Hmma_tile<Cta_tile>;

  // The number of MMAs in M/N dimensions.
  enum { MMAS_M = Mma_tile::MMAS_M };
  enum { MMAS_N = Mma_tile::MMAS_N };

  // The number of groups of warp such that we have at most 4 warps writing
  // consecutive elements.
  enum { GROUPS = fmha::Div_up<Cta_tile::WARPS_N, 4>::VALUE };
  // The number of elements that we are going to store per row.
  enum { ELEMENTS_PER_ROW = Cta_tile::WARPS_N / GROUPS };
  // The number of rows.
  enum { ROWS = Cta_tile::M * GROUPS };
  // The total number of elements.
  enum { ELEMENTS = ROWS * ELEMENTS_PER_ROW };

  // Ctor.
  template <typename Params>
  inline __device__ Softmax_base(const Params &params, void *smem, int bidb,
                                 int tidx)
      :  // packed_mask_ptr_(reinterpret_cast<const
         // char*>(params.packed_mask_ptr)),
        smem_(reinterpret_cast<float *>(smem)),
        tidx_(tidx) {
    // Move to the 1st mask loaded by the thread+ tidx;
    // packed_mask_ptr_ += bidb * params.packed_mask_stride_in_bytes + tidx *
    // sizeof(uint32_t);

    // Extract the position in the warp.
    int warp = tidx / Cta_tile::THREADS_PER_WARP;
    int lane = tidx % Cta_tile::THREADS_PER_WARP;

    // Decompose the warp index into M and N.
    int warp_m = warp % Cta_tile::WARPS_M;
    int warp_n = warp / Cta_tile::WARPS_M;

    // Decompose the warp-n index into group/position-inside-the-group.
    int warp_g = warp_n / ELEMENTS_PER_ROW;
    int warp_i = warp_n % ELEMENTS_PER_ROW;

    // The location written by the threads.
    int write_row =
        warp_g * (ROWS / GROUPS) + warp_m * Mma_tile::M_PER_MMA + lane / 4;
    int write_col = warp_i;

    // Assemble the write pointer.
    smem_write_ = &smem_[write_row * ELEMENTS_PER_ROW + write_col];

    // Assemble the read pointer.
    smem_read_ = &smem_[warp_m * Mma_tile::M_PER_MMA + lane / 4];
  }

  template <typename Mask>
  inline __device__ void apply_mask(const Mask &mask) {
#pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
#pragma unroll
        for (int ni = 0; ni < MMAS_N; ++ni) {
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            if (!mask.is_valid(mi, ni, ii, jj)) {
              elt_[2 * mi + ii][4 * ni + jj] = -INFINITY;
            }
          }
        }
      }
    }
  }

  // Apply the exp to all the elements.
  inline __device__ void apply_exp(const float (&max)[MMAS_M * 2]) {
#pragma unroll
    for (int mi = 0; mi < MMAS_M * 2; ++mi) {
#pragma unroll
      for (int ni = 0; ni < MMAS_N * 4; ++ni) {
        elt_[mi][ni] = apply_exp_(elt_[mi][ni], max[mi]);
      }
    }
  }

  // Do a CTA-wide reduction.
  template <typename Functor>
  inline __device__ void reduce_1x4(float (&dst)[MMAS_M * 2]) {
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    if (Functor::IS_SUM) {
      // Apply the summation inside the thread.
      float tmp[MMAS_M * 2][2];
#pragma unroll
      for (int mi = 0; mi < MMAS_M * 2; ++mi) {
        tmp[mi][0] = 0.f;
        tmp[mi][1] = 0.f;
#pragma unroll
        for (int ni = 0; ni < MMAS_N; ++ni) {
          tmp[mi][0] += elt_[mi][4 * ni + 0];
          tmp[mi][0] += elt_[mi][4 * ni + 1];
          tmp[mi][1] += elt_[mi][4 * ni + 2];
          tmp[mi][1] += elt_[mi][4 * ni + 3];
        }
        dst[mi] = tmp[mi][0] + tmp[mi][1];
      }
    } else
#endif  // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    {
// Apply the functor for each row inside a thread.
#pragma unroll
      for (int mi = 0; mi < MMAS_M * 2; ++mi) {
        dst[mi] = elt_[mi][0];
#pragma unroll
        for (int ni = 1; ni < MMAS_N * 4; ++ni) {
          dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
        }
      }
    }

// Apply the functor for each row inside each group of 4 threads.
#pragma unroll
    for (int mi = 0; mi < MMAS_M * 2; ++mi) {
      dst[mi] =
          Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
      __syncwarp();
      dst[mi] =
          Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
      __syncwarp();
    }

// Store the different values.
#pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
      if (tidx_ % 4 == 0) {
        smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 0) * ELEMENTS_PER_ROW] =
            dst[2 * mi + 0];
        smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 8) * ELEMENTS_PER_ROW] =
            dst[2 * mi + 1];
      }
    }

    // Make sure the values are in shared memory.
    __syncthreads();

    // Load 8 values (one for each warp). The /8 corresponds to /(4*2) where 4
    // is from the
    // float4.
    float4 tmp[1];
    if (tidx_ < Cta_tile::M) {
      tmp[0] =
          reinterpret_cast<const float4 *>(&smem_[0 * ELEMENTS / 2])[tidx_];
    }

    // Compute the reduction of those 8 values in a binary-tree fashion.
    tmp[0].x = Functor::apply(tmp[0].x, tmp[0].y);
    tmp[0].z = Functor::apply(tmp[0].z, tmp[0].w);
    tmp[0].x = Functor::apply(tmp[0].x, tmp[0].z);

    // Make sure we can write to shared memory.
    __syncthreads();

    // Store the value back to shared memory.
    if (tidx_ < Cta_tile::M) {
      smem_[tidx_] = tmp[0].x;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

// Finally read the values.
#pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
      dst[2 * mi + 0] = smem_read_[mi * Mma_tile::M_PER_MMA_PER_CTA + 0];
      dst[2 * mi + 1] = smem_read_[mi * Mma_tile::M_PER_MMA_PER_CTA + 8];
    }
  }

  // Do a CTA-wide reduction.
  template <typename Functor>
  inline __device__ void reduce_1x8(float (&dst)[MMAS_M * 2]) {
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    if (Functor::IS_SUM) {
      // Apply the summation inside the thread.
      float tmp[MMAS_M * 2][2];
#pragma unroll
      for (int mi = 0; mi < MMAS_M * 2; ++mi) {
        tmp[mi][0] = 0.f;
        tmp[mi][1] = 0.f;
#pragma unroll
        for (int ni = 0; ni < MMAS_N; ++ni) {
          tmp[mi][0] += elt_[mi][4 * ni + 0];
          tmp[mi][0] += elt_[mi][4 * ni + 1];
          tmp[mi][1] += elt_[mi][4 * ni + 2];
          tmp[mi][1] += elt_[mi][4 * ni + 3];
        }
        dst[mi] = tmp[mi][0] + tmp[mi][1];
      }
    } else
#endif  // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    {
// Apply the functor for each row inside a thread.
#pragma unroll
      for (int mi = 0; mi < MMAS_M * 2; ++mi) {
        dst[mi] = elt_[mi][0];
#pragma unroll
        for (int ni = 1; ni < MMAS_N * 4; ++ni) {
          dst[mi] = Functor::apply(dst[mi], elt_[mi][ni]);
        }
      }
    }

// Apply the functor for each row inside each group of 4 threads.
#pragma unroll
    for (int mi = 0; mi < MMAS_M * 2; ++mi) {
      dst[mi] =
          Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 1));
      __syncwarp();
      dst[mi] =
          Functor::apply(dst[mi], __shfl_xor_sync(uint32_t(-1), dst[mi], 2));
      __syncwarp();
    }

// Store the different values.
#pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
      if (tidx_ % 4 == 0) {
        smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 0) * ELEMENTS_PER_ROW] =
            dst[2 * mi + 0];
        smem_write_[(mi * Mma_tile::M_PER_MMA_PER_CTA + 8) * ELEMENTS_PER_ROW] =
            dst[2 * mi + 1];
      }
    }

    // Make sure the values are in shared memory.
    __syncthreads();

    // Load 8 values (one for each warp). The /8 corresponds to /(4*2) where 4
    // is from the
    // float4.
    float4 tmp[2];
    if (tidx_ < Cta_tile::M) {
      tmp[0] =
          reinterpret_cast<const float4 *>(&smem_[0 * ELEMENTS / 2])[tidx_];
      tmp[1] =
          reinterpret_cast<const float4 *>(&smem_[1 * ELEMENTS / 2])[tidx_];
    }

    // Compute the reduction of those 8 values in a binary-tree fashion.
    tmp[0].x = Functor::apply(tmp[0].x, tmp[0].y);
    tmp[0].z = Functor::apply(tmp[0].z, tmp[0].w);
    tmp[1].x = Functor::apply(tmp[1].x, tmp[1].y);
    tmp[1].z = Functor::apply(tmp[1].z, tmp[1].w);
    tmp[0].x = Functor::apply(tmp[0].x, tmp[0].z);
    tmp[1].x = Functor::apply(tmp[1].x, tmp[1].z);
    tmp[0].x = Functor::apply(tmp[0].x, tmp[1].x);

    // Make sure we can write to shared memory.
    __syncthreads();

    // Store the value back to shared memory.
    if (tidx_ < Cta_tile::M) {
      smem_[tidx_] = tmp[0].x;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

// Finally read the values.
#pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
      dst[2 * mi + 0] = smem_read_[mi * Mma_tile::M_PER_MMA_PER_CTA + 0];
      dst[2 * mi + 1] = smem_read_[mi * Mma_tile::M_PER_MMA_PER_CTA + 8];
    }
  }

  // Do a CTA-wide reduction.
  template <typename Functor>
  inline __device__ void reduce(float (&dst)[MMAS_M * 2]) {
    static_assert(Cta_tile::WARPS_M == 1 &&
                  (Cta_tile::WARPS_N == 4 || Cta_tile::WARPS_N == 8));
    if (Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 4) {
      reduce_1x4<Functor>(dst);
    } else if (Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 8) {
      reduce_1x8<Functor>(dst);
    } else {
      assert(false);
    }

    // Make sure we are done reading from shared memory.
    __syncthreads();
  }

  // Scale all the elements.
  inline __device__ void scale(const float (&sum)[MMAS_M * 2]) {
    // Precompute the inverse sum to normalize. Without -use_fast_math, it makes
    // a huge deal.
    float inv_sum[MMAS_M * 2];
#pragma unroll
    for (int mi = 0; mi < MMAS_M * 2; ++mi) {
      inv_sum[mi] =
          (sum[mi] == 0.f || sum[mi] != sum[mi]) ? 1.f : 1.f / sum[mi];
    }

// Update the values.
#pragma unroll
    for (int mi = 0; mi < MMAS_M * 2; ++mi) {
#pragma unroll
      for (int ni = 0; ni < MMAS_N * 4; ++ni) {
        elt_[mi][ni] *= inv_sum[mi];
      }
    }
  }

  // The pointer to the mask.
  const char *packed_mask_ptr_;
  // Shared memory for the CTA-wide reduction.
  float *smem_, *smem_write_, *smem_read_;
  // The current thread index.
  int tidx_;
  // The elements.
  float elt_[MMAS_M * 2][MMAS_N * 4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Softmax : public Softmax_base<Cta_tile, Kernel_traits> {
  // The base class.
  using Base = Softmax_base<Cta_tile, Kernel_traits>;
  // The fragment.
  using Fragment_a = fmha::Fragment_a<fmha::Row>;

  static_assert(Fragment_a::NUM_REGS == 4);

  // The MMAs.
  enum { MMAS_M = Base::MMAS_M };
  enum { MMAS_N = Base::MMAS_N };

  // The accumulators.
  using Accumulator = fmha::Fragment_accumulator;
  using Accumulator_out = Fragment<uint16_t, 8>;
  static_assert(Accumulator_out::NUM_REGS == 4);

  static_assert(std::is_same<Accumulator::Data_type, float>::value);

  // Ctor.
  template <typename Params>
  inline __device__ Softmax(const Params &params, void *smem, int bidb,
                            int tidx)
      : Base(params, smem, bidb, tidx), params_scale_bmm1_(params.scale_bmm1) {}

  // Store the tile after softmax.
  template <typename Gmem_tile>
  inline __device__ void store(Gmem_tile &gmem_tile) {
    Accumulator_out acc[MMAS_M][MMAS_N];
#pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
#pragma unroll
      for (int ni = 0; ni < MMAS_N; ++ni) {
        // The elements.
        float tmp_00 = this->elt_[2 * mi + 0][4 * ni + 0];
        float tmp_01 = this->elt_[2 * mi + 0][4 * ni + 1];
        float tmp_02 = this->elt_[2 * mi + 0][4 * ni + 2];
        float tmp_03 = this->elt_[2 * mi + 0][4 * ni + 3];
        float tmp_10 = this->elt_[2 * mi + 1][4 * ni + 0];
        float tmp_11 = this->elt_[2 * mi + 1][4 * ni + 1];
        float tmp_12 = this->elt_[2 * mi + 1][4 * ni + 2];
        float tmp_13 = this->elt_[2 * mi + 1][4 * ni + 3];

        // Transform to accumulators.
        acc[mi][ni].reg(0) = fmha::float2_to_half2(tmp_00, tmp_01);
        acc[mi][ni].reg(1) = fmha::float2_to_half2(tmp_10, tmp_11);
        acc[mi][ni].reg(2) = fmha::float2_to_half2(tmp_02, tmp_03);
        acc[mi][ni].reg(3) = fmha::float2_to_half2(tmp_12, tmp_13);
      }
    }

    // Delegate to the gmem tile to store.
    gmem_tile.store(acc);
  }

  // Pack the data to a fragment for the next GEMM.
  template <int K, int M>
  inline __device__ void pack(Fragment_a (&dst)[K][M]) const {
#pragma unroll
    for (int mi = 0; mi < M; ++mi) {
#pragma unroll
      for (int ki = 0; ki < K; ++ki) {
        // 1st row - 4 elements per row.
        float tmp_00 = this->elt_[2 * mi + 0][4 * ki + 0];
        float tmp_01 = this->elt_[2 * mi + 0][4 * ki + 1];
        float tmp_02 = this->elt_[2 * mi + 0][4 * ki + 2];
        float tmp_03 = this->elt_[2 * mi + 0][4 * ki + 3];

        // 2nd row - 4 elements per row.
        float tmp_10 = this->elt_[2 * mi + 1][4 * ki + 0];
        float tmp_11 = this->elt_[2 * mi + 1][4 * ki + 1];
        float tmp_12 = this->elt_[2 * mi + 1][4 * ki + 2];
        float tmp_13 = this->elt_[2 * mi + 1][4 * ki + 3];

        // Pack to 4 registers.
        dst[ki][mi].reg(0) = fmha::float2_to_half2(tmp_00, tmp_01);
        dst[ki][mi].reg(1) = fmha::float2_to_half2(tmp_10, tmp_11);
        dst[ki][mi].reg(2) = fmha::float2_to_half2(tmp_02, tmp_03);
        dst[ki][mi].reg(3) = fmha::float2_to_half2(tmp_12, tmp_13);
      }
    }
  }

  // Scale FP32 fragments
  inline __device__ void unpack(const Accumulator (&acc)[MMAS_M][MMAS_N]) {
    const float scalef =
        reinterpret_cast<const float &>(this->params_scale_bmm1_);

#pragma unroll
    for (int mi = 0; mi < MMAS_M; ++mi) {
#pragma unroll
      for (int ni = 0; ni < MMAS_N; ++ni) {
        // 1st row - 4 elements per row.
        this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi][ni].elt(0) * scalef;
        this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi][ni].elt(1) * scalef;
        this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi][ni].elt(4) * scalef;
        this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi][ni].elt(5) * scalef;
        // 2nd row - 4 elements per row.
        this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi][ni].elt(2) * scalef;
        this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi][ni].elt(3) * scalef;
        this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi][ni].elt(6) * scalef;
        this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi][ni].elt(7) * scalef;
      }
    }
  }
  const uint32_t params_scale_bmm1_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha

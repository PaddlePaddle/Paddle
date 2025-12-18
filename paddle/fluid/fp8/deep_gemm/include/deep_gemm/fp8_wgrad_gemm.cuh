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

// The file has been adapted from DeepSeek DeepGEMM project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE

#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

template <uint32_t SHAPE_M,
          uint32_t SHAPE_N,
          uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumLastStages,
          uint32_t kNumTMAThreads,
          uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast,
          bool kIsTMAMulticastOnA>
__global__ void __launch_bounds__(
    get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
    fp8_wgrad_gemm_kernel(
        uint32_t shape_k,
        const __grid_constant__ CUtensorMap tensor_map_a,
        const __grid_constant__ CUtensorMap tensor_map_b,
        const __grid_constant__ CUtensorMap tensor_map_scales_a,
        const __grid_constant__ CUtensorMap tensor_map_scales_b,
        const __grid_constant__ CUtensorMap tensor_map_d) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
  // Scaling checks
  DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");

  // Types
  using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

  // Shared memory
  static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);
  static constexpr uint32_t SMEM_A_SIZE_PER_STAGE =
      BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
      BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE =
      BLOCK_M * sizeof(float);
  static constexpr uint32_t SMEM_SCALES_B_SIZE_PER_STAGE =
      BLOCK_N * sizeof(float);
  static constexpr uint32_t ALIGNED_SMEM_SCALES_B_SIZE_PER_STAGE =
      ceil_div(SMEM_SCALES_B_SIZE_PER_STAGE, 128U) * 128U;

  // Configs
  constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
  constexpr uint32_t kNumThreads =
      get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
  constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;

  const uint32_t shape_k_scales = ceil_div(shape_k, BLOCK_K);
  const uint32_t num_iterations = ceil_div(shape_k, kFullKOfAllStages);
  const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const uint32_t lane_idx = get_lane_id();

  // Prefetch TMA descriptors at the very beginning
  if (threadIdx.x == kNumMathThreads) {
    // NOTES: `reinterpret_cast` must be here, or NVRTC will fail
    cute::prefetch_tma_descriptor(
        reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_a));
    cute::prefetch_tma_descriptor(
        reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_b));
    cute::prefetch_tma_descriptor(
        reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_scales_a));
    cute::prefetch_tma_descriptor(
        reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_scales_b));
    cute::prefetch_tma_descriptor(
        reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_d));
  }
  __syncwarp();

  // Align to 1024 bytes for swizzle-128B
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0,
                   "Shared memory of A/B must be aligned to 1024 bytes");

  // Data on shared memory
  auto smem_d = reinterpret_cast<float*>(smem_buffer);
  __nv_fp8_e4m3* smem_a[kNumStages];
  __nv_fp8_e4m3* smem_b[kNumStages];
  float* smem_scales_a[kNumStages];
  float* smem_scales_b[kNumStages];

  // TMA Barrier for both divisible and non-divisible cases
  Barrier* full_barriers[kNumStages + 1];
  Barrier* empty_barriers[kNumStages + 1];

// Fill shared memory pointers
#pragma unroll
  for (int i = 0; i < kNumStages; ++i) {
    smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE +
                                                 i * SMEM_A_SIZE_PER_STAGE);
    smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
        i * SMEM_B_SIZE_PER_STAGE);
    smem_scales_a[i] = reinterpret_cast<float*>(
        smem_buffer + SMEM_D_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
        i * SMEM_SCALES_A_SIZE_PER_STAGE);
    smem_scales_b[i] = reinterpret_cast<float*>(
        smem_buffer + SMEM_D_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                      SMEM_SCALES_A_SIZE_PER_STAGE) +
        i * ALIGNED_SMEM_SCALES_B_SIZE_PER_STAGE);
  }

  // Fill barriers
  DG_STATIC_ASSERT(sizeof(Barrier) % sizeof(float) == 0, "Misaligned barriers");
  auto barrier_start_ptr = reinterpret_cast<Barrier*>(
      smem_buffer + SMEM_D_SIZE +
      kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                    SMEM_SCALES_A_SIZE_PER_STAGE +
                    ALIGNED_SMEM_SCALES_B_SIZE_PER_STAGE));
#pragma unroll
  for (int i = 0; i < kNumStages + 1; ++i) {
    full_barriers[i] = barrier_start_ptr + i;
    empty_barriers[i] = barrier_start_ptr + kNumStages + 1 + i;
  }

  // Initialize barriers
  DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "To many TMA multicast");
  if (threadIdx.x == kNumMathThreads) {
// NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the
// cluster, even with TMA multicast disabled, we want to make the behavior
// aligned
#pragma unroll
    for (int i = 0; i < kNumStages; ++i) {
      full_barriers[i]->init(1);
      empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
    }
    full_barriers[kNumStages]->init(1);
    empty_barriers[kNumStages]->init(1);

    // Make initialized barrier visible in async proxy
    cutlass::arch::fence_view_async_shared();
    (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
  }

  // Synchronize all threads to make barrier visible in normal memory model
  (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

  // For pipeline unrolling
  struct DivisibleK {};
  struct NotDivisibleK {};
  auto launch_k_iterations = [&](const auto& func) {
    if constexpr (kNumLastStages == 0) {
      for (int k_iter = 0; k_iter < num_iterations; ++k_iter)
        func(k_iter, DivisibleK{});
    } else {
      for (int k_iter = 0; k_iter < num_iterations - 1; ++k_iter)
        func(k_iter, DivisibleK{});
      func(num_iterations - 1, NotDivisibleK{});
    }
  };

  // Register reconfigurations
  constexpr int kNumTMARegisters = 40;
  constexpr int kNumMathRegisters = 232;

  // Block scheduler
  uint32_t m_block_idx, n_block_idx;
  auto scheduler = Scheduler<GemmType::Normal,
                             SHAPE_N,
                             BLOCK_M,
                             BLOCK_N,
                             1,
                             kNumTMAMulticast,
                             kIsTMAMulticastOnA>(SHAPE_M);

  if (threadIdx.x >= kNumMathThreads) {
    // TMA warp-group for loading data
    cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

    // NOTES: only one thread (or warp) will be used
    if (threadIdx.x == kNumMathThreads) {
      // Persistently schedule over blocks
      while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        launch_k_iterations([&](int k_iter, auto type) {
          constexpr bool kHasDivisibleStages =
              std::is_same_v<decltype(type), DivisibleK>;
          constexpr int kNumInnerStages =
              kHasDivisibleStages ? kNumStages : kNumLastStages;
          DG_STATIC_ASSERT(kNumInnerStages != 0,
                           "Invalid number of inner stages");

          // Assign TMA multicast number into A and B
          // NOTES: there may be additional odd rows/columns or cases where
          // multicast is not possible.
          const bool is_tma_multicast_valid =
              scheduler.is_tma_multicast_valid(m_block_idx);
          const uint32_t num_tma_multicast_a =
              (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast
                                                              : 1;
          const uint32_t num_tma_multicast_b =
              (not kIsTMAMulticastOnA and is_tma_multicast_valid)
                  ? kNumTMAMulticast
                  : 1;
          DG_STATIC_ASSERT(kNumTMAMulticast <= 2,
                           "Scheduler does not support > 2 TMA multicast");

#pragma unroll
          for (uint32_t s = 0; s < kNumInnerStages; ++s) {
            // Wait consumer release
            empty_barriers[s]->wait(
                (scheduler.current_iter * num_iterations + k_iter + 1) & 1);

            // Issue TMA A
            auto& full_barrier = *full_barriers[s];
            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
            tma_copy(&tensor_map_a,
                     reinterpret_cast<uint64_t*>(&full_barrier),
                     smem_a[s],
                     k_idx,
                     m_block_idx * BLOCK_M,
                     num_tma_multicast_a);
            tma_copy(&tensor_map_scales_a,
                     reinterpret_cast<uint64_t*>(&full_barrier),
                     smem_scales_a[s],
                     m_block_idx * BLOCK_M,
                     k_idx / BLOCK_K,
                     num_tma_multicast_a);

            // Issue TMA B
            tma_copy(&tensor_map_b,
                     reinterpret_cast<uint64_t*>(&full_barrier),
                     smem_b[s],
                     k_idx,
                     n_block_idx * BLOCK_N,
                     num_tma_multicast_b);
            tma_copy(&tensor_map_scales_b,
                     reinterpret_cast<uint64_t*>(&full_barrier),
                     smem_scales_b[s],
                     n_block_idx * BLOCK_N,
                     k_idx / BLOCK_K,
                     num_tma_multicast_b);

            full_barrier.arrive_and_expect_tx(
                SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                SMEM_SCALES_A_SIZE_PER_STAGE + SMEM_SCALES_B_SIZE_PER_STAGE);
          }

// Wait unaligned cases
#pragma unroll
          for (uint32_t s = kNumInnerStages; s < kNumStages; ++s) {
            empty_barriers[s]->wait(
                (scheduler.current_iter * num_iterations + k_iter + 1) & 1);
            full_barriers[s]->arrive();
          }
        });

        // Issue TMA D
        empty_barriers[kNumStages]->wait((scheduler.current_iter + 1) & 1);
        auto& full_barrier = *full_barriers[kNumStages];
        tma_copy(&tensor_map_d,
                 reinterpret_cast<uint64_t*>(&full_barrier),
                 smem_d,
                 n_block_idx * BLOCK_N,
                 m_block_idx * BLOCK_M,
                 1);
        full_barrier.arrive_and_expect_tx(SMEM_D_SIZE);
      }

      // To safely deconstruct distributed shared barriers, we need another
      // round of empty waits
      if constexpr (kNumTMAMulticast > 1) {
#pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s)
          empty_barriers[s]->wait(
              (scheduler.current_iter * num_iterations + 1) & 1);
      }
    }
  } else {
    // Math warp-groups for WGMMA
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto math_wg_idx =
        __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
    const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
    const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

    // Empty barrier arrival
    auto empty_barrier_arrive = [&](int s) {
      if constexpr (kNumTMAMulticast == 1) {
        lane_idx == 0 ? empty_barriers[s]->arrive() : void();
      } else {
        auto target_cta = scheduler.is_peer_cta_alive
                              ? lane_idx
                              : cute::block_rank_in_cluster();
        lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta)
                                    : void();
      }
    };

    // Persistently schedule over blocks
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      // Decide the number of scales B to load
      DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
      cutlass::arch::NamedBarrier(kNumMathThreads).sync();

      // Accumulation for WGMMA or CUDA promotion
      constexpr int WAVE_BLOCK_M = WGMMA::M * get_num_math_warpgroups(BLOCK_M);
      float accum[WGMMA::kNumAccum],
          final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
      float2 scales_b[WGMMA::kNumAccum / 4];

      // Launch MMAs
      launch_k_iterations([&](int k_iter, auto type) {
        constexpr bool kHasDivisibleStages =
            std::is_same_v<decltype(type), DivisibleK>;
        constexpr int kNumInnerStages =
            kHasDivisibleStages ? kNumStages : kNumLastStages;
        DG_STATIC_ASSERT(kNumInnerStages != 0,
                         "Invalid number of inner stages");

#pragma unroll
        for (int s = 0; s < kNumInnerStages; ++s) {
          // Wait TMA arrivals
          full_barriers[s]->wait(
              (scheduler.current_iter * num_iterations + k_iter) & 1);

#pragma unroll
          for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M;
               ++local_idx) {
            auto m_offset = local_idx * WAVE_BLOCK_M;

            // Read A scales
            auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0 + m_offset);
            auto scale_a_1 = ld_shared(smem_scales_a[s] + r_1 + m_offset);

// Commit WGMMA instructions
#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum; ++i)
              warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();
#pragma unroll
            for (int k = 0; k < BLOCK_K / WGMMA::K; ++k) {
              auto desc_a = make_smem_desc(
                  smem_a[s] + (math_wg_idx * WGMMA::M + m_offset) * BLOCK_K +
                      k * WGMMA::K,
                  1);
              auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
              WGMMA::wgmma(desc_a, desc_b, accum, k);
            }
            warpgroup_commit_batch();

            // Read B scales at the first warpgroup wave
            if (local_idx == 0) {
#pragma unroll
              for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                scales_b[i] = ld_shared(reinterpret_cast<float2*>(
                    smem_scales_b[s] + i * 8 + col_idx * 2));
              __syncwarp();
            }

#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum; ++i)
              warpgroup_fence_operand(accum[i]);
            warpgroup_wait<0>();

            // Notify barrier arrival at the last warpgroup wave
            if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
              empty_barrier_arrive(s);

            // Promote with scales
            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
              const float& scale_b_0 = scales_b[i].x;
              const float& scale_b_1 = scales_b[i].y;
              shifted_accum[i * 4 + 0] +=
                  scale_a_0 * scale_b_0 * accum[i * 4 + 0];
              shifted_accum[i * 4 + 1] +=
                  scale_a_0 * scale_b_1 * accum[i * 4 + 1];
              shifted_accum[i * 4 + 2] +=
                  scale_a_1 * scale_b_0 * accum[i * 4 + 2];
              shifted_accum[i * 4 + 3] +=
                  scale_a_1 * scale_b_1 * accum[i * 4 + 3];
            }
          }
        }

        // Wait last TMA store to be finished
        if (k_iter == 0 and scheduler.current_iter > 0) {
          if (threadIdx.x == 0) {
            cute::tma_store_wait<0>();
            empty_barriers[kNumStages]->arrive();
          }
          __syncwarp();
        }

// Wait unaligned cases
#pragma unroll
        for (uint32_t s = kNumInnerStages; s < kNumStages; ++s) {
          full_barriers[s]->wait(
              (scheduler.current_iter * num_iterations + k_iter) & 1);
          empty_barrier_arrive(s);
        }
      });

      // Wait TMA D arrivals
      full_barriers[kNumStages]->wait(scheduler.current_iter & 1);

// Accumulate to D shared memory
#pragma unroll
      for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M;
           ++local_idx) {
        auto m_offset = local_idx * WAVE_BLOCK_M;
        auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
        auto smem_d_0 = reinterpret_cast<float2*>(
            smem_d + (m_offset + r_0) * BLOCK_N + col_idx * 2);
        auto smem_d_1 = reinterpret_cast<float2*>(
            smem_d + (m_offset + r_1) * BLOCK_N + col_idx * 2);
#pragma unroll
        for (auto i = 0; i < WGMMA::kNumAccum / 4; ++i) {
          float2 d_0 = ld_shared(smem_d_0 + i * 4);
          st_shared(smem_d_0 + i * 4,
                    {d_0.x + shifted_accum[i * 4 + 0],
                     d_0.y + shifted_accum[i * 4 + 1]});
          float2 d_1 = ld_shared(smem_d_1 + i * 4);
          st_shared(smem_d_1 + i * 4,
                    {d_1.x + shifted_accum[i * 4 + 2],
                     d_1.y + shifted_accum[i * 4 + 3]});
        }
      }

      cute::tma_store_fence();
      cutlass::arch::NamedBarrier(kNumMathThreads).sync();

      // Use TMA store to write back to global memory
      if (threadIdx.x == 0) {
        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d,
                                      smem_d,
                                      n_block_idx * BLOCK_N,
                                      m_block_idx * BLOCK_M);
        cute::tma_store_arrive();
      }
      __syncwarp();
    }
  }
#else
  if (blockIdx.x == 0 and threadIdx.x == 0)
    DG_DEVICE_ASSERT(false && "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop

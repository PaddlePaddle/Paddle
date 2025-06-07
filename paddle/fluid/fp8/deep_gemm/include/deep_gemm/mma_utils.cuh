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
// Licensed under the MIT License - https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE

#pragma once

#include <cuda.h>

#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>

#include "utils.cuh"

namespace deep_gemm {

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(smem_dst), "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst) {
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
                                 *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                     :: "l"(smem_dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

__forceinline__ __device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__forceinline__ __device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__forceinline__ __device__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

__forceinline__ __device__ uint32_t get_lane_id() {
    uint32_t lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__  __forceinline__ uint32_t ld_shared(const uint32_t* __restrict__ ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int4 ld_shared(const int4* __restrict__ ptr) {
    int4 ret;
    asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float ld_shared(const float* __restrict__ ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(ptr), "r"(val));
}

template <int N>
__device__ void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

union GmmaDescriptor {
    __host__ __device__ constexpr GmmaDescriptor() noexcept: desc_(0) {}

    __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept: desc_(desc) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept: desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept: desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct {
        uint16_t start_address_: 14, : 2;
        uint16_t leading_byte_offset_: 14, : 2;
        uint16_t stride_byte_offset_: 14, : 2;
        uint8_t : 1, base_offset_: 3, : 4;
        uint8_t : 6, layout_type_: 2;
    } bitfield;

    // Decay to an `uint64_t`
    __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

template <class PointerType>
__device__ GmmaDescriptor make_smem_desc(PointerType smem_ptr, int layout_type,
                                         int leading_byte_offset = 0,
                                         int stride_byte_offset = 1024) {
    GmmaDescriptor desc;
    auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

template <int N_, typename MMA>
struct FP8MMA {

    template <size_t ...Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, std::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, std::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP8MMASelector {

    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 40) return MMA_64x40x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 48) return MMA_64x48x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 56) return MMA_64x56x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 72) return MMA_64x72x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 80) return MMA_64x80x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 88) return MMA_64x88x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 96) return MMA_64x96x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 104) return MMA_64x104x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 112) return MMA_64x112x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 120) return MMA_64x120x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 144) return MMA_64x144x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 152) return MMA_64x152x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 160) return MMA_64x160x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_SS_TN();
    }

    static constexpr auto select_type() {
        return FP8MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};

} // namespace deep_gemm

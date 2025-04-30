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

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

// Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
// Copyright (c) NVIDIA Corporation.
// Licensed under the NVSHMEM Software License Agreement (version: September 3, 2019).
// See full license at: https://docs.nvidia.com/nvshmem/api/sla.html
//
// Modified from original source:
//  - nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh

#pragma once

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/utils.cuh"

namespace deep_ep {

EP_STATIC_ASSERT(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64, "Invalid QP minimum depth");

__device__ static __forceinline__
uint64_t HtoBE64(uint64_t x) {
    uint64_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}" : "=l"(ret) : "l"(x));
    return ret;
}

__device__ static __forceinline__
uint32_t HtoBE32(uint32_t x) {
    uint32_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        "prmt.b32 %0, %1, ign, 0x0123;\n\t"
        "}" : "=r"(ret) : "r"(x));
    return ret;
}

__device__ static __forceinline__
uint16_t HtoBE16(uint16_t x) {
    // TODO: simplify PTX using 16-bit instructions
    auto a = static_cast<uint32_t>(x);
    uint32_t d;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x4401;\n\t"
        "mov.b32 ign, 0x0;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(d)
        : "r"(a));
    return static_cast<uint16_t>(d);
}

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;

__device__ static __forceinline__
nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}

__device__ static __forceinline__
nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    auto state = ibgda_get_state();
    const auto num_rc_per_pe = ibgda_get_state()->num_rc_per_pe;
    return &state->globalmem.rcs[pe * num_rc_per_pe + id % num_rc_per_pe];
}

__device__ static __forceinline__
void ibgda_lock_acquire(int *lock) {
    while (atomicCAS(lock, 0, 1) == 1);

    // Prevent reordering before the lock is acquired
    memory_fence_cta();
}

__device__ static __forceinline__
void ibgda_lock_release(int *lock) {
    memory_fence_cta();

    // Prevent reordering before lock is released
    st_na_relaxed(lock, 0);
}

__device__ static __forceinline__
void ibgda_update_dbr(nvshmemi_ibgda_device_qp_t *qp, uint32_t dbrec_head) {
    // `DBREC` contains the index of the next empty `WQEBB`
    __be32 dbrec_val;
    __be32 *dbrec_ptr = qp->tx_wq.dbrec;

    // This is equivalent to `WRITE_ONCE(dbrec_ptr, HtoBE32(dbrec_head & 0xffff))`
    asm("{\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"
        "}"
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    st_na_release(dbrec_ptr, dbrec_val);
}

__device__ static __forceinline__
void ibgda_ring_db(nvshmemi_ibgda_device_qp_t *qp, uint16_t prod_idx) {
    auto bf_ptr = reinterpret_cast<uint64_t*>(qp->tx_wq.bf);
    ibgda_ctrl_seg_t ctrl_seg = {
        .opmod_idx_opcode = HtoBE32(prod_idx << 8),
        .qpn_ds = HtoBE32(qp->qpn << 8)
    };

    EP_STATIC_ASSERT(sizeof(decltype(&ctrl_seg)) == sizeof(uint64_t), "");
    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&ctrl_seg)));
}

__device__ static __forceinline__
void ibgda_post_send(nvshmemi_ibgda_device_qp_t *qp, uint64_t new_prod_idx) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_prod_idx;

    // Update `prod_idx` before ringing the doorbell, so that we know which index is needed in quiet/fence
    ibgda_lock_acquire(&mvars->post_send_lock);

    old_prod_idx = atomicMax(reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.prod_idx), new_prod_idx);
    if (new_prod_idx > old_prod_idx) {
        ibgda_update_dbr(qp, new_prod_idx);
        ibgda_ring_db(qp, new_prod_idx);
    }
    ibgda_lock_release(&mvars->post_send_lock);
}

template <bool kAlwaysDoPostSend>
__device__ static __forceinline__
void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t *qp, uint64_t base_wqe_idx,
                           uint32_t num_wqes, int message_idx = 0) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    // WQE writes must be finished first
    __threadfence();

    // Wait for prior WQE slots to be filled first
    auto *ready_idx = reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.ready_head);
    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx);

    // Always post, not in batch
    constexpr int kNumRequestInBatch = 4;
    if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)
        ibgda_post_send(qp, new_wqe_idx);
}

__device__ static __forceinline__ void
ibgda_write_rdma_write_inl_wqe(nvshmemi_ibgda_device_qp_t *qp, const uint32_t *val, uint64_t raddr,
                               __be32 rkey, uint16_t wqe_idx, void **out_wqes, uint32_t imm) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    auto *ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto *raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto *inl_seg_ptr = reinterpret_cast<mlx5_wqe_inl_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    auto *wqe_data_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(inl_seg_ptr) + sizeof(*inl_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HtoBE32(4 | MLX5_INLINE_SEG);

    // `imm == std::numeric_limits<uint32_t>::max()` means no imm writes
    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | (imm != std::numeric_limits<uint32_t>::max() ? MLX5_OPCODE_RDMA_WRITE_IMM : MLX5_OPCODE_RDMA_WRITE));
    if (imm != std::numeric_limits<uint32_t>::max())
        ctrl_seg.imm = HtoBE32(imm);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*inl_seg_ptr) == 4, "sizeof(*inl_seg_ptr) == 4");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<uint32_t*>(inl_seg_ptr), *reinterpret_cast<const uint32_t*>(&inl_seg));
    st_na_relaxed(reinterpret_cast<uint32_t*>(wqe_data_ptr), *reinterpret_cast<const uint32_t*>(val));
}

__device__ static __forceinline__
uint64_t ibgda_get_lkey_and_rkey(uint64_t laddr, __be32 *lkey,
                                 uint64_t raddr, int dst_pe, uint64_t *out_raddr, __be32 *out_rkey) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
    auto log2_cumem_granularity = state->log2_cumem_granularity;

    // Local key
    uint64_t idx = (laddr - heap_start) >> log2_cumem_granularity;
    auto device_key = state->constmem.lkeys[idx];
    auto lchunk_size = device_key.next_addr - laddr;
    *lkey = device_key.key;

    // Remote key
    uint64_t roffset = raddr - heap_start;
    idx = ((roffset >> log2_cumem_granularity) * nvshmemi_device_state_d.npes) + dst_pe;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        device_key = state->constmem.rkeys[idx];
    } else {
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    }
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;

    // Return the minimum of local and remote chunk sizes
    auto rchunk_size = device_key.next_addr - roffset;
    return min(lchunk_size, rchunk_size);
}

__device__ static __forceinline__ void
ibgda_get_rkey(uint64_t addr, int dst_pe, uint64_t *out_raddr, __be32 *out_rkey) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);

    uint64_t roffset = addr - heap_start;
    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes) + dst_pe;
    nvshmemi_ibgda_device_key_t device_key;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;
}

__device__ static __forceinline__ uint64_t
ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t *qp, uint32_t num_wqes) {
    auto mvars = &qp->mvars;
    return atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_head), static_cast<unsigned long long>(num_wqes));
}

__device__ static __forceinline__ void*
ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t* qp, uint16_t wqe_idx) {
    uint16_t cnt = qp->tx_wq.nwqes;
    uint16_t idx = wqe_idx & (cnt - 1);
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) + (idx << MLX5_SEND_WQE_SHIFT));
}

// Wait until wqe `idx - 1` is completed.
// This is a simplified version of NVSHMEM's `ibgda_poll_cq`. It can only be used for polling recv.
// Because we post recv and poll recv in the same thread, so we don't need to maintain queue status.
__device__ static __forceinline__ void
nvshmemi_ibgda_poll_recv(int dst_pe, int qp_id) {
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    auto cq = qp->rx_wq.cq;

    const uint32_t ncqes = cq->ncqes;
    auto *cqe64 = reinterpret_cast<struct mlx5_cqe64*>(cq->cqe);
    auto old_cons_idx = *cq->cons_idx;
    *cq->cons_idx = old_cons_idx + 1;

    // Wait until `wqe_counter >= old_cons_idx`
    while ((static_cast<uint16_t>(old_cons_idx - HtoBE16(ld_na_relaxed(&cqe64->wqe_counter)) - 1) < ncqes));
}

__device__ static __forceinline__ void
nvshmemi_ibgda_rma_p(int *rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    // Get rkey
    // NOTES: the `p` operation will not cross multiple remote chunks
    __be32 rkey;
    uint64_t raddr;
    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey);

    // Write WQEs
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    uint64_t base_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
    void *wqe_ptrs;
    wqe_ptrs = ibgda_get_wqe_ptr(qp, base_wqe_idx);
    ibgda_write_rdma_write_inl_wqe(qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey, base_wqe_idx, &wqe_ptrs, imm);

    // Submit requests
    ibgda_submit_requests<true>(qp, base_wqe_idx, 1);
}

__device__ static __forceinline__ void
ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr, __be32 lkey,
                           uint64_t raddr, __be32 rkey, uint32_t bytes, uint16_t wqe_idx,
                           void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    auto *ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    void *av_seg_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(av_seg_ptr));
    data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HtoBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HtoBE64(laddr);

    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16, "sizeof(*data_seg_ptr) == 16");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

__device__ static __forceinline__ void
ibgda_write_empty_recv_wqe(void *out_wqe) {
    auto *data_seg_ptr = reinterpret_cast<struct mlx5_wqe_data_seg*>(out_wqe);
    struct mlx5_wqe_data_seg data_seg;

    // Make the first segment in the WQE invalid, then the entire list will be invalid
    data_seg.byte_count = 0;
    data_seg.lkey = HtoBE64(MLX5_INVALID_LKEY);
    data_seg.addr = 0;

    EP_STATIC_ASSERT(sizeof(mlx5_wqe_data_seg) == sizeof(int4), "Invalid data type length");
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

__device__ static __forceinline__ uint64_t
nvshmemi_ibgda_allocate_recvs(nvshmemi_ibgda_device_qp* qp) {
    auto mvars = &qp->mvars;

    // Allocate if not enough
    constexpr int kMinIBGDARecvs = 32;
    auto resv_head = mvars->rx_wq.resv_head;
    auto num_valid_slots = resv_head - mvars->rx_wq.cons_idx;
    if (num_valid_slots < kMinIBGDARecvs) {
        resv_head = mvars->rx_wq.cons_idx + qp->rx_wq.nwqes;
        mvars->rx_wq.resv_head = resv_head;

        // Ensure WQE is written before `dbrec`
        __be32 dbrec_val;
        __be32 *dbrec_ptr = qp->rx_wq.dbrec;

        // Compared to sending, for each QP, we only post recv in a single thread,
        // so we don't need to do synchronization here
        // This is equivalent to `WRITE_ONCE(dbrec_ptr, HtoBE32(wqe_idx & 0xffff))`
        asm("{\n\t"
            ".reg .b32 dbrec_head_16b;\n\t"
            ".reg .b32 ign;\n\t"
            "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"
            "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"
            "}" : "=r"(dbrec_val)
                : "r"(static_cast<uint32_t>(resv_head)));
        st_na_release(dbrec_ptr, dbrec_val);
    }

    // Return old number of slots
    return num_valid_slots;
}

__device__ static __forceinline__ void
nvshmemi_ibgda_prepare_recvs(int dst_rank, int qp_id) {
    // NOTES: only one thread can run this function
    EP_DEVICE_ASSERT(nvshmemi_ibgda_allocate_recvs(ibgda_get_rc(dst_rank, qp_id)) > 16);
}

__device__ static __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    // Get lkey and rkey, store them into lanes
    uint32_t num_wqes = 0;
    __be32 my_lkey = 0;
    uint64_t my_laddr = 0;
    __be32 my_rkey = 0;
    uint64_t my_raddr = 0;
    uint64_t my_chunk_size = 0;

    // Decide how many messages (theoretically 3 for maximum)
    auto remaining_bytes = bytes;
    while (remaining_bytes > 0) {
        if (lane_id == num_wqes)
            my_chunk_size = min(remaining_bytes, ibgda_get_lkey_and_rkey(my_laddr = req_lptr, &my_lkey, req_rptr, dst_pe, &my_raddr, &my_rkey));

        // Move one more message
        auto chunk_size = __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
        remaining_bytes -= chunk_size;
        req_lptr += chunk_size;
        req_rptr += chunk_size;
        ++ num_wqes;
    }
    EP_DEVICE_ASSERT(num_wqes <= 32);

    // Process WQE
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    uint64_t base_wqe_idx = 0;
    if (lane_id == 0)
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
    if (lane_id < num_wqes) {
        auto wqe_ptr = ibgda_get_wqe_ptr(qp, base_wqe_idx + lane_id);
        ibgda_write_rdma_write_wqe(qp, my_laddr, my_lkey, my_raddr, my_rkey, my_chunk_size,
                                   base_wqe_idx, &wqe_ptr);
    }
    __syncwarp();

    // Submit
    if (lane_id == 0)
        ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, message_idx);
    __syncwarp();
}

__device__ static __forceinline__ void ibgda_write_amo_add_wqe(
        nvshmemi_ibgda_device_qp_t *qp, const int &value,
        uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey,
        uint16_t wqe_idx, void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg = {0};
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_atomic_seg atomic_seg_1;
    struct mlx5_wqe_data_seg data_seg;

    auto ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto atomic_seg_ptr = reinterpret_cast<mlx5_wqe_atomic_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    auto data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(atomic_seg_ptr) + sizeof(*atomic_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    // NOTES: `0x08000000` means `IBGDA_4_BYTE_EXT_AMO_OPMOD`
    ctrl_seg.opmod_idx_opcode = HtoBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | 0x08000000);
    auto atomic_32_masked_fa_seg = reinterpret_cast<ibgda_atomic_32_masked_fa_seg_t*>(&atomic_seg_1);
    atomic_32_masked_fa_seg->add_data = HtoBE32(value);
    atomic_32_masked_fa_seg->field_boundary = 0;

    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 4);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

    data_seg.byte_count = HtoBE32(sizeof(int));
    data_seg.lkey = lkey;
    data_seg.addr = HtoBE64(laddr);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*atomic_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == sizeof(int4), "Invalid vectorization");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(atomic_seg_ptr), *reinterpret_cast<int4*>(&atomic_seg_1));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<int4*>(&data_seg));
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id) {
    nvshmemi_ibgda_device_qp_t *qp = ibgda_get_rc(pe, qp_id);

    __be32 rkey;
    uint64_t raddr;
    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), pe, &raddr, &rkey);

    uint64_t my_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
    void *wqe_ptrs = ibgda_get_wqe_ptr(qp, my_wqe_idx);

    ibgda_write_amo_add_wqe(qp, value, reinterpret_cast<uint64_t>(qp->ibuf.buf),
                            qp->ibuf.lkey, raddr, rkey, my_wqe_idx, &wqe_ptrs);

    ibgda_submit_requests<true>(qp, my_wqe_idx, 1);
}

} // namespace deep_ep

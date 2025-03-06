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

#include <cstring>
#include <vector>

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/launch.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/utils.cuh"
// #include "ibgda_device.cuh"

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** task_fifo_ptrs, int head, int rank) {
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

void barrier(int** task_fifo_ptrs,
             int head,
             int rank,
             int num_ranks,
             cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                 \
  LAUNCH_KERNEL(&cfg, barrier<ranks>, task_fifo_ptrs, head, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 32, stream);
  SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

// namespace internode {

// nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
// nvshmem_team_config_t cpu_rdma_team_config;

// std::vector<uint8_t> get_unique_id() {
//     nvshmemx_uniqueid_t unique_id;
//     nvshmemx_get_uniqueid(&unique_id);
//     std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
//     std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
//     return result;
// }

// __global__ void ibgda_initialize_recv_queue(int rank) {
//     auto thread_idx = static_cast<int>(threadIdx.x);
//     auto num_threads = static_cast<int>(blockDim.x);

//     auto dst_rank = static_cast<int>(blockIdx.x);
//     if (dst_rank != rank) {
//         for (int qp_id = thread_idx; qp_id <
//         ibgda_get_state()->num_rc_per_pe; qp_id += num_threads) {
//             auto qp = ibgda_get_rc(dst_rank, qp_id);

//             // Clean some necessary variables
//             for (int i = 0; i < qp->rx_wq.nwqes; ++ i)
//                 ibgda_write_empty_recv_wqe(ibgda_get_wqe_ptr(qp, i));
//             qp->mvars.rx_wq.resv_head = 0;
//             qp->mvars.rx_wq.cons_idx = 0;

//             // Allocate receive slots
//             nvshmemi_ibgda_allocate_recvs(qp);
//         }
//     }
// }

// int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int
// num_ranks, bool low_latency_mode) {
//     nvshmemx_uniqueid_t root_unique_id;
//     nvshmemx_init_attr_t attr;
//     std::memcpy(&root_unique_id, root_unique_id_val.data(),
//     sizeof(nvshmemx_uniqueid_t)); nvshmemx_set_attr_uniqueid_args(rank,
//     num_ranks, &root_unique_id, &attr);
//     nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

//     // Create sub-RDMA teams
//     // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency
//     kernels are used if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS)
//     {
//         EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
//         EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
//         EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, rank %
//         NUM_MAX_NVL_PEERS, NUM_MAX_NVL_PEERS,
//                                                   num_ranks /
//                                                   NUM_MAX_NVL_PEERS,
//                                                   &cpu_rdma_team_config, 0,
//                                                   &cpu_rdma_team) == 0);
//         EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
//     }

//     // Normal operations use IBRC, while low-latency operations use IBGDA
//     if (low_latency_mode) {
//         nvshmemi_device_host_state_t* dev_state_ptr = nullptr;
//         CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_state_ptr),
//         nvshmemi_device_state_d));

//         bool ibgda_is_initialized = false;
//         cudaMemcpy(&dev_state_ptr->ibgda_is_initialized,
//         &ibgda_is_initialized, sizeof(bool), cudaMemcpyHostToDevice);

//         // Initialize recv queues for low-latency mode AR
//         ibgda_initialize_recv_queue<<<num_ranks, 128>>>(rank);
//     }
//     nvshmem_barrier_all();
//     return nvshmem_my_pe();
// }

// void* alloc(size_t size, size_t alignment) {
//     return nvshmem_align(alignment, size);
// }

// void free(void* ptr) {
//     nvshmem_free(ptr);
// }

// void barrier() {
//     nvshmem_barrier_all();
// }

// void finalize() {
//     if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
//         nvshmem_team_destroy(cpu_rdma_team);
//         cpu_rdma_team = NVSHMEM_TEAM_INVALID;
//     }
//     nvshmem_finalize();
// }

// } // namespace internode

}  // namespace deep_ep

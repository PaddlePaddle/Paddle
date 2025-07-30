/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <memory>
#include <vector>
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/phi/backends/dynload/nccl.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/timer.h"
#include "thrust/pair.h"
#elif defined(PADDLE_WITH_XPU_KP)
#include <xpu/runtime.h>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif

#include "paddle/fluid/framework/barrier.h"
#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/memory.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

#define TYPEALIGN(ALIGNVAL, LEN) \
  (((uint64_t)(LEN) + ((ALIGNVAL)-1)) & ~((uint64_t)((ALIGNVAL)-1)))

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
class HeterComm {
  using HeterCommType = HeterComm<KeyType, ValType, GradType, GPUAccessor>;
  static const int COPY_KEY = 0x01;
  static const int COPY_VAL = 0x02;
  static const int COPY_ALL = COPY_KEY | COPY_VAL;

 public:
  HeterComm(size_t capacity, std::shared_ptr<HeterPsResource> resource);
  HeterComm(size_t capacity,
            std::shared_ptr<HeterPsResource> resource,
            GPUAccessor& gpu_accessor);  // NOLINT
  virtual ~HeterComm();
  HeterComm(const HeterComm&) = delete;
  HeterComm& operator=(const HeterComm&) = delete;
  // reset table
  void reset_table(const int dev_id,
                   size_t capacity,
                   const OptimizerConfig& sgd_config,
                   const OptimizerConfig& embedx_config,
                   bool infer_mode);
  void set_mode(bool infer_mode);
  template <typename StreamType>
  size_t merge_keys(const int gpu_num,
                    const KeyType* d_keys,
                    const size_t& len,
                    KeyType* d_sorted_keys,
                    KeyType* d_merged_keys,
                    uint32_t* d_restore_idx,
                    StreamType stream);
  void dynamic_merge_grad(int gpu_num,
                          KeyType* d_keys,
                          float* d_grads,
                          size_t len,
                          int& uniq_len,        // NOLINT
                          size_t& segment_len,  // NOLINT
                          bool enable_segment_merge_grad);
  void segment_merge_grad(int gpu_num,
                          KeyType* d_keys,
                          float* d_grads,
                          const uint32_t* d_index,
                          size_t len,
                          const uint32_t* d_fea_num_info,
                          size_t uniq_len,
                          size_t& segment_len);  // NOLINT
  void build_ps(int num,
                KeyType* h_keys,
                ValType* h_vals,
                size_t len,
                size_t chunk_size,
                int stream_num,
                int offset = -1);
  void split_input_to_shard(KeyType* d_keys,
                            int* d_idx_ptr,
                            size_t len,
                            int* left,
                            int* right,
                            int gpu_num);
  void merge_grad(int gpu_num,
                  KeyType* d_keys,
                  GradType* d_grads,
                  size_t len,
                  int& uniq_len);  // NOLINT
  void dynamic_merge_grad(int gpu_num,
                          KeyType* d_keys,
                          float* d_grads,
                          size_t len,
                          int& uniq_len);  // NOLINT
  void pull_sparse(int num, KeyType* d_keys, float* d_vals, size_t len);
  void build_ps(int num,
                KeyType* h_keys,
                char* pool,
                size_t len,
                size_t feature_value_size,
                size_t chunk_size,
                int stream_num);
  void dump();
  void show_one_table(int gpu_num);
  void show_table_collisions();
  int get_index_by_devid(int devid);

#if defined(PADDLE_WITH_CUDA)
  template <typename Sgd>
  void push_sparse(int num,
                   KeyType* d_keys,
                   float* d_grads,
                   size_t len,
                   Sgd& sgd);  // NOLINT
#elif defined(PADDLE_WITH_XPU_KP)
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len);
#endif

  void set_sparse_sgd(const OptimizerConfig& optimizer_config);
  void set_embedx_sgd(const OptimizerConfig& optimizer_config);

  int log2i(int x);

  template <typename DstPlace, typename SrcPlace, typename StreamType>
  void memory_copy(DstPlace dst_place,
                   void* dst,
                   SrcPlace src_place,
                   const void* src,
                   size_t count,
                   StreamType stream = 0);
  template <typename StreamType>
  void MemcpyPeerAsync(void* dst,
                       const void* src,
                       size_t count,
                       StreamType stream);

#if defined(PADDLE_WITH_CUDA)
  template <typename Sgd>
  void update_one_table(int num,
                        KeyType* d_keys,
                        GradType* d_grads,
                        size_t len,
                        Sgd& sgd);  // NOLINT

  void set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                              const std::vector<ncclComm_t>& inter_comms,
                              int comm_size,
                              int rank_id) {
    nccl_inner_comms_ = inner_comms;
    nccl_inter_comms_ = inter_comms;
    node_size_ = comm_size;
    rank_id_ = rank_id;
  }

  void set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) {
    multi_mf_dim_ = multi_mf_dim;
    max_mf_dim_ = max_mf_dim;
  }

#endif

  bool need_transfer(int send_id, int receive_id) {
    return ((send_id / 4 != receive_id / 4) &&
            (send_id + 4) % device_num_ != receive_id);
  }

  // void dump_to_cpu(int index);

  int get_transfer_devid(int send_id) { return (send_id + 4) % device_num_; }

  void end_pass();
#if defined(PADDLE_WITH_CUDA)
  // dedup
  int dedup_keys_and_fillidx(const int gpu_id,
                             const int total_fea_num,
                             const KeyType* d_keys,   // input
                             KeyType* d_merged_keys,  // output
                             KeyType* d_sorted_keys,
                             uint32_t* d_restore_idx,
                             uint32_t* d_sorted_idx,
                             uint32_t* d_offset,
                             uint32_t* d_merged_cnts,
                             bool filter_zero,
                             cudaStream_t stream = 0);
#endif
  template <typename T, typename StreamType>
  void split_idx_to_shard(KeyType* d_keys,
                          T* d_idx_ptr,
                          size_t len,
                          T* left,
                          T* right,
                          int gpu_num,
                          StreamType stream);

  struct Node {
    ppStream in_stream;
    ppStream out_stream;
    char* key_storage;
    char* val_storage;
    int sync;
    size_t key_bytes_len;
    size_t val_bytes_len;
    int dev_num;
  };

  struct Path {
    std::vector<Node> nodes_;
  };

  struct CopyTask {
    Path* path;
    int step;
    CopyTask(Path* path_, int step_) : path(path_), step(step_) {}
  };
  // inner card
  struct InnerResource {
    uint32_t* d_idx = nullptr;
    size_t* h_part_sizes = nullptr;
    std::vector<size_t> h_offsets;
    uint32_t* d_offset_ptr = nullptr;

    KeyType* d_keys_parted = nullptr;
    char* d_vals_parted = nullptr;
    std::vector<KeyType*> d_remote_keys;
    std::vector<char*> d_remote_vals;
    KeyType* d_trans_keys = nullptr;
    char* d_trans_vals = nullptr;

    // resize vector
    void resize(const int num_gpu) {
      h_offsets.resize(num_gpu);
      d_remote_keys.resize(num_gpu);
      d_remote_vals.resize(num_gpu);
    }
  };
  // Resource for partition shard Key by nodes
  struct ShardResource {
    uint32_t* d_local_idx_parted = nullptr;  // uint32_t for multisplit
    std::vector<size_t> h_local_part_sizes;
    std::vector<size_t> h_local_part_offsets;
    std::vector<size_t> h_remote_part_sizes;
    std::vector<size_t> h_remote_part_offsets;
    uint32_t* d_node_size_ptr = nullptr;
    std::vector<uint32_t> h_push_fea_sizes;
    // shard part
    void resize_part_size(const int node_size) {
      if (h_local_part_sizes.size() >= static_cast<size_t>(node_size)) {
        return;
      }
      h_local_part_sizes.resize(node_size);
      h_local_part_offsets.resize(node_size + 1);
      h_remote_part_sizes.resize(node_size);
      h_remote_part_offsets.resize(node_size + 1);
      h_push_fea_sizes.resize(node_size * node_size);
    }
  };
  // pull partition shard key by devices
  struct PullResource {
    size_t h_recv_fea_num = 0;
    uint32_t* d_restore_keys_idx = nullptr;
  };

  struct LocalStorage {
    LocalStorage() { sem_wait = std::make_unique<Semaphore>(); }
    void init(int device_num, int dev_id, phi::Stream stream) {
      place_ = phi::GPUPlace(dev_id);
      h_recv_offsets.resize(device_num);
      h_fea_sizes.resize(device_num);
      stream_ = stream;
    }
    template <typename T>
    T* alloc_cache(const size_t& len,
                   std::shared_ptr<memory::Allocation>& alloc,  // NOLINT
                   bool need_copy = false) {
      size_t need_mem = len * sizeof(T);
      if (alloc.get() == nullptr) {
        alloc = memory::Alloc(place_, need_mem, stream_);
      } else if (need_mem > alloc->size()) {
        if (need_copy) {
          std::shared_ptr<memory::Allocation> tmp =
              memory::Alloc(place_, need_mem);
#if defined(PADDLE_WITH_CUDA)
          PADDLE_ENFORCE_GPU_SUCCESS(
              cudaMemcpyAsync(tmp->ptr(),  // output
                              alloc->ptr(),
                              alloc->size(),
                              cudaMemcpyDeviceToDevice,
                              reinterpret_cast<cudaStream_t>(stream_.id())));
          PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(
              reinterpret_cast<cudaStream_t>(stream_.id())));
#else
          memory::Copy(place_,
                       tmp->ptr(),
                       place_,
                       alloc->ptr(),
                       alloc->size(),
                       reinterpret_cast<void*>(stream_.id()));
#endif
          alloc.reset();
          alloc = tmp;
        } else {
          alloc.reset();
          alloc = memory::Alloc(place_, need_mem, stream_);
        }
      }
      return reinterpret_cast<T*>(alloc->ptr());
    }
    void alloc(const size_t& len,
               const size_t& value_bytes = sizeof(GradType),
               const int copy_mode = 0) {
      all_keys =
          alloc_cache<KeyType>(len, all_keys_mem, (copy_mode & COPY_KEY));
      all_grads = alloc_cache<char>(
          len * value_bytes, all_grads_mem, (copy_mode & COPY_VAL));
      local_keys =
          alloc_cache<KeyType>(len, local_keys_mem, (copy_mode & COPY_KEY));
      local_grads = alloc_cache<char>(
          len * value_bytes, local_grads_mem, (copy_mode & COPY_VAL));
      d_merged_keys = all_keys;
      d_merged_push_keys = local_keys;
      d_merged_vals = all_grads;
      d_merged_push_vals = local_grads;
    }
    void check(const size_t& len,
               const size_t& value_bytes = sizeof(GradType)) {
      PADDLE_ENFORCE_GE(all_keys_mem->size(),
                        len,
                        common::errors::InvalidArgument(
                            "Invalid size of all keys memory. Expect to be "
                            "equal to length %d. But received %d.",
                            len,
                            all_keys_mem->size()));
      PADDLE_ENFORCE_GE(
          all_grads_mem->size(),
          len * value_bytes,
          common::errors::InvalidArgument(
              "Invalid size of all gradients memory. Expect to be equal to "
              "length * value bytes %d. But received %d.",
              len * value_bytes,
              all_grads_mem->size()));
    }
    void init_pull(const size_t& len) {
      pull_res.h_recv_fea_num = len;
      pull_res.d_restore_keys_idx = alloc_cache<uint32_t>(len, local_pull_idx);
    }
    void init_shard(const size_t& len, const size_t& node_size) {
      shard_res.d_local_idx_parted =
          alloc_cache<uint32_t>(len, local_shard_idx);
      shard_res.d_node_size_ptr =
          alloc_cache<uint32_t>(node_size * node_size, d_node_size_buf);
      shard_res.resize_part_size(node_size);
    }
    void init_inner(const size_t& len, const int& device_num) {
      inner_res.d_idx = alloc_cache<uint32_t>(len, local_inner_idx);
      inner_res.d_offset_ptr =
          alloc_cache<uint32_t>(device_num * 2, inner_offset);
      inner_res.resize(device_num);
    }
    void init_trans(const size_t& fea_num, const size_t& value_bytes) {
      d_merged_trans_keys = alloc_cache<KeyType>(fea_num * 2, trans_keys_buff);
      d_merged_push_trans_keys = &d_merged_trans_keys[fea_num];
      d_merged_trans_vals =
          alloc_cache<char>(fea_num * 2 * value_bytes, trans_vals_buff);
      d_merged_push_trans_vals = &d_merged_trans_vals[fea_num * value_bytes];
    }

#if defined(PADDLE_WITH_CUDA)
    phi::GPUPlace place_;

#elif defined(PADDLE_WITH_XPU_KP)
    phi::XPUPlace place_;
#endif
    phi::Stream stream_;
    std::shared_ptr<memory::Allocation> all_keys_mem = nullptr;
    std::shared_ptr<memory::Allocation> all_grads_mem = nullptr;

    KeyType* all_keys;
    char* all_grads;

    std::shared_ptr<memory::Allocation> local_keys_mem = nullptr;
    std::shared_ptr<memory::Allocation> local_grads_mem = nullptr;
    KeyType* local_keys;
    char* local_grads;

    // all2all
    std::shared_ptr<memory::Allocation> local_inner_idx = nullptr;
    std::shared_ptr<memory::Allocation> local_pull_idx = nullptr;
    std::shared_ptr<memory::Allocation> local_shard_idx = nullptr;
    std::shared_ptr<memory::Allocation> inner_offset = nullptr;
    std::shared_ptr<memory::Allocation> d_node_size_buf = nullptr;

    InnerResource inner_res;
    ShardResource shard_res;
    PullResource pull_res;

    KeyType* d_merged_keys = nullptr;
    char* d_merged_vals = nullptr;
    KeyType* d_merged_push_keys = nullptr;
    char* d_merged_push_vals = nullptr;
    std::vector<size_t> h_recv_offsets;
    std::vector<size_t> h_fea_sizes;
    // inner trans comm and stream buffer
    size_t h_trans_size;
    size_t h_trans_offset;

    // node trans comm and stream buffer
    std::unique_ptr<Semaphore> sem_wait;
    std::shared_ptr<memory::Allocation> trans_keys_buff = nullptr;
    std::shared_ptr<memory::Allocation> trans_vals_buff = nullptr;
    KeyType* d_merged_trans_keys = nullptr;
    char* d_merged_trans_vals = nullptr;
    KeyType* d_merged_push_trans_keys = nullptr;
    char* d_merged_push_trans_vals = nullptr;

    platform::Timer all2all_span_;
    platform::Timer inner_span_;
    platform::Timer inner_barrier_;
    platform::Timer node_span_;
    platform::Timer node_barrier_;

    platform::Timer node_wait_;
    platform::Timer node_trans_;
    platform::Timer node_p2p_;
    platform::Timer local_oper_;
    platform::Timer nvcomp_comp_;
    platform::Timer nvcomp_decomp_;
    size_t total_keys_ = 0;
    size_t local_keys_ = 0;
    size_t remote_keys_ = 0;
  };

  void init_path();

  template <typename StreamType>
  void sync_stream(const StreamType& stream) {
#if defined(PADDLE_WITH_CUDA)
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#elif defined(PADDLE_WITH_XPU_KP)
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(stream));
#endif
  }

  template <typename StreamType>
  void create_stream(StreamType* stream) {
#if defined(PADDLE_WITH_CUDA)
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(stream));
#elif defined(PADDLE_WITH_XPU_KP)
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_create(stream));
#endif
  }

  template <typename StreamType>
  void destroy_stream(StreamType stream) {
#if defined(PADDLE_WITH_CUDA)
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream));
#elif defined(PADDLE_WITH_XPU_KP)
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_destroy(stream));
#endif
  }

  void create_storage(int start_index,
                      int end_index,
                      size_t keylen,
                      size_t vallen);
  void create_tmp_storage(void*& dest,  // NOLINT
                          int start_index,
                          int end_index,
                          size_t vallen);
  void destroy_tmp_storage(void*& p, int start_index, int end_index);  // NOLINT
  void destroy_storage(int start_index, int end_index);
  void walk_to_dest(int start_index,
                    int gpu_num,
                    int* h_left,
                    int* h_right,
                    KeyType* src_key,
                    GradType* src_val);
  void walk_to_dest(int start_index,
                    int gpu_num,
                    int* h_left,
                    int* h_right,
                    KeyType* src_key,
                    char* src_val,
                    size_t val_size);
  void walk_to_src(int start_index,
                   int gpu_num,
                   int* h_left,
                   int* h_right,
                   ValType* src_val);
  void walk_to_src(int start_index,
                   int gpu_num,
                   int* h_left,
                   int* h_right,
                   char* src_val,
                   size_t val_size);

 protected:
  void pull_merge_sparse(const int gpu_id,
                         KeyType* d_keys,
                         float* d_vals,
                         size_t len);
  void pull_normal_sparse(const int gpu_id,
                          KeyType* d_keys,
                          float* d_vals,
                          size_t len);
  void pull_one_table(const int gpu_id,
                      KeyType* d_keys,
                      float* d_vals,
                      const size_t& len,
                      const cudaStream_t& stream);

  // node all2all pull
  void pull_sparse_all2all(const int& gpu_id,
                           KeyType* d_keys,
                           float* d_vals,
                           const size_t& len);

  template <typename Sgd>
  void push_normal_sparse(int num,
                          KeyType* d_keys,
                          float* d_grads,
                          size_t len,
                          Sgd& sgd);  // NOLINT

  void shard_inner_keys(const size_t& total_fea_num,
                        const KeyType* d_keys,
                        const int& gpu_id,
                        const int& gpu_num,
                        InnerResource* res,
                        const cudaStream_t& stream);
  void gather_inner_keys_p2p(const size_t& total_fea_num,
                             const KeyType* d_keys,
                             InnerResource& res,  // NOLINT
                             const int& gpu_id,
                             const int& gpu_num,
                             const int& trans_id,
                             const cudaStream_t& stream);
  size_t gather_inner_keys_by_copy(const int& gpu_id,
                                   const size_t& fea_size,
                                   const KeyType* d_keys,
                                   const cudaStream_t& stream);
  void partition_shard_keys(const int& gpu_id,
                            const size_t& total_fea_num,
                            const KeyType* d_keys,
                            uint32_t* d_idx_parted,
                            KeyType* d_keys_parted,
                            size_t* h_part_sizes,
                            const int& shard_num,
                            const cudaStream_t& stream);
  size_t send_data_by_all2all(const int& gpu_id,
                              const int& nccl_node_size,
                              const int& nccl_rank_id,
                              const int& value_bytes,
                              const size_t* h_send_part_sizes,
                              const size_t* h_send_part_offsets,
                              const size_t* h_recv_part_sizes,
                              const size_t* h_recv_part_offsets,
                              const char* d_send_buff,
                              char* d_rev_buff,
                              const cudaStream_t& stream);
  size_t gather_inter_keys_by_all2all(const int& gpu_id,
                                      const size_t& fea_size,
                                      const KeyType* d_in_keys,
                                      const cudaStream_t& stream,
                                      bool debug = false);
  void scatter_inter_vals_by_all2all(const int& gpu_id,
                                     const size_t& fea_size,
                                     const char* d_in_vals,
                                     void* d_out_vals,
                                     const size_t& value_bytes,
                                     void* d_tmp_vals,
                                     const cudaStream_t& stream);
  void recalc_local_and_remote_size(const int& gpu_id,
                                    const size_t& pull_size,
                                    const size_t& node_num,
                                    const uint32_t* d_tmp_size_list,
                                    const uint32_t* d_inter_size_list,
                                    const cudaStream_t& stream);

  template <typename T>
  void scatter_inter_vals_by_all2all_common(const int& gpu_id,
                                            const size_t& len,
                                            const size_t& value_bytes,
                                            const T* d_in_vals,
                                            T* d_out_vals,
                                            T* d_tmp_vals,
                                            const cudaStream_t& stream,
                                            bool sage = false,
                                            bool slot = false) {
    AnyDeviceGuard guard(resource_->dev_id(gpu_id));
    auto& cache = storage_[gpu_id];
    auto& res = cache.shard_res;

    auto h_local_part_sizes = res.h_local_part_sizes.data();
    auto h_local_part_offsets = res.h_local_part_offsets.data();

    auto h_remote_part_sizes = res.h_remote_part_sizes.data();
    auto h_remote_part_offsets = res.h_remote_part_offsets.data();

    size_t total_fea_num = 0;
    if (rdma_checker_->need_rdma_trans() && !sage) {
      // Sage mode can not run this branch currently, otherwise the process will
      // hang here.
      total_fea_num =
          send_vals_by_all2all_trans(gpu_id,
                                     rank_id_,
                                     node_size_,
                                     reinterpret_cast<const char*>(d_in_vals),
                                     reinterpret_cast<char*>(d_tmp_vals),
                                     value_bytes,
                                     stream);
    } else {
      // sage is true, set default to run here.
      total_fea_num =
          send_data_by_all2all(gpu_id,
                               node_size_,
                               rank_id_,
                               value_bytes,
                               h_remote_part_sizes,
                               h_remote_part_offsets,
                               h_local_part_sizes,
                               h_local_part_offsets,
                               reinterpret_cast<const char*>(d_in_vals),
                               reinterpret_cast<char*>(d_tmp_vals),
                               stream);
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

    // fill vals
    // slot feature don't need scatter
    if (!slot) {
      heter_comm_kernel_->scatter_vals(
          reinterpret_cast<const T*>(d_tmp_vals),  // in
          reinterpret_cast<T*>(d_out_vals),        // out
          res.d_local_idx_parted,
          len,
          value_bytes,
          stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  void scatter_inner_vals_p2p(const size_t& total_fea_num,
                              void* d_out_vals,
                              InnerResource& res,  // NOLINT
                              const int& gpu_id,
                              const int& gpu_num,
                              const int& trans_id,
                              const size_t& value_bytes,
                              const cudaStream_t& stream);
  void scatter_inner_vals_by_copy(const int& gpu_id,
                                  const size_t& fea_size,
                                  const char* d_in_vals,
                                  void* d_out_vals,
                                  const size_t& value_bytes,
                                  const cudaStream_t& stream);
  void gather_inner_data_p2p(const size_t& total_fea_num,
                             const KeyType* d_keys,
                             const void* d_vals,
                             InnerResource& res,  // NOLINT
                             const int& gpu_id,
                             const int& gpu_num,
                             const int& trans_id,
                             const size_t& value_bytes,
                             const cudaStream_t& stream);
  template <typename Sgd>
  void push_sparse_all2all(const int& gpu_id,
                           KeyType* d_keys,
                           float* d_grads,
                           const size_t& len,
                           Sgd& sgd);  // NOLINT
  size_t merge_grad(const int& gpu_id,
                    const size_t& len,
                    const KeyType* d_in_keys,
                    KeyType* d_out_keys,
                    const void* d_in_grads,
                    void* d_out_grads,
                    const cudaStream_t& stream);
  size_t gather_inner_gradient_by_copy(const int& gpu_id,
                                       const size_t& push_size,
                                       KeyType* d_keys,
                                       void* d_push_vals,
                                       const size_t& value_bytes,
                                       const cudaStream_t& stream);
  size_t gather_sparse_gradient_by_all2all(const int& gpu_id,
                                           const size_t& push_size,
                                           const KeyType* d_keys,
                                           const char* d_push_vals,
                                           const size_t& value_bytes,
                                           KeyType* d_out_keys,
                                           KeyType* d_tmp_keys,
                                           char* d_out_vals,
                                           char* d_tmp_vals,
                                           const cudaStream_t& stream);
  size_t send_keys_by_all2all_trans(const int& gpu_id,
                                    const int& rank_id,
                                    const int& node_size,
                                    const size_t& fea_size,
                                    const KeyType* d_in_keys,
                                    KeyType* d_out_keys,
                                    const cudaStream_t& stream);
  size_t send_vals_by_all2all_trans(const int& gpu_id,
                                    const int& rank_id,
                                    const int& node_size,
                                    const char* d_in_vals,
                                    char* d_out_vals,
                                    const size_t& value_bytes,
                                    const cudaStream_t& stream);
  size_t send_gradient_by_all2all_trans(const int& gpu_id,
                                        const int& rank_id,
                                        const int& node_size,
                                        const size_t& fea_size,
                                        const KeyType* d_keys,
                                        const char* d_push_vals,
                                        const size_t& value_bytes,
                                        KeyType* d_out_keys,
                                        char* d_out_vals,
                                        const cudaStream_t& stream);
  // debug time
  void print_debug_time(const int& gpu_id, bool force = false);
  // alloc temp memory
  template <typename T, typename TPlace>
  T* AllocCache(std::shared_ptr<memory::Allocation>* alloc,
                const TPlace& place,
                const size_t& byte_len) {
    if (alloc->get() == nullptr || byte_len > (*alloc)->size()) {
      alloc->reset();
      if (resource_->multi_mf()) {
        *alloc = memory::Alloc(place, byte_len);
      } else {
        auto stream = resource_->local_stream(place.GetDeviceId(), 0);
        auto id = phi::Stream(reinterpret_cast<phi::StreamId>(stream));
        *alloc = memory::Alloc(place, byte_len, id);
      }
    }
    return reinterpret_cast<T*>((*alloc)->ptr());
  }
  template <typename TPlace>
  std::shared_ptr<memory::Allocation> MemoryAlloc(const TPlace& place,
                                                  const size_t& byte_len) {
    if (resource_->multi_mf()) {
      return memory::Alloc(place, byte_len);
    } else {
      auto stream = resource_->local_stream(place.GetDeviceId(), 0);
      auto id = phi::Stream(reinterpret_cast<phi::StreamId>(stream));
      return memory::Alloc(place, byte_len, id);
    }
  }

  using Table = HashTable<KeyType, ValType>;
  using PtrTable = HashTable<KeyType, float*>;
  std::vector<Table*> tables_;
  std::vector<PtrTable*> ptr_tables_;
  std::shared_ptr<HeterPsResource> resource_;
  std::vector<std::vector<Path>> path_;
  float load_factor_{0.75};
  int block_size_{256};
  std::unique_ptr<HeterCommKernel> heter_comm_kernel_;

  GPUAccessor gpu_accessor_;

 protected:
  int topo_aware_{0};
  std::vector<LocalStorage> storage_;
  DynamicGradMerger merger_;
  int device_num_ = 8;
  int multi_node_{0};
  int rank_id_ = 0;
  int node_size_ = 1;
  // inner sync barrier
  Barrier barrier_;
  size_t val_type_size_;
  size_t pull_type_size_;
  size_t grad_type_size_;
  size_t max_type_size_;
  bool enable_gpu_direct_access_ = false;
  // set compress bound
  float max_value_bound_ = 10.0;
  float max_grad_bound_ = 10.0;

#if defined(PADDLE_WITH_CUDA)
  GpuRDMAChecker* rdma_checker_ = nullptr;
  std::vector<ncclComm_t> nccl_inner_comms_;
  std::vector<ncclComm_t> nccl_inter_comms_;
  int multi_mf_dim_{8};
  int max_mf_dim_ = 8;
  std::vector<std::shared_ptr<cub::CachingDeviceAllocator>> allocators_;
#endif
  int64_t start_time_ = 0;
  bool is_infer_mode_ = false;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_inl.h"

#endif

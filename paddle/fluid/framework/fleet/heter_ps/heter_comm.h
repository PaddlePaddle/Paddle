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
#include <thread>
#include <vector>
#ifdef PADDLE_WITH_CUDA
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "hashtable.h"       // NOLINT
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "thrust/pair.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#endif
#include "heter_resource.h"  // NOLINT
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
struct CustomGradMerger {
  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__ __device__ T
  operator()(const T& a, const T& b) const {
    T out;
    out.slot = a.slot;
    out.show = a.show + b.show;
    out.clk = a.clk + b.clk;
    out.lr_g = a.lr_g + b.lr_g;
    for (int i = 0; i < MF_DIM; ++i) {
      out.mf_g[i] = a.mf_g[i] + b.mf_g[i];
    }
    return out;
  }
};

#elif defined(PADDLE_WITH_XPU)
struct CustomGradMerger {
  template <typename T>
  __forceinline__ __device__ T operator()(const T& a, const T& b) const {
    T out;
    out.slot = a.slot;
    out.show = a.show + b.show;
    out.clk = a.clk + b.clk;
    out.lr_g = a.lr_g + b.lr_g;
    for (int i = 0; i < MF_DIM; ++i) {
      out.mf_g[i] = a.mf_g[i] + b.mf_g[i];
    }
    return out;
  }
};
#endif

template <typename KeyType, typename ValType, typename GradType>
class HeterComm {
 public:
  HeterComm(size_t capacity, std::shared_ptr<HeterPsResource> resource);
  virtual ~HeterComm();
  HeterComm(const HeterComm&) = delete;
  HeterComm& operator=(const HeterComm&) = delete;

  void split_input_to_shard(KeyType* d_keys, int* d_idx_ptr, size_t len,
                            int* left, int* right, int gpu_num);
  void merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
                  int& uniq_len);  // NOLINT
  void pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len,
                size_t chunk_size, int stream_num);
  void dump();
  void show_one_table(int gpu_num);
  int get_index_by_devid(int devid);

  template <typename Sgd>
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                   Sgd& sgd);  // NOLINT


  int log2i(int x);

  template <typename DstPlace, typename SrcPlace, typename StreamType>
  void memory_copy(DstPlace dst_place, void* dst, SrcPlace src_place,
                   const void* src, size_t count, StreamType stream = 0);

#if defined(PADDLE_WITH_CUDA)
  template <typename Sgd>
  void push_sparse_multi_node(int num, KeyType* d_keys, GradType* d_grads,
                              size_t len, Sgd& sgd);  // NOLINT

  template <typename Sgd>
  void update_one_table(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                        Sgd& sgd);  // NOLINT

  int gather_one_node_grad(int num, KeyType* d_keys, GradType* d_grads,
                           int len);

  int gather_multi_node_grad(int num, KeyType* d_keys, GradType* d_grads,
                             int len);


  void set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                              const std::vector<ncclComm_t>& inter_comms,
                              int comm_size) {
    nccl_inner_comms_ = inner_comms;
    nccl_inter_comms_ = inter_comms;
    node_size_ = comm_size;
  }
#endif

  bool need_transfer(int send_id, int receive_id) {
    return ((send_id / 4 != receive_id / 4) && (send_id + 4) % 8 != receive_id);
  }

  // void dump_to_cpu(int index);

  void end_pass();

  int get_transfer_devid(int send_id) { return (send_id + 4) % 8; }


  void end_pass();

  struct Node {
    ppStream in_stream;
    ppStream out_stream;
    char* key_storage;
    char* val_storage;
    int sync;
    int key_bytes_len;
    int val_bytes_len;
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

  struct LocalStorage {
    LocalStorage() {}
    void init(int size, int dev_id) {
      place_ = platform::CUDAPlace(dev_id);
      alloc(size, true);
    }

    void alloc(int size, bool force = false) {
      if (force || size > all_keys_mem->size()) {
        all_keys_mem.reset();
        all_grads_mem.reset();
        all_keys_mem = memory::Alloc(place_, size * sizeof(KeyType));
        all_grads_mem = memory::Alloc(place_, size * sizeof(GradType));
        all_keys = reinterpret_cast<KeyType*>(all_keys_mem->ptr());
        all_grads = reinterpret_cast<GradType*>(all_grads_mem->ptr());
      }
      if (force || size > local_keys_mem->size()) {
        local_keys_mem.reset();
        local_grads_mem.reset();
        local_keys_mem = memory::Alloc(place_, size * sizeof(KeyType));
        local_grads_mem = memory::Alloc(place_, size * sizeof(GradType));
        local_keys = reinterpret_cast<KeyType*>(local_keys_mem->ptr());
        local_grads = reinterpret_cast<GradType*>(local_grads_mem->ptr());
      }
    }

    platform::CUDAPlace place_;

    std::shared_ptr<memory::Allocation> all_keys_mem;
    std::shared_ptr<memory::Allocation> all_grads_mem;

    KeyType* all_keys;
    GradType* all_grads;

    std::shared_ptr<memory::Allocation> local_keys_mem;
    std::shared_ptr<memory::Allocation> local_grads_mem;
    KeyType* local_keys;
    GradType* local_grads;
  };

  void init_path();

  template <typename StreamType>
  void sync_stream(const StreamType& stream) {
    if (stream >= 0) {
#if defined(PADDLE_WITH_CUDA)
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#elif defined(PADDLE_WITH_XPU)
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(stream));
#endif
    }
  }

  template <typename KeyT, typename ValueT, typename StreamType>
  void sort_pairs(void* d_temp_storage, size_t& temp_storage_bytes,  // NOLINT
                  const KeyT* d_keys_in, KeyT* d_keys_out,
                  const ValueT* d_values_in, ValueT* d_values_out,
                  int num_items, int begin_bit = 0,
                  int end_bit = sizeof(KeyT) * 8, StreamType stream = 0,
                  bool debug_synchronous = false);

  template <typename KeysInputIteratorT, typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT, typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT, typename ReductionOpT>
  void ReduceByKey(void* d_temp_storage, size_t& temp_storage_bytes,  // NOLINT
                   KeysInputIteratorT d_keys_in,
                   UniqueOutputIteratorT d_unique_out,
                   ValuesInputIteratorT d_values_in,
                   AggregatesOutputIteratorT d_aggregates_out,
                   NumRunsOutputIteratorT d_num_runs_out,
                   ReductionOpT reduction_op, int num_items,
                   cudaStream_t stream = 0, bool debug_synchronous = false);

  void create_storage(int start_index, int end_index, int keylen, int vallen);
  void destroy_storage(int start_index, int end_index);
  void walk_to_dest(int start_index, int gpu_num, int* h_left, int* h_right,
                    KeyType* src_key, GradType* src_val);
  void walk_to_src(int start_index, int gpu_num, int* h_left, int* h_right,
                   ValType* src_val);

 protected:
  using Table = HashTable<KeyType, ValType>;
  std::vector<Table*> tables_;
#endif
  std::shared_ptr<HeterPsResource> resource_;
  std::vector<std::vector<Path>> path_;
  float load_factor_{0.75};
  int block_size_{256};

 private:
  std::vector<LocalStorage> storage_;
  CustomGradMerger merger_;
  int topo_aware_{0};
  int feanum_{1800 * 2048};
  int multi_node_{0};
  int node_size_;

#if defined(PADDLE_WITH_CUDA)
  std::vector<ncclComm_t> nccl_inner_comms_;
  std::vector<ncclComm_t> nccl_inter_comms_;
#endif
  int node_size_;
#if defined(PADDLE_WITH_CUDA)
  std::vector<std::shared_ptr<cub::CachingDeviceAllocator>> allocators_;
#elif defined(PADDLE_WITH_XPU_KP)
  // std::vector<std::shared_ptr<>> allocators_;
#endif
};

}  // end namespace framework
}  // end namespace paddle

#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_inl.h"

#endif

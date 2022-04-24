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
#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "thrust/pair.h"
#elif defined(PADDLE_WITH_XPU_KP)
// #include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"
#include <xpu/runtime.h>
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#endif

#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

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

#if defined(PADDLE_WITH_CUDA)
  template <typename Sgd>
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                   Sgd& sgd);  // NOLINT
#elif defined(PADDLE_WITH_XPU_KP)
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len);
#endif

#if defined(PADDLE_WITH_XPU_KP)
  void set_sparse_sgd(const OptimizerConfig& optimizer_config);
  void set_embedx_sgd(const OptimizerConfig& optimizer_config);
#endif

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

    void alloc(size_t size, bool force = false) {
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

#if defined(PADDLE_WITH_CUDA)
    platform::CUDAPlace place_;

#elif defined(PADDLE_WITH_XPU_KP)
    platform::XPUPlace place_;
#endif
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

  void create_storage(int start_index, int end_index, int keylen, int vallen);
  void destroy_storage(int start_index, int end_index);
  void walk_to_dest(int start_index, int gpu_num, int* h_left, int* h_right,
                    KeyType* src_key, GradType* src_val);
  void walk_to_src(int start_index, int gpu_num, int* h_left, int* h_right,
                   ValType* src_val);

 protected:
  using Table = HashTable<KeyType, ValType>;
  std::vector<Table*> tables_;
  std::shared_ptr<HeterPsResource> resource_;
  std::vector<std::vector<Path>> path_;
  float load_factor_{0.75};
  int block_size_{256};
  std::unique_ptr<HeterCommKernel> heter_comm_kernel_;

 private:
  int topo_aware_{0};
  std::vector<LocalStorage> storage_;
  int feanum_{1800 * 2048};
  int multi_node_{0};
  int node_size_;

#if defined(PADDLE_WITH_CUDA)
  std::vector<ncclComm_t> nccl_inner_comms_;
  std::vector<ncclComm_t> nccl_inter_comms_;
  std::vector<std::shared_ptr<cub::CachingDeviceAllocator>> allocators_;
#endif
};

}  // end namespace framework
}  // end namespace paddle

#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_inl.h"

#endif

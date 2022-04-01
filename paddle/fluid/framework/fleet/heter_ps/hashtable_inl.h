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

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
template <typename value_type>
struct ReplaceOp {
  __host__ __device__ value_type operator()(value_type new_value,
                                            value_type old_value) {
    return new_value;
  }
};

template <typename Table>
__global__ void insert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals,
                              size_t len) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void insert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              size_t len, char* pool, int start_index) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    kv.first = keys[i];
    kv.second = (Table::mapped_type)(pool + (start_index + i) * 80);
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void search_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              typename Table::mapped_type* const vals,
                              size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      vals[i] = it->second;
    }
  }
}

template <typename Table>
__global__ void dy_mf_search_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    char* const vals, size_t len,
                                    size_t pull_feature_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);

    if (it != table->end()) {
      *(FeatureValue*)(vals + i * pull_feature_value_size) = *(it->second);
    }
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const GradType* const grads, size_t len,
                              Sgd sgd) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      sgd.update_value((it.getter())->second, grads[i]);
    }
  }
}

template <typename Table, typename Sgd>
__global__ void dy_mf_update_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    const char* const grads, size_t len,
                                    Sgd sgd, size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      FeaturePushValue* cur = (FeaturePushValue*)(grads + i * grad_value_size);
      sgd.dy_mf_update_value((it.getter())->second, *cur);
    } else {
      printf("yxf::push miss key: %d", keys[i]);
    }
  }
}
#elif defined(PADDLE_WITH_XPU)

template <typename KeyType, typename ValType, typename Table>
__global__ void insert_kernel(Table* table, const KeyType* const keys,
                              const ValType* const vals, size_t len) {
  int cid = core_id();
  int ncores = core_num();
  if (cid >= ncores) {
    return;
  }
  int thread_id = ncores * cluster_id() + cid;
  int nthreads = ncores * cluster_num();

  const int buf_size = 1024;
  __local__ KeyType local_keys[buf_size];
  __local__ ValType local_vals[buf_size];
  int len_per_loop = min(buf_size, roundup_div(len, nthreads));

  for (int i = thread_id * len_per_loop; i < len;
       i += nthreads * len_per_loop) {
    int read_len = min(len_per_loop, len - i);
    GM2LM(keys, local_keys, read_len * sizeof(KeyType));
    GM2LM(vals, local_vals, read_len * sizeof(ValType));
    for (int k = 0; k < read_len; k++) {
      auto status = table->insert(local_keys[k], local_vals[k]);
      assert(status != false && "error: insert fails: table is full");
    }
  }
}

// template <typename Table>
// __global__ void insert_kernel(Table* table,
//                              const typename Table::key_type* const keys,
//                              size_t len, char* pool, int start_index) {
//  ReplaceOp<typename Table::mapped_type> op;
//  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;
//
//  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//
//  if (i < len) {
//    kv.first = keys[i];
//    kv.second = (Table::mapped_type)(pool + (start_index + i) * 80);
//    auto it = table->insert(kv, op);
//    assert(it != table->end() && "error: insert fails: table is full");
//  }
// }

template <typename KeyType, typename ValType, typename Table>
__global__ void search_kernel(Table* table, const KeyType* const keys,
                              ValType* const vals, size_t len) {
  int cid = core_id();
  int ncores = core_num();
  if (cid >= ncores) {
    return;
  }
  int thread_id = ncores * cluster_id() + cid;
  int nthreads = ncores * cluster_num();

  const int buf_size = 1024;
  __local__ KeyType local_keys[buf_size];
  __local__ ValType local_vals[buf_size];

  int len_per_loop = min(buf_size, roundup_div(len, nthreads));
  for (int i = thread_id * len_per_loop; i < len;
       i += nthreads * len_per_loop) {
    int read_len = min(len_per_loop, len - i);
    GM2LM(keys, local_keys, read_len * sizeof(KeyType));
    for (int k = 0; k < read_len; k++) {
      ValType* val = table->find(local_keys[k]);
      if (val != NULL) {
        local_vals[k] = *val;
      }
    }
    LM2GM(local_vals, vals + i, read_len * sizeof(ValType));
  }
}

template <typename KeyType, typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table, const KeyType* const keys,
                              const GradType* const grads, size_t len,
                              Sgd* sgd) {
  int cid = core_id();
  int ncores = core_num();
  if (cid >= ncores) {
    return;
  }
  int thread_id = ncores * cluster_id() + cid;
  int nthreads = ncores * cluster_num();

  const int buf_size = 1024;
  __local__ KeyType local_keys[buf_size];
  __local__ GradType local_grads[buf_size];

  int len_per_loop = min(buf_size, roundup_div(len, nthreads));
  for (int i = thread_id * len_per_loop; i < len;
       i += nthreads * len_per_loop) {
    int read_len = min(len_per_loop, len - i);

    GM2LM(keys, local_keys, read_len * sizeof(KeyType));
    GM2LM(grads, local_grads, read_len * sizeof(GradType));

    for (int k = 0; k < read_len; k++) {
      ValType* val = table->find(local_keys[k]);
      if (val != NULL) {
        sgd->update_value(*val, grads[i]);
      }
    }
  }
}

#endif

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity) {
#if defined(PADDLE_WITH_CUDA)
  container_ = new TableContainer<KeyType, ValType>(capacity);
#elif defined(PADDLE_WITH_XPU)
  auto tmp_container = XPUCacheArray<KeyType, ValType>(capacity);
  xpu_malloc(reinterpret_cast<void**>(&container_),
             sizeof(XPUCacheArray<KeyType, ValType>));
  xpu_memcpy(container_, &tmp_container,
             sizeof(XPUCacheArray<KeyType, ValType>), XPU_HOST_TO_DEVICE);
#endif
  rwlock_.reset(new phi::RWLock);
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
#if defined(PADDLE_WITH_CUDA)
  delete container_;
#elif defined(PADDLE_WITH_XPU)
  xpu_free(container_)
#endif
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::show() {
  container_->print();
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, ValType* d_vals,
                                      size_t len, StreamType stream) {
  if (len == 0) {
    return;
  }
#if defined(PADDLE_WITH_CUDA)
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys,
                                                       d_vals, len);
#elif defined(PADDLE_WITH_XPU)
  search_kernel<<<4, 64, stream>>>(container_, d_keys, d_vals, len);
#endif
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, char* d_vals,
                                      size_t len, StreamType stream) {
  if (len == 0) {
    return;
  }
#if defined(PADDLE_WITH_CUDA)
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  dy_mf_search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len, pull_feature_value_size_);
#endif
}

template <typename KeyType, typename ValType, typename StreamType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys,
                                         const ValType* d_vals, size_t len,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
#if defined(PADDLE_WITH_CUDA)
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys,
                                                       d_vals, len);
#elif defined(PADDLE_WITH_XPU)
  insert_kernel<<<4, 64, stream>>>(container_, d_keys, d_vals, len);
#endif
}

template <typename KeyType, typename ValType, typename StreamType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys, size_t len,
                                         char* pool, size_t start_index,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
  if (pool == NULL) {
    return;
  }
#if defined(PADDLE_WITH_CUDA)
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, len,
                                                       pool, start_index);
#elif defined(PADDLE_WITH_XPU)
// insert_kernel<<<4, 64, stream>>>(container_, d_keys, len, pool, start_index);
#endif
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::dump_to_cpu(int devid, StreamType stream) {
#if defined(PADDLE_WITH_CUDA)
  container_->prefetch(cudaCpuDeviceId, stream);
  std::vector<std::thread> threads;
  size_t num = container_->size();
  KeyType unuse_key = std::numeric_limits<KeyType>::max();
  thrust::pair<KeyType, ValType>* kv = container_->data();

  int thread_num = 8;
  int len_per_thread = num / thread_num;
  int remain = num % thread_num;
  int begin = 0;

  auto dump_func = [unuse_key, kv](int left, int right) {
    for (int i = left; i < right; i++) {
      if (kv[i].first == unuse_key) {
        continue;
      }
      ValType& gpu_val = kv[i].second;
#ifdef PADDLE_WITH_PSLIB
      auto* downpour_value =
          (paddle::ps::DownpourFixedFeatureValue*)(gpu_val.cpu_ptr);
      int downpour_value_size = downpour_value->size();
      if (gpu_val.mf_size > 0 && downpour_value_size == 7) {
        downpour_value->resize(gpu_val.mf_size + downpour_value_size);
      }
      float* cpu_val = downpour_value->data();
      // cpu_val[0] = 0;
      cpu_val[1] = gpu_val.delta_score;
      cpu_val[2] = gpu_val.show;
      cpu_val[3] = gpu_val.clk;
      cpu_val[4] = gpu_val.lr;
      cpu_val[5] = gpu_val.lr_g2sum;
      cpu_val[6] = gpu_val.slot;
      if (gpu_val.mf_size > 0) {
        for (int x = 0; x < gpu_val.mf_size; x++) {
          cpu_val[x + 7] = gpu_val.mf[x];
        }
      }
#endif
#ifdef PADDLE_WITH_PSCORE
      auto* downpour_value =
          (paddle::distributed::FixedFeatureValue*)(gpu_val.cpu_ptr);
      int downpour_value_size = downpour_value->size();
      if (gpu_val.mf_size > 0 && downpour_value_size == 7) {
        downpour_value->resize(gpu_val.mf_size + downpour_value_size);
      }
      float* cpu_val = downpour_value->data();
      // cpu_val[0] = 0;
      cpu_val[2] = gpu_val.delta_score;
      cpu_val[3] = gpu_val.show;
      cpu_val[4] = gpu_val.clk;
      cpu_val[5] = gpu_val.lr;
      cpu_val[6] = gpu_val.lr_g2sum;
      cpu_val[0] = gpu_val.slot;
      if (gpu_val.mf_size > 0) {
        for (int x = 0; x < gpu_val.mf_size; x++) {
          cpu_val[x + 7] = gpu_val.mf[x];
        }
      }
#endif
    }
  };

  for (int i = 0; i < thread_num; i++) {
    threads.push_back(std::thread(
        dump_func, begin, begin + len_per_thread + (i < remain ? 1 : 0)));
    begin += len_per_thread + (i < remain ? 1 : 0);
  }
  for (std::thread& t : threads) {
    t.join();
  }

#endif

  // container_->prefetch(devid, stream);
}

template <typename KeyType, typename ValType>
template <typename GradType, typename Sgd, typename StreamType>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const GradType* d_grads, size_t len,
                                         Sgd sgd, StreamType stream) {
  if (len == 0) {
    return;
  }
#if defined(PADDLE_WITH_CUDA)
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys,
                                                       d_grads, len, sgd);
#elif defined(PADDLE_WITH_XPU)
  // move sgd to global memory
  Sgd* xpu_sgd;
  xpu_malloc(reinterpret_cast<void**>(&xpu_sgd), sizeof(Sgd));
  update_kernel<<<4, 64, stream>>>(container_, d_keys, d_grads, len, xpu_sgd);
#endif
}

template <typename KeyType, typename ValType>
template <typename Sgd, typename StreamType>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const char* d_grads, size_t len,
                                         Sgd sgd, StreamType stream) {
  if (len == 0) {
    return;
  }
#if defined(PADDLE_WITH_CUDA)
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  dy_mf_update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_grads, len, sgd, push_grad_value_size_);
#endif
}

}  // end namespace framework
}  // end namespace paddle
#endif

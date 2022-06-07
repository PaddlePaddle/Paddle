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
#include "paddle/fluid/framework/heter_util.h"

namespace paddle {
namespace framework {

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
                              size_t len, char* pool, size_t feature_value_size,
                              int start_index) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    kv.first = keys[i];
    uint64_t offset = uint64_t(start_index + i) * feature_value_size;
    kv.second = (Table::mapped_type)(pool + offset);
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
                                    char* vals, size_t len,
                                    size_t pull_feature_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      uint64_t offset = i * pull_feature_value_size;
      FeatureValue* cur = (FeatureValue*)(vals + offset);
      FeatureValue& input = *(FeatureValue*)(it->second);
      cur->slot    = input.slot;
      cur->show    = input.show;
      cur->clk     = input.clk;
      cur->mf_dim = input.mf_dim;
      cur->lr = input.lr;
      cur->mf_size = input.mf_size;
      cur->cpu_ptr = input.cpu_ptr;
      cur->delta_score = input.delta_score;
      cur->lr_g2sum = input.lr_g2sum;
      for(int j = 0; j < cur->mf_dim + 1; ++j) {
         cur->mf[j] = input.mf[j];
      }
    } else {
      if (keys[i] != 0) printf("pull miss key: %d",keys[i]);
      FeatureValue* cur = (FeatureValue*)(vals + i * pull_feature_value_size);
      cur->delta_score = 0;
      cur->show = 0;
      cur->clk = 0;
      cur->slot = -1;
      cur->lr = 0;
      cur->lr_g2sum = 0;
      cur->mf_size = 0;
      cur->mf_dim = 8;
      cur->cpu_ptr;
      for (int j = 0; j < cur->mf_dim + 1; j++) {
        cur->mf[j] = 0;
      }
      
    }
  }
}

__global__ void  curand_init_kernel(curandState* p_value, int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    curand_init(clock64(), i, 0, p_value + i);
  }
}

class CuRandState {
public:
  CuRandState() = default;
  CuRandState(const CuRandState&) = delete;
  CuRandState(CuRandState&&) = delete;
  ~CuRandState() {
    CHECK(cudaFree(states_) == cudaSuccess);
  }
  curandState* get(size_t size, gpuStream_t stream) {
    if (size > size_) {
      size_t new_size = size * 2;
      curandState* new_state = nullptr;
      CHECK(cudaMalloc(reinterpret_cast<void**>(&new_state), new_size * sizeof(curandState)) == cudaSuccess);
      if (size_ > 0) {
        CHECK(cudaMemcpyAsync(new_state, states_,
                      size_ * sizeof(curandState), cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
      }
      int init_len = new_size - size_;
      const int BLOCK_SIZE_{256};
      const int init_kernel_grid = (init_len - 1) / BLOCK_SIZE_ + 1;
      curand_init_kernel<<<init_kernel_grid, BLOCK_SIZE_, 0, stream>>>(new_state + size_, init_len);
      if (size_ != 0) {
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(cudaFree(states_) == cudaSuccess);
      }
      states_ = new_state;
      size_ = new_size;
    }
    return states_;
  }

  static HeterObjectPool<CuRandState>& pool() {
    static HeterObjectPool<CuRandState> p;
    return p;
  }

  static std::shared_ptr<CuRandState> get() {
    return pool().Get();
  }

  static void CUDART_CB pushback_cu_rand_state(void *data) {
    auto state = static_cast<std::shared_ptr<CuRandState>*>(data);
    pool().Push(std::move(*state));
    delete state;
  }

  static void push(std::shared_ptr<CuRandState> state, gpuStream_t stream) {
    CHECK(cudaLaunchHostFunc(stream, pushback_cu_rand_state,
      new std::shared_ptr<CuRandState>(std::move(state))) == cudaSuccess); 
  }
private:
  size_t size_ = 0;
  curandState* states_ = nullptr;
};

template <typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const GradType* const grads, curandState* p_state, size_t len,
                              Sgd sgd) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      sgd.update_value((it.getter())->second, grads[i], p_state[i]);
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
      if(keys[i] != 0) printf("push miss key: %d", keys[i]);
    }
  }
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity) {
  container_ = new TableContainer<KeyType, ValType>(capacity);
  rwlock_.reset(new phi::RWLock);
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
  delete container_;
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::show() {
  container_->print();
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, ValType* d_vals,
                                      size_t len, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys,
                                                       d_vals, len);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, char* d_vals,
                                      size_t len, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  dy_mf_search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len, pull_feature_value_size_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys,
                                         const ValType* d_vals, size_t len,
                                         gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys,
                                                       d_vals, len);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys, size_t len,
                                         char* pool,
                                         size_t feature_value_size,
                                         size_t start_index,
                                         gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  if (pool == NULL) {
    return;
  }
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, len,
                                                       pool, feature_value_size, start_index);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::dump_to_cpu(int devid, cudaStream_t stream) {
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

  // container_->prefetch(devid, stream);
}

template <typename KeyType, typename ValType>
template <typename GradType, typename Sgd>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const GradType* d_grads, size_t len,
                                         Sgd sgd, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  auto state = CuRandState::get();
  auto d_state = state->get(len, stream);
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_grads, d_state, len, sgd);
  CuRandState::push(state, stream);
}

template <typename KeyType, typename ValType>
template <typename Sgd>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const char* d_grads, size_t len,
                                         Sgd sgd, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;

  dy_mf_update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_grads, len, sgd, push_grad_value_size_);
}

}  // end namespace framework
}  // end namespace paddle
#endif

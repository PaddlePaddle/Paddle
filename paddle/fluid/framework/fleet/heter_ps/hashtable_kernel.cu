/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <thread>

#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"

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
                              size_t len,
                              char* pool,
                              size_t feature_value_size,
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

template <typename Table, typename GPUAccessor>
__global__ void dy_mf_search_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    char* vals,
                                    size_t len,
                                    size_t pull_feature_value_size,
                                    GPUAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // return;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      uint64_t offset = i * pull_feature_value_size;
      float* cur = (float*)(vals + offset);
      float* input = it->second;
      gpu_accessor.PullValueFill(cur, input);
    }
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table,
                              const OptimizerConfig& optimizer_config,
                              const typename Table::key_type* const keys,
                              const GradType* const grads,
                              size_t len,
                              Sgd sgd) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      sgd.update_value(optimizer_config, (it.getter())->second, grads[i]);
    }
  }
}

template <typename Table, typename Sgd>
__global__ void dy_mf_update_kernel(Table* table,
                                    const OptimizerConfig& optimizer_config,
                                    const typename Table::key_type* const keys,
                                    const char* const grads,
                                    size_t len,
                                    Sgd sgd,
                                    size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      float* cur = (float*)(grads + i * grad_value_size);
      sgd.dy_mf_update_value(optimizer_config, (it.getter())->second, cur);
    } else {
      printf("warning: push miss key: %lu", keys[i]);
    }
  }
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity) {
  container_ = new TableContainer<KeyType, ValType>(capacity);
  CUDA_RT_CALL(
      cudaMalloc((void**)&device_optimizer_config_, sizeof(OptimizerConfig)));
  CUDA_RT_CALL(cudaMemcpy((void*)device_optimizer_config_,
                          &host_optimizer_config_,
                          sizeof(OptimizerConfig),
                          cudaMemcpyHostToDevice));
  rwlock_.reset(new phi::RWLock);
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
  delete container_;
  cudaFree(device_optimizer_config_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::set_sparse_sgd(
    const OptimizerConfig& optimizer_config) {
  host_optimizer_config_.set_sparse_sgd(optimizer_config);
  cudaMemcpy((void*)device_optimizer_config_,
             &host_optimizer_config_,
             sizeof(OptimizerConfig),
             cudaMemcpyHostToDevice);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::set_embedx_sgd(
    const OptimizerConfig& optimizer_config) {
  host_optimizer_config_.set_embedx_sgd(optimizer_config);
  cudaMemcpy((void*)device_optimizer_config_,
             &host_optimizer_config_,
             sizeof(OptimizerConfig),
             cudaMemcpyHostToDevice);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::show() {
  container_->print();
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys,
                                      ValType* d_vals,
                                      size_t len,
                                      StreamType stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
template <typename StreamType, typename GPUAccessor>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys,
                                      char* d_vals,
                                      size_t len,
                                      StreamType stream,
                                      GPUAccessor& fv_accessor) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  dy_mf_search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len, pull_feature_value_size_, fv_accessor);
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys,
                                         const ValType* d_vals,
                                         size_t len,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys,
                                         size_t len,
                                         char* pool,
                                         size_t feature_value_size,
                                         size_t start_index,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
  if (pool == NULL) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, len, pool, feature_value_size, start_index);
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::dump_to_cpu(int devid, StreamType stream) {
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
template <typename Sgd, typename StreamType>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const float* d_grads,
                                         size_t len,
                                         Sgd sgd,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, *device_optimizer_config_, d_keys, d_grads, len, sgd);
}

template <typename KeyType, typename ValType>
template <typename Sgd, typename StreamType>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const char* d_grads,
                                         size_t len,
                                         Sgd sgd,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  dy_mf_update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_,
      *device_optimizer_config_,
      d_keys,
      d_grads,
      len,
      sgd,
      push_grad_value_size_);
}

template class HashTable<unsigned long, float>;
template class HashTable<unsigned long, float*>;
template class HashTable<long, int>;
template class HashTable<unsigned long, int>;
template class HashTable<unsigned long, unsigned long>;
template class HashTable<unsigned long, unsigned long*>;
template class HashTable<unsigned long, long>;
template class HashTable<unsigned long, long*>;
template class HashTable<long, long>;
template class HashTable<long, unsigned long>;
template class HashTable<long, unsigned int>;

template void HashTable<unsigned long, float>::get<cudaStream_t>(
    const unsigned long* d_keys,
    float* d_vals,
    size_t len,
    cudaStream_t stream);

template void
HashTable<unsigned long, float*>::get<cudaStream_t, CommonFeatureValueAccessor>(
    const unsigned long* d_keys,
    char* d_vals,
    size_t len,
    cudaStream_t stream,
    CommonFeatureValueAccessor& fv_accessor);

template void HashTable<long, int>::get<cudaStream_t>(const long* d_keys,
                                                      int* d_vals,
                                                      size_t len,
                                                      cudaStream_t stream);

template void HashTable<unsigned long, int>::get<cudaStream_t>(
    const unsigned long* d_keys, int* d_vals, size_t len, cudaStream_t stream);
template void HashTable<unsigned long, unsigned long>::get<cudaStream_t>(
    const unsigned long* d_keys,
    unsigned long* d_vals,
    size_t len,
    cudaStream_t stream);
template void HashTable<unsigned long, long>::get<cudaStream_t>(
    const unsigned long* d_keys, long* d_vals, size_t len, cudaStream_t stream);
template void HashTable<long, unsigned long>::get<cudaStream_t>(
    const long* d_keys, unsigned long* d_vals, size_t len, cudaStream_t stream);
template void HashTable<long, long>::get<cudaStream_t>(const long* d_keys,
                                                       long* d_vals,
                                                       size_t len,
                                                       cudaStream_t stream);
template void HashTable<long, unsigned int>::get<cudaStream_t>(
    const long* d_keys, unsigned int* d_vals, size_t len, cudaStream_t stream);
// template void
// HashTable<unsigned long, paddle::framework::FeatureValue>::get<cudaStream_t>(
//    const unsigned long* d_keys, char* d_vals, size_t len, cudaStream_t
//    stream);

template void HashTable<unsigned long, float>::insert<cudaStream_t>(
    const unsigned long* d_keys,
    const float* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<unsigned long, float*>::insert<cudaStream_t>(
    const unsigned long* d_keys,
    size_t len,
    char* pool,
    size_t feature_value_size,
    size_t start_index,
    cudaStream_t stream);

template void HashTable<long, int>::insert<cudaStream_t>(const long* d_keys,
                                                         const int* d_vals,
                                                         size_t len,
                                                         cudaStream_t stream);
template void HashTable<long, long>::insert<cudaStream_t>(const long* d_keys,
                                                          const long* d_vals,
                                                          size_t len,
                                                          cudaStream_t stream);

template void HashTable<unsigned long, int>::insert<cudaStream_t>(
    const unsigned long* d_keys,
    const int* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<unsigned long, long>::insert<cudaStream_t>(
    const unsigned long* d_keys,
    const long* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<long, unsigned long>::insert<cudaStream_t>(
    const long* d_keys,
    const unsigned long* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<long, unsigned int>::insert<cudaStream_t>(
    const long* d_keys,
    const unsigned int* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<unsigned long, unsigned long>::insert<cudaStream_t>(
    const unsigned long* d_keys,
    const unsigned long* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<unsigned long, float*>::dump_to_cpu<cudaStream_t>(
    int devid, cudaStream_t stream);

template void HashTable<unsigned long, float*>::update<
    SparseAdagradOptimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const unsigned long* d_keys,
                  const char* d_grads,
                  size_t len,
                  SparseAdagradOptimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);
template void HashTable<unsigned long, float*>::update<
    SparseAdamOptimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const unsigned long* d_keys,
                  const char* d_grads,
                  size_t len,
                  SparseAdamOptimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);
template void HashTable<unsigned long, float*>::update<
    SparseAdamSharedOptimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const unsigned long* d_keys,
                  const char* d_grads,
                  size_t len,
                  SparseAdamSharedOptimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);

// template void HashTable<unsigned long,
// paddle::framework::FeatureValue>::update<
//    Optimizer<paddle::framework::FeatureValue,
//              paddle::framework::FeaturePushValue>,
//    cudaStream_t>(const unsigned long* d_keys, const char* d_grads, size_t
//    len,
//                  Optimizer<paddle::framework::FeatureValue,
//                            paddle::framework::FeaturePushValue>
//                      sgd,
//                  cudaStream_t stream);

#endif
}  // end namespace framework
}  // end namespace paddle
#endif

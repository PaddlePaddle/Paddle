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
                              size_t len,
                              int dft_val,
                              uint64_t* global_num) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  __shared__ uint64_t local_num;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len) {
    kv.first = keys[i];
    kv.second = dft_val;  // fake value
    auto it = table->insert(kv, op, &local_num);
    assert(it != table->end() && "error: insert fails: table is full");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(global_num, local_num);
  }
}

template <typename Table>
__global__ void insert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals,
                              size_t len,
                              uint64_t* global_num) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  __shared__ uint64_t local_num;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->insert(kv, op, &local_num);
    assert(it != table->end() && "error: insert fails: table is full");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(global_num, local_num);
  }
}

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
    PADDLE_ENFORCE(it != table->end(), "error: insert fails: table is full");
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
__global__ void search_ranks_kernel(Table* table,
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
__global__ void dy_mf_search_kernel_fill(
    Table* table,
    const typename Table::key_type* const keys,
    char* vals,
    size_t len,
    size_t pull_feature_value_size,
    GPUAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      uint64_t offset = i * pull_feature_value_size;
      float* cur = reinterpret_cast<float*>(vals + offset);
      float* input = it->second;
      gpu_accessor.PullValueFill(cur, input);
    } else {
      float* cur = reinterpret_cast<float*>(&vals[i * pull_feature_value_size]);
      gpu_accessor.PullZeroValue(cur);
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
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      uint64_t offset = i * pull_feature_value_size;
      float* cur = reinterpret_cast<float*>(vals + offset);
      float* input = it->second;
      gpu_accessor.PullValueFill(cur, input);
    } else {
      PADDLE_ENFORCE(false, "warning: pull miss key: %lu", keys[i]);
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
      const float* cur =
          reinterpret_cast<const float*>(grads + i * grad_value_size);
      sgd.dy_mf_update_value(optimizer_config, (it.getter())->second, cur);
    } else {
      PADDLE_ENFORCE(false, "warning: push miss key: %lu", keys[i]);
    }
  }
}

template <typename Table>
__global__ void get_keys_kernel(Table* table,
                                typename Table::key_type* d_out,
                                uint64_t* global_cursor,
                                uint64_t unused_key) {
  extern __shared__ typename Table::key_type local_key[];
  __shared__ uint64_t local_num;
  __shared__ uint64_t global_num;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  uint64_t len = table->size();
  if (idx < len) {
    typename Table::value_type val = *(table->data() + idx);
    if (val.first != unused_key) {
      uint64_t dst = atomicAdd(&local_num, 1);
      local_key[dst] = val.first;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(global_cursor, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    d_out[global_num + threadIdx.x] = local_key[threadIdx.x];
  }
}

template <typename Table>
__global__ void get_key_values_kernel(Table* table,
                                      typename Table::key_type* d_keys,
                                      typename Table::mapped_type* d_vals,
                                      uint64_t* global_cursor,
                                      uint64_t unused_key) {
  __shared__ typename Table::key_type local_key[256];
  // __shared__ typename Table::mapped_type local_val[256];
  __shared__ uint8_t local_val[256];
  __shared__ uint64_t local_num;
  __shared__ uint64_t global_num;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  uint64_t len = table->size();
  if (idx < len) {
    typename Table::value_type val = *(table->data() + idx);
    if (val.first != unused_key) {
      uint64_t dst = atomicAdd(&local_num, 1);
      local_key[dst] = val.first;
      local_val[dst] = val.second;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(global_cursor, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    d_keys[global_num + threadIdx.x] = local_key[threadIdx.x];
    d_vals[global_num + threadIdx.x] = local_val[threadIdx.x];
  }
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity, cudaStream_t stream) {
  stream_ = stream;
  container_ = new TableContainer<KeyType, ValType>(capacity, stream);
  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&device_optimizer_config_),
                          sizeof(OptimizerConfig)));
  CUDA_RT_CALL(
      cudaMemcpyAsync(reinterpret_cast<void*>(device_optimizer_config_),
                      &host_optimizer_config_,
                      sizeof(OptimizerConfig),
                      cudaMemcpyHostToDevice,
                      stream));
  cudaStreamSynchronize(stream);
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
  cudaMemcpyAsync(device_optimizer_config_,
                  &host_optimizer_config_,
                  sizeof(OptimizerConfig),
                  cudaMemcpyHostToDevice,
                  stream_);
  cudaStreamSynchronize(stream_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::set_embedx_sgd(
    const OptimizerConfig& optimizer_config) {
  host_optimizer_config_.set_embedx_sgd(optimizer_config);
  cudaMemcpyAsync(device_optimizer_config_,
                  &host_optimizer_config_,
                  sizeof(OptimizerConfig),
                  cudaMemcpyHostToDevice,
                  stream_);
  cudaStreamSynchronize(stream_);
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
                                      const GPUAccessor& fv_accessor) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  // infer need zero fill
  if (infer_mode_) {
    dy_mf_search_kernel_fill<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        container_, d_keys, d_vals, len, pull_feature_value_size_, fv_accessor);
  } else {
    dy_mf_search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        container_, d_keys, d_vals, len, pull_feature_value_size_, fv_accessor);
  }
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::get_ranks(const KeyType* d_keys,
                                            ValType* d_vals,
                                            size_t len,
                                            StreamType stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_ranks_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys,
                                         size_t len,
                                         uint64_t* global_num,
                                         int dft_val,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, len, dft_val, global_num);
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys,
                                         const ValType* d_vals,
                                         size_t len,
                                         uint64_t* global_num,
                                         StreamType stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len, global_num);
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
void HashTable<KeyType, ValType>::get_keys(KeyType* d_out,
                                           uint64_t* global_cursor,
                                           StreamType stream) {
  size_t len = container_->size();
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  KeyType unuse_key = std::numeric_limits<KeyType>::max();
  size_t shared_mem_size = sizeof(KeyType) * BLOCK_SIZE_;
  get_keys_kernel<<<grid_size, BLOCK_SIZE_, shared_mem_size, stream>>>(
      container_, d_out, global_cursor, unuse_key);
}

template <typename KeyType, typename ValType>
template <typename StreamType>
void HashTable<KeyType, ValType>::get_key_values(KeyType* d_keys,
                                                 ValType* d_vals,
                                                 uint64_t* global_cursor,
                                                 StreamType stream) {
  const int BLOCK_SIZE = 128;
  size_t len = container_->size();
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  KeyType unuse_key = std::numeric_limits<KeyType>::max();
  size_t shared_mem_size = (sizeof(KeyType) + sizeof(ValType)) * BLOCK_SIZE_;
  get_key_values_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, global_cursor, unuse_key);
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

template class HashTable<uint64_t, float>;
template class HashTable<uint64_t, float*>;
template class HashTable<int64_t, int>;
template class HashTable<uint64_t, int>;
template class HashTable<uint64_t, uint64_t>;
template class HashTable<uint64_t, uint64_t*>;
template class HashTable<uint64_t, int64_t>;
template class HashTable<uint64_t, int64_t*>;
template class HashTable<uint64_t, uint32_t*>;
template class HashTable<uint64_t, uint32_t>;
template class HashTable<int64_t, int64_t>;
template class HashTable<int64_t, uint64_t>;
template class HashTable<int64_t, unsigned int>;

template void HashTable<uint64_t, float>::get<cudaStream_t>(
    const uint64_t* d_keys, float* d_vals, size_t len, cudaStream_t stream);

template void
HashTable<uint64_t, float*>::get<cudaStream_t, CommonFeatureValueAccessor>(
    const uint64_t* d_keys,
    char* d_vals,
    size_t len,
    cudaStream_t stream,
    const CommonFeatureValueAccessor& fv_accessor);

template void HashTable<int64_t, int>::get<cudaStream_t>(const int64_t* d_keys,
                                                         int* d_vals,
                                                         size_t len,
                                                         cudaStream_t stream);

template void HashTable<uint64_t, int>::get<cudaStream_t>(
    const uint64_t* d_keys, int* d_vals, size_t len, cudaStream_t stream);
template void HashTable<uint64_t, unsigned int>::get<cudaStream_t>(
    const uint64_t* d_keys,
    unsigned int* d_vals,
    size_t len,
    cudaStream_t stream);
template void HashTable<uint64_t, uint64_t>::get<cudaStream_t>(
    const uint64_t* d_keys, uint64_t* d_vals, size_t len, cudaStream_t stream);
template void HashTable<uint64_t, int64_t>::get<cudaStream_t>(
    const uint64_t* d_keys, int64_t* d_vals, size_t len, cudaStream_t stream);
template void HashTable<int64_t, uint64_t>::get<cudaStream_t>(
    const int64_t* d_keys, uint64_t* d_vals, size_t len, cudaStream_t stream);
template void HashTable<int64_t, int64_t>::get<cudaStream_t>(
    const int64_t* d_keys, int64_t* d_vals, size_t len, cudaStream_t stream);
template void HashTable<int64_t, unsigned int>::get<cudaStream_t>(
    const int64_t* d_keys,
    unsigned int* d_vals,
    size_t len,
    cudaStream_t stream);
template void HashTable<uint64_t, unsigned int>::get_ranks<cudaStream_t>(
    const uint64_t* d_keys,
    unsigned int* d_vals,
    size_t len,
    cudaStream_t stream);
// template void
// HashTable<uint64_t, paddle::framework::FeatureValue>::get<cudaStream_t>(
//    const uint64_t* d_keys, char* d_vals, size_t len, cudaStream_t
//    stream);
template void HashTable<uint64_t, float>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    const float* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<uint64_t, float*>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    size_t len,
    char* pool,
    size_t feature_value_size,
    size_t start_index,
    cudaStream_t stream);

template void HashTable<int64_t, int>::insert<cudaStream_t>(
    const int64_t* d_keys, const int* d_vals, size_t len, cudaStream_t stream);

template void HashTable<int64_t, int64_t>::insert<cudaStream_t>(
    const int64_t* d_keys,
    const int64_t* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<uint64_t, int>::insert<cudaStream_t>(
    const uint64_t* d_keys, const int* d_vals, size_t len, cudaStream_t stream);

template void HashTable<uint64_t, int64_t>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    const int64_t* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<int64_t, uint64_t>::insert<cudaStream_t>(
    const int64_t* d_keys,
    const uint64_t* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<int64_t, unsigned int>::insert<cudaStream_t>(
    const int64_t* d_keys,
    const unsigned int* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<uint64_t, uint32_t>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    const uint32_t* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<uint64_t, uint64_t>::get_keys<cudaStream_t>(
    uint64_t* d_out, uint64_t* global_cursor, cudaStream_t stream);

template void HashTable<uint64_t, uint32_t>::get_keys<cudaStream_t>(
    uint64_t* d_out, uint64_t* global_cursor, cudaStream_t stream);

template void HashTable<uint64_t, uint32_t>::get_key_values<cudaStream_t>(
    uint64_t* d_keys,
    uint32_t* d_vals,
    uint64_t* global_cursor,
    cudaStream_t stream);

template void HashTable<uint64_t, uint64_t>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    uint64_t len,
    uint64_t* global_num,
    int dft_val,
    cudaStream_t stream);

template void HashTable<uint64_t, uint32_t>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    uint64_t len,
    uint64_t* global_num,
    int dft_val,
    cudaStream_t stream);

template void HashTable<uint64_t, uint32_t>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    const uint32_t* d_values,
    uint64_t len,
    uint64_t* global_num,
    cudaStream_t stream);

template void HashTable<uint64_t, uint64_t>::insert<cudaStream_t>(
    const uint64_t* d_keys,
    const uint64_t* d_vals,
    size_t len,
    cudaStream_t stream);

template void HashTable<uint64_t, float*>::dump_to_cpu<cudaStream_t>(
    int devid, cudaStream_t stream);

template void HashTable<uint64_t, float*>::update<
    SparseAdagradOptimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const uint64_t* d_keys,
                  const char* d_grads,
                  size_t len,
                  SparseAdagradOptimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);

template void HashTable<uint64_t, float*>::update<
    SparseAdagradV2Optimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const uint64_t* d_keys,
                  const char* d_grads,
                  size_t len,
                  SparseAdagradV2Optimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);

template void HashTable<uint64_t, float*>::update<
    StdAdagradOptimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const uint64_t* d_keys,
                  const char* d_grads,
                  size_t len,
                  StdAdagradOptimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);
template void HashTable<uint64_t, float*>::update<
    SparseAdamOptimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const uint64_t* d_keys,
                  const char* d_grads,
                  size_t len,
                  SparseAdamOptimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);
template void HashTable<uint64_t, float*>::update<
    SparseAdamSharedOptimizer<CommonFeatureValueAccessor>,
    cudaStream_t>(const uint64_t* d_keys,
                  const char* d_grads,
                  size_t len,
                  SparseAdamSharedOptimizer<CommonFeatureValueAccessor> sgd,
                  cudaStream_t stream);

// template void HashTable<uint64_t,
// paddle::framework::FeatureValue>::update<
//    Optimizer<paddle::framework::FeatureValue,
//              paddle::framework::FeaturePushValue>,
//    cudaStream_t>(const uint64_t* d_keys, const char* d_grads, size_t
//    len,
//                  Optimizer<paddle::framework::FeatureValue,
//                            paddle::framework::FeaturePushValue>
//                      sgd,
//                  cudaStream_t stream);

#endif
}  // namespace framework
}  // namespace paddle
#endif

/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/fleet/heter_ps/cudf/concurrent_unordered_map.cuh.h"
#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {

class MemoryPool {
 public:
  MemoryPool(size_t capacity, size_t block_size) : _capacity(capacity), _block_size(block_size) {
    _mem = (char*)malloc(block_size * _capacity);
  }
  MemoryPool() {
    free(_mem);
  }
  size_t block_size() {
    return _block_size;
  }
  char *mem() {
    return _mem;
  }
  
  size_t capacity() {
    return _capacity;
  }
  void* mem_address(const uint32_t &idx) {
    return (void*)&_mem[(idx - 1) * _block_size];
  }
 private:
  char* _mem = NULL;
  size_t _capacity;
  size_t _block_size;
}

class HBMMemoryPool : public managed {
 public:
  HBMMemoryPool(size_t capacity, size_t block_size) : _capacity(capacity), _block_size(block_size) {
    cudaMalloc(&_mem, (block_size * capacity / 8 + 1) * 8);
    cudaMemset(_mem, 0, block_size * capacity);
  }
  HBMMemoryPool(size_t capacity, size_t block_size, *MemoryPool mem_pool, int stream_num) {
    cudaMalloc(&_mem, (block_size * capacity / 8 + 1) * 8);
    gpuStream_t streams[stream_num];
    int cur_len = 0;
    int cur_stream = 0;
    void* start_mem_pool = (void*)mem_pool.mem();

    while (cur_len < len) {
      cur_stream = cur_stream % stream_num;
      int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaMemcpyAsync(_mem + cur_len * _block_size, mem_pool->mem_address(cur_len),
                          _block_size * tmp_len, cudaMemcpyHostToDevice,
                          streams[cur_stream]));
      cur_stream += 1;
      cur_len += tmp_len;
    }

  }
    
  ~HBMMemoryPool() {
    cudaFree(_mem);
  }
 
  size_t block_size() {
    return _block_size;
  }

  void clear(void) {
    cudaMemset(_mem, 0, _block_size * _capacity);
  }
  
  void reset(size_t capacity) {
    cudaFree(_mem);
    cudaMalloc(&_mem, (block_size * capacity / 8 + 1) * 8);
    cudaMemset(_mem, 0, block_size * capacity);
  }

  char *mem() {
    return _mem;
  }
  
  size_t capacity() {
    return _capacity;
  }
  __forceinline__
  __device__ void* mem_address(const uint32_t &idx) {
    return (void*)&_mem[(idx - 1) * _block_size];
  }
 private:
  char* _mem = NULL;
  size_t _capacity;
  size_t _block_size;
};


}  // end namespace framework
}  // end namespace paddle
#endif
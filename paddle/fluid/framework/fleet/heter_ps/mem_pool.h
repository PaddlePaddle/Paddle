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

#ifdef PADDLE_WITH_HETERPS
#include <iostream>
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/fleet/heter_ps/cudf/managed.cuh"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"

namespace paddle {
namespace framework {

class MemoryPool {
 public:
  MemoryPool(size_t capacity, size_t block_size)
      : capacity_(capacity), block_size_(block_size) {
    VLOG(3) << "mem_pool init with block_size: " << block_size
            << " capacity: " << capacity;
    mem_ = reinterpret_cast<char*>(malloc(block_size * capacity_));
  }
  ~MemoryPool() {
    VLOG(3) << "mem pool delete";
    free(mem_);
  }
  size_t block_size() { return block_size_; }
  char* mem() { return mem_; }

  size_t capacity() { return capacity_; }
  size_t byte_size() { return capacity_ * block_size_; }
  void* mem_address(const uint32_t& idx) { return &mem_[(idx)*block_size_]; }

 private:
  char* mem_ = NULL;
  size_t capacity_;
  size_t block_size_;
};

// Derived from managed, alloced managed hbm
class HBMMemoryPool : public managed {
 public:
  HBMMemoryPool(size_t capacity, size_t block_size)
      : capacity_(capacity), block_size_(block_size) {}
  explicit HBMMemoryPool(MemoryPool* mem_pool) {
    capacity_ = mem_pool->capacity();
    block_size_ = mem_pool->block_size();
    VLOG(3) << "hbm memory pool with capacity" << capacity_
            << " bs: " << block_size_;
    CUDA_CHECK(cudaMalloc(&mem_, block_size_ * capacity_));
    CUDA_CHECK(cudaMemcpy(
        mem_, mem_pool->mem(), mem_pool->byte_size(), cudaMemcpyHostToDevice));
  }

  ~HBMMemoryPool() {
    VLOG(3) << "delete hbm memory pool";
    cudaFree(mem_);
  }

  size_t block_size() { return block_size_; }

  void clear(void) { cudaMemset(mem_, 0, block_size_ * capacity_); }

  void reset(size_t capacity) {
    cudaFree(mem_);
    mem_ = NULL;
    capacity_ = capacity;
    CUDA_CHECK(cudaMalloc(&mem_, (block_size_ * capacity / 8 + 1) * 8));
    CUDA_CHECK(cudaMemset(mem_, 0, block_size_ * capacity));
  }

  char* mem() { return mem_; }

  size_t capacity() { return capacity_; }
  __forceinline__ __device__ void* mem_address(const uint32_t& idx) {
    return &mem_[(idx)*block_size_];
  }

 private:
  char* mem_ = NULL;
  size_t capacity_;
  size_t block_size_;
};

class HBMMemoryPoolFix : public managed {
 public:
  HBMMemoryPoolFix() {
    capacity_ = 0;
    size_ = 0;
    block_size_ = 0;
    max_byte_capacity_ = 0;
  }

  ~HBMMemoryPoolFix() {
    VLOG(3) << "delete hbm memory pool";
    cudaFree(mem_);
  }

  size_t block_size() { return block_size_; }

  void clear(void) { cudaMemset(mem_, 0, block_size_ * capacity_); }

  void reset(size_t capacity, size_t block_size) {
    if (max_byte_capacity_ < capacity * block_size) {
      if (mem_ != NULL) {
        cudaFree(mem_);
      }
      max_byte_capacity_ = (block_size * capacity / 8 + 1) * 8;
      CUDA_CHECK(cudaMalloc(&mem_, max_byte_capacity_));
    }
    size_ = capacity;
    block_size_ = block_size;
    capacity_ = max_byte_capacity_ / block_size;
  }

  char* mem() { return mem_; }

  size_t capacity() { return capacity_; }
  size_t size() { return size_; }
  __forceinline__ __device__ void* mem_address(const uint32_t& idx) {
    return &mem_[(idx)*block_size_];
  }

 private:
  char* mem_ = NULL;
  size_t capacity_;
  size_t size_;
  size_t block_size_;
  size_t max_byte_capacity_;
};

}  // namespace framework
}  // namespace paddle
#endif
#endif

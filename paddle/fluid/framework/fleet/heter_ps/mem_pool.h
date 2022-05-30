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
// #include
// "paddle/fluid/framework/fleet/heter_ps/cudf/concurrent_unordered_map.cuh.h"
#include <iostream>
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/fleet/heter_ps/cudf/managed.cuh"

namespace paddle {
namespace framework {

class MemoryPool {
 public:
  MemoryPool(size_t capacity, size_t block_size)
      : capacity_(capacity), block_size_(block_size) {
    VLOG(3) << "mem_pool init with block_size: " << block_size
            << " capacity: " << capacity;
    mem_ = (char*)malloc(block_size * capacity_);
  }
  ~MemoryPool() {
    VLOG(3) << "mem pool delete";
    free(mem_);
  }
  size_t block_size() { return block_size_; }
  char* mem() { return mem_; }

  size_t capacity() { return capacity_; }
  size_t byte_size() { return capacity_ * block_size_; }
  void* mem_address(const uint32_t& idx) {
    return (void*)&mem_[(idx)*block_size_];
  }

 private:
  char* mem_ = NULL;
  size_t capacity_;
  size_t block_size_;
};

class HBMMemoryPool : public managed {
 public:
  HBMMemoryPool(size_t capacity, size_t block_size)
      : capacity_(capacity), block_size_(block_size) {}
  HBMMemoryPool(MemoryPool* mem_pool) {
    capacity_ = mem_pool->capacity();
    block_size_ = mem_pool->block_size();
    VLOG(3) << "hbm memory pool with capacity" << capacity_
            << " bs: " << block_size_;
    cudaMalloc(&mem_, block_size_ * capacity_);
    cudaMemcpy(mem_, mem_pool->mem(), mem_pool->byte_size(),
               cudaMemcpyHostToDevice);
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
    cudaMalloc(&mem_, (block_size_ * capacity / 8 + 1) * 8);
    cudaMemset(mem_, 0, block_size_ * capacity);
  }

  friend std::ostream& operator<<(std::ostream& out, HBMMemoryPool& p) {
    for (size_t k = 0; k < 5; k++) {
      auto x = (FeatureValue*)(p.mem() + k * p.capacity());
      out << "show: " << x->show << " clk: " << x->clk << " slot: " << x->slot
          << " lr: " << x->lr << " mf_dim: " << x->mf_size
          << " mf_size: " << x->mf_size << " mf:";
      for (int i = 0; i < x->mf_size + 1; ++i) {
        out << " " << x->mf[i];
      }
      out << "\n";
    }
    return out;
  }

  char* mem() { return mem_; }

  size_t capacity() { return capacity_; }
  __forceinline__ __device__ void* mem_address(const uint32_t& idx) {
    return (void*)&mem_[(idx)*block_size_];
  }

 private:
  char* mem_ = NULL;
  size_t capacity_;
  size_t block_size_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
#endif

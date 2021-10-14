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
__constant__   int global_device_num = 1;
// set global device num
inline void set_global_device_num(int device_num) {
  cudaMemcpyToSymbol(global_device_num, &device_num, sizeof(int));
}
// gpu hash function
template <typename Key>
struct HashFunc {
  using argument_type = Key;
  using result_type = uint32_t;
  __forceinline__ __host__ __device__ 
  result_type operator()(const Key& key) const {
    return static_cast<uint32_t>(key / global_device_num);
  }
};

template <typename KeyType, typename Hasher = HashFunc<KeyType>>
class BlockMemoryPool : public managed {
 public:
  BlockMemoryPool(size_t capacity, size_t block_size) : _capacity(capacity), _block_size(block_size) {
    cudaMalloc(&_mem, (block_size * capacity / 8 + 1) * 8);
    cudaMalloc(&_key_mem, sizeof(int8_t) * (capacity / 8 + 1) * 8);
    cudaMemset(_key_mem, 0, sizeof(int8_t) * capacity);
    cudaMemset(_mem, 0, block_size * capacity);
  }
    
  ~BlockMemoryPool() {
    cudaFree(_mem);
    cudaFree(_key_mem);
  }
 
  __forceinline__
  __device__ void allocate(void** ptr, const KeyType &key) {
    size_t index = hf(key) % _capacity;
    size_t count = 0;
    while (true) {
        if (atomicCAS(_key_mem + index, (int8_t)0, (int8_t)1) == (int8_t)1) {
            if (count++ >= _capacity) {
                assert(false && "allocate fail.");
            }
        } else {
            break;
        }
        index = (index + 1) % _capacity;
    }
    *ptr = _mem + index * _block_size;
  }
 
  size_t block_size() {
    return _block_size;
  }

  void clear(void) {
    cudaMemset(_key_mem, 0, sizeof(int8_t) * _capacity);
    cudaMemset(_mem, 0, _block_size * _capacity);
  }
  
  char *mem() {
    return _mem;
  }
  
  size_t capacity() {
    return _capacity;
  }
    
  __forceinline__
  __device__ uint32_t acquire(const KeyType &key) {
    uint32_t index = (uint32_t)(hf(key) % _capacity);
    size_t count = 0;
    while (true) {
      if (atomicCAS(_key_mem + index, (int8_t)0, (int8_t)1) == (int8_t)1) {
        if (count++ >= _capacity) {
          assert(false && "allocate fail.");
        }
      } else {
        break;
      }
      index = (index + 1) % _capacity;
    }
    return (index + 1);
  }

  __forceinline__
  __device__ void* mem_address(const uint32_t &idx) {
    return (void*)&_mem[(idx - 1) * _block_size];
  }
 private:
  char* _mem = NULL;
  int8_t* _key_mem = NULL;
  size_t _capacity;
  size_t _block_size;
  const Hasher hf = Hasher();
};


}  // end namespace framework
}  // end namespace paddle
#endif
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/graph_reindex_kernel.h"

namespace phi {

template <typename T>
inline __device__ size_t Hash(T id, int64_t size) {
  return static_cast<unsigned long long int>(id) % size;  // NOLINT
}

template <typename T>
inline __device__ bool AttemptInsert(
    size_t pos, T id, int index, T* keys, int* key_index) {
  if (sizeof(T) == 4) {
    const T key = atomicCAS(reinterpret_cast<unsigned int*>(&keys[pos]),
                            static_cast<unsigned int>(-1),
                            static_cast<unsigned int>(id));
    if (key == -1 || key == id) {
      atomicMin(reinterpret_cast<unsigned int*>(&key_index[pos]),  // NOLINT
                static_cast<unsigned int>(index));                 // NOLINT
      return true;
    } else {
      return false;
    }
  } else if (sizeof(T) == 8) {
    const T key = atomicCAS(
        reinterpret_cast<unsigned long long int*>(&keys[pos]),  // NOLINT
        static_cast<unsigned long long int>(-1),                // NOLINT
        static_cast<unsigned long long int>(id));               // NOLINT
    if (key == -1 || key == id) {
      atomicMin(reinterpret_cast<unsigned int*>(&key_index[pos]),  // NOLINT
                static_cast<unsigned int>(index));                 // NOLINT
      return true;
    } else {
      return false;
    }
  }
}

template <typename T>
inline __device__ void Insert(
    T id, int index, int64_t size, T* keys, int* key_index) {
  size_t pos = Hash(id, size);
  size_t delta = 1;
  while (!AttemptInsert(pos, id, index, keys, key_index)) {
    pos = Hash(pos + delta, size);
    delta += 1;
  }
}

template <typename T>
inline __device__ int64_t Search(T id, const T* keys, int64_t size) {
  int64_t pos = Hash(id, size);

  int64_t delta = 1;
  while (keys[pos] != id) {
    pos = Hash(pos + delta, size);
    delta += 1;
  }

  return pos;
}

template <typename T>
__global__ void BuildHashTable(
    const T* items, int num_items, int64_t size, T* keys, int* key_index) {
  CUDA_KERNEL_LOOP(index, num_items) {
    Insert(items[index], index, size, keys, key_index);
  }
}

template <typename T>
__global__ void BuildHashTable(const T* items, int num_items, int* key_index) {
  CUDA_KERNEL_LOOP(index, num_items) {
    atomicMin(
        reinterpret_cast<unsigned int*>(&key_index[items[index]]),  // NOLINT
        static_cast<unsigned int>(index));                          // NOLINT
  }
}

template <typename T>
__global__ void ResetHashTable(const T* items,
                               int num_items,
                               int* key_index,
                               int* values) {
  CUDA_KERNEL_LOOP(index, num_items) {
    key_index[items[index]] = -1;
    values[items[index]] = -1;
  }
}

template <typename T>
__global__ void GetItemIndexCount(const T* items,
                                  int* item_count,
                                  int num_items,
                                  int64_t size,
                                  const T* keys,
                                  int* key_index) {
  CUDA_KERNEL_LOOP(i, num_items) {
    int64_t pos = Search(items[i], keys, size);
    if (key_index[pos] == i) {
      item_count[i] = 1;
    }
  }
}

template <typename T>
__global__ void GetItemIndexCount(const T* items,
                                  int* item_count,
                                  int num_items,
                                  int* key_index) {
  CUDA_KERNEL_LOOP(i, num_items) {
    if (key_index[items[i]] == i) {
      item_count[i] = 1;
    }
  }
}

template <typename T>
__global__ void FillUniqueItems(const T* items,
                                int num_items,
                                int64_t size,
                                T* unique_items,
                                const int* item_count,
                                const T* keys,
                                int* values,
                                int* key_index) {
  CUDA_KERNEL_LOOP(i, num_items) {
    int64_t pos = Search(items[i], keys, size);
    if (key_index[pos] == i) {
      values[pos] = item_count[i];
      unique_items[item_count[i]] = items[i];
    }
  }
}

template <typename T>
__global__ void FillUniqueItems(const T* items,
                                int num_items,
                                T* unique_items,
                                const int* item_count,
                                int* values,
                                int* key_index) {
  CUDA_KERNEL_LOOP(i, num_items) {
    if (key_index[items[i]] == i) {
      values[items[i]] = item_count[i];
      unique_items[item_count[i]] = items[i];
    }
  }
}

template <typename T>
__global__ void ReindexSrcOutput(T* src_output,
                                 int64_t num_items,
                                 int64_t size,
                                 const T* keys,
                                 const int* values) {
  CUDA_KERNEL_LOOP(i, num_items) {
    int64_t pos = Search(src_output[i], keys, size);
    src_output[i] = values[pos];
  }
}

template <typename T>
__global__ void ReindexSrcOutput(T* src_output,
                                 int num_items,
                                 const int* values) {
  CUDA_KERNEL_LOOP(i, num_items) { src_output[i] = values[src_output[i]]; }
}

template <typename T>
__global__ void ReindexInputNodes(const T* nodes,
                                  int num_items,
                                  T* reindex_nodes,
                                  int64_t size,
                                  const T* keys,
                                  const int* values) {
  CUDA_KERNEL_LOOP(i, num_items) {
    int64_t pos = Search(nodes[i], keys, size);
    reindex_nodes[i] = values[pos];
  }
}

}  // namespace phi

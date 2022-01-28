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

#pragma once

namespace paddle {
namespace operators {

template <typename IdType>
inline __device__ size_t Hash(IdType id, int64_t size) {
  return id % size;
}

template <typename IdType>
inline __device__ bool AttemptInsert(size_t pos, IdType id, int64_t index,
                                     IdType* keys, int64_t* key_index) {
  if (sizeof(IdType) == 4) {
    const IdType key =
        atomicCAS(reinterpret_cast<unsigned int*>(&keys[pos]),
                  static_cast<unsigned int>(-1), static_cast<unsigned int>(id));
    if (key == -1 || key == id) {
      atomicMin(
          reinterpret_cast<unsigned long long int*>(&key_index[pos]),  // NOLINT
          static_cast<unsigned long long int>(index));                 // NOLINT
      return true;
    } else {
      return false;
    }
  } else if (sizeof(IdType) == 8) {
    const IdType key = atomicCAS(
        reinterpret_cast<unsigned long long int*>(&keys[pos]),  // NOLINT
        static_cast<unsigned long long int>(-1),                // NOLINT
        static_cast<unsigned long long int>(id));               // NOLINT
    if (key == -1 || key == id) {
      atomicMin(
          reinterpret_cast<unsigned long long int*>(&key_index[pos]),  // NOLINT
          static_cast<unsigned long long int>(index));                 // NOLINT
      return true;
    } else {
      return false;
    }
  }
}

template <typename IdType>
inline __device__ void Insert(IdType id, int64_t index, int64_t size,
                              IdType* keys, int64_t* key_index) {
  size_t pos = Hash(id, size);
  size_t delta = 1;
  while (!AttemptInsert(pos, id, index, keys, key_index)) {
    pos = Hash(pos + delta, size);
    delta += 1;
  }
}

template <typename IdType>
inline __device__ int64_t Search(IdType id, const IdType* keys, int64_t size) {
  int64_t pos = Hash(id, size);

  int64_t delta = 1;
  while (keys[pos] != id) {
    pos = Hash(pos + delta, size);
    delta += 1;
  }

  return pos;
}

template <typename IdType>
__global__ void BuildHashTable(const IdType* items, int64_t num_items,
                               int64_t size, IdType* keys, int64_t* key_index) {
  CUDA_KERNEL_LOOP_TYPE(index, num_items, int64_t) {
    Insert(items[index], index, size, keys, key_index);
  }
}

template <typename IdType>
__global__ void GetItemIndexCount(const IdType* items, int* item_count,
                                  int64_t num_items, int64_t size,
                                  const IdType* keys, int64_t* key_index) {
  CUDA_KERNEL_LOOP_TYPE(i, num_items, int64_t) {
    int64_t pos = Search(items[i], keys, size);
    if (key_index[pos] == i) {
      item_count[i] = 1;
    }
  }
}

template <typename IdType>
__global__ void FillUniqueItems(const IdType* items, int64_t num_items,
                                int64_t size, IdType* unique_items,
                                const int* item_count, const IdType* keys,
                                IdType* values, int64_t* key_index) {
  CUDA_KERNEL_LOOP_TYPE(i, num_items, int64_t) {
    int64_t pos = Search(items[i], keys, size);
    if (key_index[pos] == i) {
      values[pos] = item_count[i];
      unique_items[item_count[i]] = items[i];
    }
  }
}

template <typename IdType>
__global__ void ReindexSrcOutput(IdType* src_output, int64_t num_items,
                                 int64_t size, const IdType* keys,
                                 const IdType* values) {
  CUDA_KERNEL_LOOP_TYPE(i, num_items, int64_t) {
    int64_t pos = Search(src_output[i], keys, size);
    src_output[i] = values[pos];
  }
}

template <typename IdType>
__global__ void ReindexInputNodes(const IdType* nodes, int64_t num_items,
                                  IdType* reindex_nodes, int64_t size,
                                  const IdType* keys, const IdType* values) {
  CUDA_KERNEL_LOOP_TYPE(i, num_items, int64_t) {
    int64_t pos = Search(nodes[i], keys, size);
    reindex_nodes[i] = values[pos];
  }
}

}  // namespace operators
}  // namespace paddle

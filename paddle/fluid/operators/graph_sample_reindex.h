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

namespace paddle {
namespace operators {

template <typename IdType>
class DeviceHashTable {
 public:
  static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
  struct Mapping {
    IdType key;  // True ID Key
    int index;   // Smallest ID index: for maintaining unique order.
    int value;   // Reindex ID value
  };
  typedef Mapping* Iterator;

  DeviceHashTable(int64_t num, int scale) {
    int64_t log_num = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
    int64_t size = log_num << scale;
    void* p;
    cudaMalloc(&p, sizeof(Mapping) * size);
    cudaMemset(p, kEmptyKey, sizeof(Mapping) * size);
    table_ = reinterpret_cast<Mapping*>(p);
    size_ = size;
  }

  ~DeviceHashTable() { cudaFree(table_); }

  inline __device__ Iterator Search(IdType id) {
    printf("Enter Search function.\n");
    printf("Search id: %lld\n", id);
    int64_t pos = SearchForPosition(id);
    printf("Search position: %lld \n", pos);
    return &table_[pos];
  }

  inline __device__ Iterator Insert(IdType id, int64_t index) {
    printf("Enter Insert function.\n");
    size_t pos = Hash(id);
    printf("Insert id: %d, pos: %lld\n", id, pos);

    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }
    printf("Out Insert function.\n");
    return GetMutable(pos);
  }

 private:
  Mapping* table_;
  int64_t size_;

  inline __device__ int64_t Hash(IdType id) {
    // printf("Enter Hash function.\n");
    // int64_t tmp1 = static_cast<int64_t>(id);
    // int64_t tmp2 = this->size_;
    // int64_t out = tmp1 % tmp2;
    // printf("Get here.");
    // int64_t out = static_cast<int64_t>(id) % this->size_;
    // printf("Out Hash function.\n");
    // return out;
    // return 0;
    int64_t size = 16;
    return static_cast<int64_t>(id % static_cast<IdType>(size));
    // return id % this->size_;
  }

  inline __device__ int64_t SearchForPosition(IdType id) {
    printf("Ente SearchForPosition function.\n");
    printf("SearchForPosition id: %lld\n", id);
    // int64_t pos = Hash(id);
    // printf("%lld %lld\n", this->size_, static_cast<int64_t>(id));
    int64_t pos_new = static_cast<int64_t>(id % 16);
    // printf("%lld\n", pos_new);
    // printf("id: %lld, pos_new: %lld, size_: %lld", id, pos_new, size_);
    // IdType delta = 1;
    // while (table_[pos].key != id) {
    //  pos = Hash(pos + delta);
    //  delta += 1;
    //}
    printf("Out SearchForPosition function.\n");
    return pos_new;
  }

  inline __device__ bool AttemptInsertAt(size_t pos, IdType id, int64_t index) {
    using T = unsigned long long int;
    const IdType key = atomicCAS(reinterpret_cast<T*>(&GetMutable(pos)->key),
                                 static_cast<T>(kEmptyKey), static_cast<T>(id));
    if (key == kEmptyKey || key == id) {  // Why kEmptyKey
      printf("atomicMin\n");
      atomicMin(reinterpret_cast<unsigned int*>(&GetMutable(pos)->index),
                static_cast<unsigned int>(index));
      return true;
    } else {
      printf("False\n");
      return false;  // Search somewhere else.
    }
  }

  inline __device__ Mapping* GetMutable(size_t pos) {
    assert(pos < size_);
    return table_ + pos;
  }
};

template <typename IdType>
__global__ void build_hashtable_duplicates(const IdType* items,
                                           int64_t num_items,
                                           DeviceHashTable<IdType>* table) {
  CUDA_KERNEL_LOOP_TYPE(index, num_items, int64_t) {
    table->Insert(items[index], index);
  }
}

template <typename IdType>
__global__ void get_item_index_count(IdType* items, int* item_count,
                                     int64_t num_items,
                                     DeviceHashTable<IdType>* table) {
  CUDA_KERNEL_LOOP_TYPE(i, num_items, int64_t) {
    printf("%lld %lld %lld\n", i, num_items, items[i]);
    using Mapping = typename DeviceHashTable<IdType>::Mapping;
    table->Search(items[i]);
    // Mapping mapping = *(table.Search(items[i]));
    // if (mapping.index == i) {
    //  item_count[i] = 1;
    //}
  }
}

template <typename IdType>
__global__ void fill_unique_items(IdType* items, int64_t num_items,
                                  IdType* unique_items, const int* item_count,
                                  DeviceHashTable<IdType>& table) {
  CUDA_KERNEL_LOOP_TYPE(i, num_items, int64_t) {
    // using Mapping = typename DeviceHashTable<IdType>::Mapping;
    // Mapping &mapping = *(table.Search(items[i]));
    // if (mapping.index == i) {
    //  mapping.value = item_count[i];
    //  unique_items[item_count[i]] = items[i];
    //}
  }
}

template <typename IdType>
__global__ void reindex_src_output(IdType* src, int64_t num_items,
                                   DeviceHashTable<IdType>& table) {
  CUDA_KERNEL_LOOP_TYPE(i, num_items, int64_t) {
    // using Iterator = typename DeviceHashTable<IdType>::Iterator;
    // Iterator iter = table.Search(src[i]);
    // src[i] = static_cast<IdType>((*iter).value);
  }
}

}  // namespace operators
}  // namespace paddle

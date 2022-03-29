// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "gflags/gflags.h"

#include <mct/hash-map.hpp>
#include "paddle/fluid/distributed/common/chunk_allocator.h"

namespace paddle {
namespace distributed {

static const int CTR_SPARSE_SHARD_BUCKET_NUM_BITS = 6;
static const size_t CTR_SPARSE_SHARD_BUCKET_NUM =
    static_cast<size_t>(1) << CTR_SPARSE_SHARD_BUCKET_NUM_BITS;

class FixedFeatureValue {
 public:
  FixedFeatureValue() {}
  ~FixedFeatureValue() {}
  float* data() { return _data.data(); }
  size_t size() { return _data.size(); }
  void resize(size_t size) { _data.resize(size); }
  void shrink_to_fit() { _data.shrink_to_fit(); }

 private:
  std::vector<float> _data;
};

template <class KEY, class VALUE>
struct alignas(64) SparseTableShard {
 public:
  typedef typename mct::closed_hash_map<KEY, mct::Pointer, std::hash<KEY>>
      map_type;
  struct iterator {
    typename map_type::iterator it;
    size_t bucket;
    map_type* buckets;
    friend bool operator==(const iterator& a, const iterator& b) {
      return a.it == b.it;
    }
    friend bool operator!=(const iterator& a, const iterator& b) {
      return a.it != b.it;
    }
    const KEY& key() const { return it->first; }
    VALUE& value() const { return *(VALUE*)(void*)it->second; }     // NOLINT
    VALUE* value_ptr() const { return (VALUE*)(void*)it->second; }  // NOLINT
    iterator& operator++() {
      ++it;

      while (it == buckets[bucket].end() &&
             bucket + 1 < CTR_SPARSE_SHARD_BUCKET_NUM) {
        it = buckets[++bucket].begin();
      }

      return *this;
    }
    iterator operator++(int) {
      iterator ret = *this;
      ++*this;
      return ret;
    }
  };
  struct local_iterator {
    typename map_type::iterator it;
    friend bool operator==(const local_iterator& a, const local_iterator& b) {
      return a.it == b.it;
    }
    friend bool operator!=(const local_iterator& a, const local_iterator& b) {
      return a.it != b.it;
    }
    const KEY& key() const { return it->first; }
    VALUE& value() const { return *(VALUE*)(void*)it->second; }  // NOLINT
    local_iterator& operator++() {
      ++it;
      return *this;
    }
    local_iterator operator++(int) { return {it++}; }
  };

  ~SparseTableShard() { clear(); }
  bool empty() { return _alloc.size() == 0; }
  size_t size() { return _alloc.size(); }
  void set_max_load_factor(float x) {
    for (size_t bucket = 0; bucket < CTR_SPARSE_SHARD_BUCKET_NUM; bucket++) {
      _buckets[bucket].max_load_factor(x);
    }
  }
  size_t bucket_count() { return CTR_SPARSE_SHARD_BUCKET_NUM; }
  size_t bucket_size(size_t bucket) { return _buckets[bucket].size(); }
  void clear() {
    for (size_t bucket = 0; bucket < CTR_SPARSE_SHARD_BUCKET_NUM; bucket++) {
      map_type& data = _buckets[bucket];
      for (auto it = data.begin(); it != data.end(); ++it) {
        _alloc.release((VALUE*)(void*)it->second);  // NOLINT
      }
      data.clear();
    }
  }
  iterator begin() {
    auto it = _buckets[0].begin();
    size_t bucket = 0;
    while (it == _buckets[bucket].end() &&
           bucket + 1 < CTR_SPARSE_SHARD_BUCKET_NUM) {
      it = _buckets[++bucket].begin();
    }
    return {it, bucket, _buckets};
  }
  iterator end() {
    return {_buckets[CTR_SPARSE_SHARD_BUCKET_NUM - 1].end(),
            CTR_SPARSE_SHARD_BUCKET_NUM - 1, _buckets};
  }
  local_iterator begin(size_t bucket) { return {_buckets[bucket].begin()}; }
  local_iterator end(size_t bucket) { return {_buckets[bucket].end()}; }
  iterator find(const KEY& key) {
    size_t hash = _hasher(key);
    size_t bucket = compute_bucket(hash);
    auto it = _buckets[bucket].find_with_hash(key, hash);
    if (it == _buckets[bucket].end()) {
      return end();
    }
    return {it, bucket, _buckets};
  }
  VALUE& operator[](const KEY& key) { return emplace(key).first.value(); }
  std::pair<iterator, bool> insert(const KEY& key, const VALUE& val) {
    return emplace(key, val);
  }
  std::pair<iterator, bool> insert(const KEY& key, VALUE&& val) {
    return emplace(key, std::move(val));
  }
  template <class... ARGS>
  std::pair<iterator, bool> emplace(const KEY& key, ARGS&&... args) {
    size_t hash = _hasher(key);
    size_t bucket = compute_bucket(hash);
    auto res = _buckets[bucket].insert_with_hash({key, NULL}, hash);

    if (res.second) {
      res.first->second = _alloc.acquire(std::forward<ARGS>(args)...);
    }

    return {{res.first, bucket, _buckets}, res.second};
  }
  iterator erase(iterator it) {
    _alloc.release((VALUE*)(void*)it.it->second);  // NOLINT
    size_t bucket = it.bucket;
    auto it2 = _buckets[bucket].erase(it.it);
    while (it2 == _buckets[bucket].end() &&
           bucket + 1 < CTR_SPARSE_SHARD_BUCKET_NUM) {
      it2 = _buckets[++bucket].begin();
    }
    return {it2, bucket, _buckets};
  }
  void quick_erase(iterator it) {
    _alloc.release((VALUE*)(void*)it.it->second);  // NOLINT
    _buckets[it.bucket].quick_erase(it.it);
  }
  local_iterator erase(size_t bucket, local_iterator it) {
    _alloc.release((VALUE*)(void*)it.it->second);  // NOLINT
    return {_buckets[bucket].erase(it.it)};
  }
  void quick_erase(size_t bucket, local_iterator it) {
    _alloc.release((VALUE*)(void*)it.it->second);  // NOLINT
    _buckets[bucket].quick_erase(it.it);
  }
  size_t erase(const KEY& key) {
    auto it = find(key);
    if (it == end()) {
      return 0;
    }
    quick_erase(it);
    return 1;
  }
  size_t compute_bucket(size_t hash) {
    if (CTR_SPARSE_SHARD_BUCKET_NUM == 1) {
      return 0;
    } else {
      return hash >> (sizeof(size_t) * 8 - CTR_SPARSE_SHARD_BUCKET_NUM_BITS);
    }
  }

 private:
  map_type _buckets[CTR_SPARSE_SHARD_BUCKET_NUM];
  ChunkAllocator<VALUE> _alloc;
  std::hash<KEY> _hasher;
};

}  // namespace distributed
}  // namespace paddle

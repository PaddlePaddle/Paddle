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
#include <glog/logging.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

// Fast allocation and deallocation of objects by allocating them in chunks.
template <class T>
class ChunkAllocator {
 public:
  explicit ChunkAllocator(size_t chunk_size = 64) {
    PADDLE_ENFORCE_EQ(
        sizeof(Node),
        std::max(sizeof(void*), sizeof(T)),
        common::errors::InvalidArgument(
            "The size of Node is invalid. Expected sizeof(Node) == "
            "max(sizeof(void*), sizeof(T)).\nBut received sizeof(Node) = %u "
            "and max(sizeof(void*), sizeof(T)) = %u.",
            sizeof(Node),
            std::max(sizeof(void*), sizeof(T))));
    _chunk_size = chunk_size;
    _chunks = NULL;
    _free_nodes = NULL;
    _counter = 0;
  }
  ChunkAllocator(const ChunkAllocator&) = delete;
  ~ChunkAllocator() {
    while (_chunks != NULL) {
      Chunk* x = _chunks;
      _chunks = _chunks->next;
      free(x);
    }
  }
  template <class... ARGS>
  T* acquire(ARGS&&... args) {
    if (_free_nodes == NULL) {
      create_new_chunk();
    }

    T* x = (T*)(void*)_free_nodes;  // NOLINT
    _free_nodes = _free_nodes->next;
    new (x) T(std::forward<ARGS>(args)...);
    _counter++;
    return x;
  }
  void release(T* x) {
    x->~T();
    Node* node = (Node*)(void*)x;  // NOLINT
    node->next = _free_nodes;
    _free_nodes = node;
    _counter--;
  }
  size_t size() const { return _counter; }

 private:
  struct alignas(T) Node {
    union {
      Node* next;
      char data[sizeof(T)];
    };
  };
  struct Chunk {
    Chunk* next;
    Node nodes[];
  };

  size_t _chunk_size;  // how many elements in one chunk
  Chunk* _chunks;      // a list
  Node* _free_nodes;   // a list
  size_t _counter;     // how many elements are acquired

  void create_new_chunk() {
    Chunk* chunk;
    size_t alloc_size = sizeof(Chunk) + sizeof(Node) * _chunk_size;
    int error = posix_memalign(reinterpret_cast<void**>(&chunk),
                               std::max<size_t>(sizeof(void*), alignof(Chunk)),
                               alloc_size);
    PADDLE_ENFORCE_EQ(error,
                      0,
                      common::errors::ResourceExhausted(
                          "Fail to alloc memory of %ld size, error code is %d.",
                          alloc_size,
                          error));
    chunk->next = _chunks;
    _chunks = chunk;

    for (size_t i = 0; i < _chunk_size; i++) {
      Node* node = &chunk->nodes[i];
      node->next = _free_nodes;
      _free_nodes = node;
    }
  }
};

}  // namespace distributed
}  // namespace paddle

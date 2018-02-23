/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <cstddef>

namespace paddle {
namespace memory {
namespace detail {

// Forward Declarations
class MetadataCache;

/*! \brief A class used to interpret the contents of a memory block */
class MemoryBlock {
 public:
  enum Type {
    FREE_CHUNK,    // memory is free and idle
    ARENA_CHUNK,   // memory is being occupied
    HUGE_CHUNK,    // memory is out of management
    INVALID_CHUNK  // memory is invalid
  };

 public:
  void init(MetadataCache& cache, Type t, size_t index, size_t size,
            void* left_buddy, void* right_buddy);

 public:
  /*! \brief The type of the allocation */
  Type type(MetadataCache& cache) const;

  /*! \brief The size of the data region */
  size_t size(MetadataCache& cache) const;

  /*! \brief An index to track the allocator */
  size_t index(MetadataCache& cache) const;

  /*! \brief The total size of the block */
  size_t total_size(MetadataCache& cache) const;

  /*! \brief Check the left buddy of the block */
  bool has_left_buddy(MetadataCache& cache) const;

  /*! \brief Check the right buddy of the block */
  bool has_right_buddy(MetadataCache& cache) const;

  /*! \brief Get the left buddy */
  MemoryBlock* left_buddy(MetadataCache& cache) const;

  /*! \brief Get the right buddy */
  MemoryBlock* right_buddy(MetadataCache& cache) const;

 public:
  /*! \brief Split the allocation into left/right blocks */
  void split(MetadataCache& cache, size_t size);

  /*! \brief Merge left and right blocks together */
  void merge(MetadataCache& cache, MemoryBlock* right_buddy);

  /*! \brief Mark the allocation as free */
  void mark_as_free(MetadataCache& cache);

  /*! \brief Change the type of the allocation */
  void set_type(MetadataCache& cache, Type t);

 public:
  /*! \brief Get a pointer to the memory block's data */
  void* data() const;

  /*! \brief Get a pointer to the memory block's metadata */
  MemoryBlock* metadata() const;

 public:
  static size_t overhead();
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle

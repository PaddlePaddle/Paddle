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

#include <memory>

#include "paddle/infrt/common/macros.h"
#include "paddle/infrt/common/memory.h"
#include "paddle/infrt/common/target.h"

namespace infrt {

#ifdef __cplusplus
extern "C" {
#endif

#define INFRT_ALWAYS_INLINE __attribute__((always_inline)) inline

//! Code for the primitive types supported in INFRT.
typedef enum infrt_type_code_t {
  infrt_type_unk = -1,   //! Unknown type
  infrt_type_int = 0,    //! signed int
  infrt_type_uint = 1,   //! unsigned int
  infrt_type_float = 2,  //! floating point
  infrt_type_handle = 3  //! void*
} infrt_type_code_t;

#ifndef INFRT_ATTRIBUTE_ALIGN
#define INFRT_ATTRIBUTE_ALIGN(n) __attribute__((aligned(n)))
#endif

/**
 * A tuntime tag for type in INFRT system.
 */
typedef struct infrt_type_t {
#if __cplusplus >= 201103L
  INFRT_ATTRIBUTE_ALIGN(1) infrt_type_code_t code;
#else
  uint8_t code;
#endif

  //! Number of bits.
  uint8_t bits;

  //! Number of elements in a vector, 1 for scalar.
  uint16_t lanes;

  //! Number of '*', e.g. for `float*`, the num_asterisks is 1, `float**` it is
  //! 2.
  uint8_t num_asterisks{0};

#ifdef __cplusplus
  INFRT_ALWAYS_INLINE infrt_type_t()
      : code(infrt_type_int), bits(0), lanes(0) {}
  INFRT_ALWAYS_INLINE infrt_type_t(infrt_type_code_t code,
                                   uint8_t bits,
                                   uint16_t lanes = 1,
                                   uint8_t num_asterisks = 0)
      : code(code), bits(bits), lanes(lanes), num_asterisks(num_asterisks) {}
  INFRT_ALWAYS_INLINE bool operator==(const infrt_type_t& other) const {
    return code == other.code && bits == other.bits && lanes == other.lanes;
  }
  INFRT_ALWAYS_INLINE bool operator!=(const infrt_type_t& other) const {
    return !(*this == other);
  }
  INFRT_ALWAYS_INLINE uint16_t bytes() const { return (bits + 7) / 8; }
#endif  // __cplusplus
} infrt_type_t;

//! Help to define the size of a dimension, due to polyhedral representation, we
//! no need to record the extend or
//! min(default to 0).
typedef int infrt_dimension_t;

//! Help to tell the kind of the device.
typedef enum infrt_device_kind_t {
  infrt_unk_device = -1,    // Undefined device.
  infrt_x86_device = 0,     // X86 device
  infrt_opencl_device = 1,  // OpenCL device
  infrt_arm_device = 2      // ARM device
} infrt_device_kind_t;

struct infrt_buffer_t;

/**
 * All INFRT backends implementation should provide an interface to be used.
 */
struct infrt_device_interface_impl_t;

struct infrt_device_interface_t {
  int (*malloc)(void* context, struct infrt_buffer_t* buf);
  int (*free)(void* context, struct infrt_buffer_t* buf);
  int (*sync)(void* context, struct infrt_buffer_t* buf);
  int (*release)(void* context,
                 const struct infrt_device_interface_t* device_interface);
  int (*copy_to_host)(void* context, struct infrt_buffer_t* buf);
  int (*copy_to_device)(void* context, struct infrt_buffer_t* buf);
  int (*buffer_copy)(void* context,
                     struct infrt_buffer_t* src,
                     struct infrt_buffer_t* dst);
  struct infrt_device_interface_impl_t* impl;
};

//! The raw representation of a buffer,used in the generated code/lib.
#define INFRT_BUFFER_MAX_DIMS 8
typedef struct infrt_buffer_t {
  //! Tell which kind of device this buffer locates.
  infrt_device_kind_t device;

  //! The interface used to operate on device.
  const struct infrt_device_interface_t* device_interface;

  //! A pointer to the memory in host.
  uint8_t* memory;

  //! Extra flags.
  uint64_t flag;

  //! Data type.
  infrt_type_t type;

  //! Number of dimensions.
  int32_t dimensions;
  infrt_dimension_t dims[INFRT_BUFFER_MAX_DIMS];

  //! Allocate and deallocate lazily, default true.
  char lazy;

  //! The actual memory size(in bytes).
  uint64_t memory_size;

  uint16_t align;

#ifdef __cplusplus
  infrt_buffer_t()
      : device(infrt_unk_device),
        device_interface(NULL),
        memory(NULL),
        flag(0UL),
        type(infrt_type_t()),
        dimensions(0),
        lazy(true),
        memory_size(0),
        align(0) {}

  static void delete_(struct infrt_buffer_t* x) { delete x; }

  ~infrt_buffer_t() {}

  // NOTE the buffer should be resized first.
  static void alloc(struct infrt_buffer_t*);

  //! Set the shape of the buffer. NOTE this just record the shape, not allocate
  //! the memory.
  INFRT_ALWAYS_INLINE void resize(const infrt_dimension_t* dims,
                                  int dimensions) {
    this->dimensions = dimensions;
    memcpy(this->dims, dims, dimensions * sizeof(infrt_dimension_t));
  }

  INFRT_ALWAYS_INLINE uint64_t num_elements() const {
    uint64_t res = 1;
    for (int i = 0; i < dimensions; i++) {
      res *= dims[i];
    }
    return res;
  }

  INFRT_ALWAYS_INLINE int device_sync(void* ctx = NULL) {
    if (device_interface && device_interface->sync) {
      return device_interface->sync(ctx, this);
    }
    return 0;
  }

  INFRT_ALWAYS_INLINE uint8_t* begin() const { return 0; }
  INFRT_ALWAYS_INLINE uint8_t* end() const {
    return memory + num_elements() * type.bytes();
  }

#endif  // __cplusplus
} infrt_buffer_t;

#ifdef __cplusplus
struct infrt_device_interface_impl_t {
  int (*malloc)(void* context, struct infrt_buffer_t* buf);
  int (*free)(void* context, struct infrt_buffer_t* buf);
  int (*sync)(void* context, struct infrt_buffer_t* buf);
  int (*release)(void* context);
  int (*copy_to_host)(void* context, struct infrt_buffer_t* buf);
  int (*copy_to_device)(void* context, struct infrt_buffer_t* buf);
  int (*buffer_copy)(void* context,
                     struct infrt_buffer_t* src,
                     struct infrt_buffer_t* dst);
};

// The device implementations
extern struct infrt_device_interface_t* infrt_x86_device_interface();
#endif  // __cplusplus

#ifdef __cplusplus
}  // extern "C"
#endif

#define INFRT_LOG(fmt, ...)     \
  do {                          \
    fprintf(stderr,             \
            "%s:%d:%s(): " fmt, \
            __FILE__,           \
            __LINE__,           \
            __func__,           \
            __VA_ARGS__);       \
  } while (0)

#define INFRT_CHECK(cond)                \
  if (!(cond)) {                         \
    INFRT_LOG("check %s failed", #cond); \
    abort();                             \
  }
/**
 * Buffer helps to hold the memory, and offers a set of methods to help manage
 * the memory.
 */
struct Buffer final {
  Buffer() = default;
  explicit Buffer(const common::Target& target) { SetTarget(target); }

  //! Resize the memory hold by this buffer *exactlly* to \p size.
  void Resize(uint32_t size);
  void Resize(uint32_t alignment, uint32_t size);

  //! Lazily resize the memory.
  void ResizeLazy(uint32_t size);
  void ResizeLazy(uint32_t alignment, uint32_t size);

  //! Resize the memory to \p size in target \p target.
  void Resize(uint32_t size, const common::Target& target);
  void Resize(uint32_t alignment, uint32_t size, const common::Target& target);

  //! Lazily resize the memory to \p size in target \p target.
  void ResizeLazy(uint32_t size, const common::Target& target);
  void ResizeLazy(uint32_t alignment,
                  uint32_t size,
                  const common::Target& target);

  void SetTarget(const common::Target& target);

  const infrt_buffer_t* data() const { return &data_; }
  infrt_buffer_t* data() { return &data_; }

  //! Free all the memory owned by this buffer.
  void Free() {
    if (!data_.memory) return;
    memory_mng_cache_->free(data_.memory);
  }

 private:
  inline void* Malloc(uint32_t size) INFRT_RESULT_SHOULD_USE {
    CHECK(memory_mng_cache_) << "Should set target first";
    return memory_mng_cache_->malloc(size);
  }

  inline void* AlignedAlloc(uint32_t alignment,
                            uint32_t size) INFRT_RESULT_SHOULD_USE {
    CHECK(memory_mng_cache_) << "Should set target first";
    return memory_mng_cache_->aligned_alloc(alignment, size);
  }

 private:
  infrt_buffer_t data_;

  //! The place where this buffer locates.
  common::Target target_;

  //! Number of bytes of this buffer.
  uint32_t size_{};

  //! Hold the corresponding memory manager for speed.
  MemoryInterface* memory_mng_cache_{};
};

}  // namespace infrt

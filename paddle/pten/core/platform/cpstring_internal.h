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

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) ||                  \
    defined(_WIN32)
#define PD_PSTRING_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define PD_PSTRING_LITTLE_ENDIAN 0
#else
#error "Unable to detect endianness."
#endif

#if defined(__clang__) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
static inline uint32_t swap32(uint32_t host_int) {
  return __builtin_bswap32(host_int);
}

#elif defined(_MSC_VER)
static inline uint32_t swap32(uint32_t host_int) {
  return _byteswap_ulong(host_int);
}

#elif defined(__APPLE__)
static inline uint32_t swap32(uint32_t host_int) {
  return OSSwapInt32(host_int);
}

#else
static inline uint32_t swap32(uint32_t host_int) {
#if defined(__GLIBC__)
  return bswap_32(host_int);
#else   // defined(__GLIBC__)
  return (((host_int & uint32_t{0xFF}) << 24) |
          ((host_int & uint32_t{0xFF00}) << 8) |
          ((host_int & uint32_t{0xFF0000}) >> 8) |
          ((host_int & uint32_t{0xFF000000}) >> 24));
#endif  // defined(__GLIBC__)
}
#endif

#if PD_PSTRING_LITTLE_ENDIAN
#define PD_le32toh(x) x
#else  // PD_PSTRING_LITTLE_ENDIAN
#define PD_le32toh(x) swap32(x)
#endif  // PD_PSTRING_LITTLE_ENDIAN

static inline size_t PD_align16(size_t i) { return (i + 0xF) & ~0xF; }

static inline size_t PD_max(size_t a, size_t b) { return a > b ? a : b; }
static inline size_t PD_min(size_t a, size_t b) { return a < b ? a : b; }

typedef enum PD_PString_Type {  // NOLINT
  PD_PSTR_SMALL = 0x00,
  PD_PSTR_LARGE = 0x01,
  PD_PSTR_OFFSET = 0x02,
  PD_PSTR_VIEW = 0x03,
  PD_PSTR_TYPE_MASK = 0x03
} PD_PString_Type;

typedef struct PD_PString_Large {  // NOLINT
  size_t size;
  size_t cap;
  char *ptr;
} PD_PString_Large;

typedef struct PD_PString_Offset {  // NOLINT
  uint32_t size;
  uint32_t offset;
  uint32_t count;
} PD_PString_Offset;

typedef struct PD_PString_View {  // NOLINT
  size_t size;
  const char *ptr;
} PD_PString_View;

typedef struct PD_PString_Raw {  // NOLINT
  uint8_t raw[24];
} PD_PString_Raw;

typedef union PD_PString_Union {  // NOLINT
  PD_PString_Large large;
  PD_PString_Offset offset;
  PD_PString_View view;
  PD_PString_Raw raw;
} PD_PString_Union;

enum {
  PD_PString_SmallCapacity =
      (sizeof(PD_PString_Union) - sizeof(/* null delim */ char) -
       sizeof(/* uint8_t size */ uint8_t)),
};

typedef struct PD_PString_Small {  // NOLINT
  uint8_t size;
  char str[PD_PString_SmallCapacity + sizeof(/* null delim */ char)];
} PD_PString_Small;

typedef struct PD_PString {  // NOLINT
  union {
    // small conflicts with '#define small char' in RpcNdr.h for MSVC, so we use
    // smll instead.
    PD_PString_Small smll;
    PD_PString_Large large;
    PD_PString_Offset offset;
    PD_PString_View view;
    PD_PString_Raw raw;
  } u;
} PD_PString;

// TODO(dero): Fix for OSS, and add C only build test.
// _Static_assert(CHAR_BIT == 8);
// _Static_assert(sizeof(PD_PString) == 24);

static inline PD_PString_Type PD_PString_GetType(const PD_PString *str) {
  return (PD_PString_Type)(str->u.raw.raw[0] & PD_PSTR_TYPE_MASK);  // NOLINT
}

// XXX(dero): For the big-endian case, this function could potentially be more
// performant and readable by always storing the string size as little-endian
// and always byte-swapping on big endian, resulting in a simple 'bswap'+'shr'
// (for architectures that have a bswap op).
static inline size_t PD_PString_ToActualSizeT(size_t size) {
#if PD_PSTRING_LITTLE_ENDIAN
  return size >> 2;
#else   // PD_PSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);  // NOLINT

  return (((mask << 2) & size) >> 2) | (~mask & size);
#endif  // PD_PSTRING_LITTLE_ENDIAN
}

static inline size_t PD_PString_ToInternalSizeT(size_t size,
                                                PD_PString_Type type) {
#if PD_PSTRING_LITTLE_ENDIAN
  return (size << 2) | type;
#else   // PD_PSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);  // NOLINT

  return (mask & (size << 2)) | (~mask & size) |
         ((size_t)type << ((sizeof(size_t) - 1) * 8));  // NOLINT
#endif  // PD_PSTRING_LITTLE_ENDIAN
}

static inline void PD_PString_Init(PD_PString *str) {
  memset(str->u.raw.raw, 0, sizeof(PD_PString_Raw));
}

static inline void PD_PString_Dealloc(PD_PString *str) {
  if (PD_PString_GetType(str) == PD_PSTR_LARGE &&
      str->u.large.ptr != NULL) {  // NOLINT
    free(str->u.large.ptr);
    PD_PString_Init(str);
  }
}

static inline size_t PD_PString_GetSize(const PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return str->u.smll.size >> 2;
    case PD_PSTR_LARGE:
      return PD_PString_ToActualSizeT(str->u.large.size);
    case PD_PSTR_OFFSET:
      return PD_le32toh(str->u.offset.size) >> 2;
    case PD_PSTR_VIEW:
      return PD_PString_ToActualSizeT(str->u.view.size);
    default:
      return 0;  // Unreachable.
  }
}

static inline size_t PD_PString_GetCapacity(const PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return PD_PString_SmallCapacity;
    case PD_PSTR_LARGE:
      return str->u.large.cap;
    case PD_PSTR_OFFSET:
    case PD_PSTR_VIEW:
    default:
      return 0;
  }
}

static inline const char *PD_PString_GetDataPointer(const PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return str->u.smll.str;
    case PD_PSTR_LARGE:
      return str->u.large.ptr;
    case PD_PSTR_OFFSET:
      return (const char *)str + str->u.offset.offset;  // NOLINT
    case PD_PSTR_VIEW:
      return str->u.view.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

static inline char *PD_PString_ResizeUninitialized(PD_PString *str,
                                                   size_t new_size) {
  size_t curr_size = PD_PString_GetSize(str);
  size_t copy_size = PD_min(new_size, curr_size);

  PD_PString_Type curr_type = PD_PString_GetType(str);
  const char *curr_ptr = PD_PString_GetDataPointer(str);

  // Case: SMALL/LARGE/VIEW/OFFSET -> SMALL
  if (new_size <= PD_PString_SmallCapacity) {
    str->u.smll.size = (uint8_t)((new_size << 2) | PD_PSTR_SMALL);  // NOLINT
    str->u.smll.str[new_size] = '\0';

    if (curr_type != PD_PSTR_SMALL && copy_size) {
      memcpy(str->u.smll.str, curr_ptr, copy_size);
    }

    if (curr_type == PD_PSTR_LARGE) {
      free((void *)curr_ptr);  // NOLINT
    }

    // We do not clear out the newly excluded region.

    return str->u.smll.str;
  }

  // Case: SMALL/LARGE/VIEW/OFFSET -> LARGE
  size_t new_cap;
  size_t curr_cap = PD_PString_GetCapacity(str);

  if (new_size < curr_size && new_size < curr_cap / 2) {
    // TODO(dero): Replace with shrink_to_fit flag.
    new_cap = PD_align16(curr_cap / 2 + 1) - 1;
  } else if (new_size > curr_cap) {
    new_cap = PD_align16(new_size + 1) - 1;
  } else {
    new_cap = curr_cap;
  }

  char *new_ptr;
  if (new_cap == curr_cap) {
    new_ptr = str->u.large.ptr;
  } else if (curr_type == PD_PSTR_LARGE) {
    new_ptr = (char *)realloc(str->u.large.ptr, new_cap + 1);  // NOLINT
  } else {
    new_ptr = (char *)malloc(new_cap + 1);  // NOLINT
    if (copy_size) {
      memcpy(new_ptr, curr_ptr, copy_size);
    }
  }

  str->u.large.size = PD_PString_ToInternalSizeT(new_size, PD_PSTR_LARGE);
  str->u.large.ptr = new_ptr;
  str->u.large.ptr[new_size] = '\0';
  str->u.large.cap = new_cap;

  return str->u.large.ptr;
}

static inline char *PD_PString_GetMutableDataPointer(PD_PString *str) {
  switch (PD_PString_GetType(str)) {
    case PD_PSTR_SMALL:
      return str->u.smll.str;
    case PD_PSTR_OFFSET:
    case PD_PSTR_VIEW:
      // Convert OFFSET/VIEW to SMALL/LARGE
      PD_PString_ResizeUninitialized(str, PD_PString_GetSize(str));
      return (PD_PString_GetType(str) == PD_PSTR_SMALL) ? str->u.smll.str
                                                        : str->u.large.ptr;
    case PD_PSTR_LARGE:
      return str->u.large.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

static inline void PD_PString_Reserve(PD_PString *str, size_t new_cap) {
  PD_PString_Type curr_type = PD_PString_GetType(str);

  if (new_cap <= PD_PString_SmallCapacity) {
    // We do nothing, we let Resize/GetMutableDataPointer handle the
    // conversion to SMALL from VIEW/OFFSET when the need arises.
    // In the degenerate case, where new_cap <= PD_PString_SmallCapacity,
    // curr_size > PD_PString_SmallCapacity, and the type is VIEW/OFFSET, we
    // defer the malloc to Resize/GetMutableDataPointer.
    return;
  }

  if (curr_type == PD_PSTR_LARGE && new_cap <= str->u.large.cap) {
    // We handle reduced cap in resize.
    return;
  }

  // Case: VIEW/OFFSET -> LARGE or grow an existing LARGE type
  size_t curr_size = PD_PString_GetSize(str);
  const char *curr_ptr = PD_PString_GetDataPointer(str);

  // Since VIEW and OFFSET types are read-only, their capacity is effectively 0.
  // So we make sure we have enough room in the VIEW and OFFSET cases.
  new_cap = PD_align16(PD_max(new_cap, curr_size) + 1) - 1;

  if (curr_type == PD_PSTR_LARGE) {
    str->u.large.ptr =
        (char *)realloc(str->u.large.ptr, new_cap + 1);  // NOLINT
  } else {
    // Convert to Large
    char *new_ptr = (char *)malloc(new_cap + 1);  // NOLINT
    memcpy(new_ptr, curr_ptr, curr_size);

    str->u.large.size = PD_PString_ToInternalSizeT(curr_size, PD_PSTR_LARGE);
    str->u.large.ptr = new_ptr;
    str->u.large.ptr[curr_size] = '\0';
  }

  str->u.large.cap = new_cap;
}

static inline void PD_PString_ReserveAmortized(PD_PString *str,
                                               size_t new_cap) {
  const size_t curr_cap = PD_PString_GetCapacity(str);
  if (new_cap > curr_cap) {
    PD_PString_Reserve(str, new_cap > 2 * curr_cap ? new_cap : 2 * curr_cap);
  }
}

static inline char *PD_PString_Resize(PD_PString *str,
                                      size_t new_size,
                                      char c) {
  size_t curr_size = PD_PString_GetSize(str);
  char *cstr = PD_PString_ResizeUninitialized(str, new_size);

  if (new_size > curr_size) {
    memset(cstr + curr_size, c, new_size - curr_size);
  }

  return cstr;
}

static inline void PD_PString_AssignView(PD_PString *dst,
                                         const char *src,
                                         size_t size) {
  PD_PString_Dealloc(dst);

  dst->u.view.size = PD_PString_ToInternalSizeT(size, PD_PSTR_VIEW);
  dst->u.view.ptr = src;
}

static inline void PD_PString_AppendN(PD_PString *dst,
                                      const char *src,
                                      size_t src_size) {
  if (!src_size) return;

  size_t dst_size = PD_PString_GetSize(dst);

  // For append use cases, we want to ensure amortized growth.
  PD_PString_ReserveAmortized(dst, dst_size + src_size);
  char *dst_c = PD_PString_ResizeUninitialized(dst, dst_size + src_size);

  memcpy(dst_c + dst_size, src, src_size);
}

static inline void PD_PString_Append(PD_PString *dst, const PD_PString *src) {
  const char *src_c = PD_PString_GetDataPointer(src);
  size_t size = PD_PString_GetSize(src);

  PD_PString_AppendN(dst, src_c, size);
}

static inline void PD_PString_Copy(PD_PString *dst,
                                   const char *src,
                                   size_t size) {
  char *dst_c = PD_PString_ResizeUninitialized(dst, size);

  if (size) memcpy(dst_c, src, size);
}

static inline void PD_PString_Assign(PD_PString *dst, const PD_PString *src) {
  if (dst == src) return;

  PD_PString_Dealloc(dst);

  switch (PD_PString_GetType(src)) {
    case PD_PSTR_SMALL:
    case PD_PSTR_VIEW:
      *dst = *src;
      return;
    case PD_PSTR_LARGE: {
      const char *src_c = PD_PString_GetDataPointer(src);
      size_t size = PD_PString_GetSize(src);

      PD_PString_Copy(dst, src_c, size);
    }
      return;
    case PD_PSTR_OFFSET: {
      const char *src_c = PD_PString_GetDataPointer(src);
      size_t size = PD_PString_GetSize(src);

      PD_PString_AssignView(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}

static inline void PD_PString_Move(PD_PString *dst, PD_PString *src) {
  if (dst == src) return;

  PD_PString_Dealloc(dst);

  switch (PD_PString_GetType(src)) {
    case PD_PSTR_SMALL:
    case PD_PSTR_VIEW:
      *dst = *src;
      return;
    case PD_PSTR_LARGE:
      *dst = *src;
      PD_PString_Init(src);
      return;
    case PD_PSTR_OFFSET: {
      const char *src_c = PD_PString_GetDataPointer(src);
      size_t size = PD_PString_GetSize(src);

      PD_PString_AssignView(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}

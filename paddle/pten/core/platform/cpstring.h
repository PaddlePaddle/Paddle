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

#include <stdint.h>
#include <stdlib.h>

#include "paddle/pten/core/platform/cpstring_internal.h"

// Initialize a new tstring.  This must be called before using any function
// below.
inline void PD_PString_Init(PD_PString *str);
// Deallocate a tstring.
inline void PD_PString_Dealloc(PD_PString *str);

// Resizes `str' to `new_size'.  This function will appropriately grow or shrink
// the string buffer to fit a `new_size' string.  Grown regions of the string
// will be initialized with `c'.
inline char *PD_PString_Resize(PD_PString *str, size_t new_size, char c);
// Similar to PD_PString_Resize, except the newly allocated regions will remain
// uninitialized.  This is useful if you plan on overwriting the newly grown
// regions immediately after allocation; doing so will elide a superfluous
// initialization of the new buffer.
inline char *PD_PString_ResizeUninitialized(PD_PString *str, size_t new_size);
// Reserves a string buffer with a capacity of at least `new_cap'.
// Reserve will not change the size, or the contents of the existing
// string.  This is useful if you have a rough idea of `str's upperbound in
// size, and want to avoid allocations as you append to `str'. It should not be
// considered safe to write in the region between size and capacity; explicitly
// resize before doing so.
inline void PD_PString_Reserve(PD_PString *str, size_t new_cap);
// Similar to PD_PString_Reserve, except that we ensure amortized growth, i.e.
// that we grow the capacity by at least a constant factor >1.
inline void PD_PString_ReserveAmortized(PD_PString *str, size_t new_cap);

// Returns the size of the string.
inline size_t PD_PString_GetSize(const PD_PString *str);
// Returns the capacity of the string buffer.  It should not be considered safe
// to write in the region between size and capacity---call Resize or
// ResizeUninitialized before doing so.
inline size_t PD_PString_GetCapacity(const PD_PString *str);
// Returns the underlying type of the tstring:
// PD_PSTR_SMALL:
//    Small string optimization; the contents of strings
//    less than 22-bytes are stored in the PD_PString struct. This avoids any
//    heap allocations.
// PD_PSTR_LARGE:
//    Heap allocated string.
// PD_PSTR_OFFSET: (currently unused)
//    An offset defined string.  The string buffer begins at an internally
//    defined little-endian offset from `str'; i.e. GetDataPointer() = str +
//    offset.  This type is useful for memory mapping or reading string tensors
//    directly from file, without the need to deserialize the data.  For
//    security reasons, it is imperative that OFFSET based string tensors are
//    validated before use, or are from a trusted source.
// PD_PSTR_VIEW:
//    A view into an unowned character string.
//
// NOTE:
//    VIEW and OFFSET types are immutable, so any modifcation via Append,
//    AppendN, or GetMutableDataPointer of a VIEW/OFFSET based tstring will
//    result in a conversion to an owned type (SMALL/LARGE).
inline PD_PString_Type PD_PString_GetType(const PD_PString *str);

// Returns a const char pointer to the start of the underlying string. The
// underlying character buffer may not be null-terminated.
inline const char *PD_PString_GetDataPointer(const PD_PString *str);
// Returns a char pointer to a mutable representation of the underlying string.
// In the case of VIEW and OFFSET types, `src' is converted to an owned type
// (SMALL/LARGE).  The underlying character buffer may not be null-terminated.
inline char *PD_PString_GetMutableDataPointer(PD_PString *str);

// Sets `dst' as a VIEW type to `src'.  `dst' will not take ownership of `src'.
// It is the user's responsibility to ensure that the lifetime of `src' exceeds
// `dst'.  Any mutations to `dst' via Append, AppendN, or GetMutableDataPointer,
// will result in a copy into an owned SMALL or LARGE type, and will not modify
// `src'.
inline void PD_PString_AssignView(PD_PString *dst,
                                  const char *src,
                                  size_t size);

// Appends `src' onto `dst'.  If `dst' is a VIEW or OFFSET type, it will first
// be converted to an owned LARGE or SMALL type.  `dst' should not point to
// memory owned by `src'.
inline void PD_PString_Append(PD_PString *dst, const PD_PString *src);
inline void PD_PString_AppendN(PD_PString *dst, const char *src, size_t size);

// Copy/Move/Assign semantics
//
//        | src     | dst          | complexity
// Copy   | *       |  SMALL/LARGE | fixed/O(size)
// Assign | SMALL   |  SMALL       | fixed
// Assign | OFFSET  |  VIEW        | fixed
// Assign | VIEW    |  VIEW        | fixed
// Assign | LARGE   |  LARGE       | O(size)
// Move   | *       |  same as src | fixed

// Copies `src' to `dst'. `dst' will be an owned type (SMALL/LARGE). `src'
// should not point to memory owned by `dst'.
inline void PD_PString_Copy(PD_PString *dst, const char *src, size_t size);
// Assigns a `src' tstring to `dst'.  An OFFSET `src' type will yield a `VIEW'
// `dst'.  LARGE `src' types will be copied to a new buffer; all other `src'
// types will incur a fixed cost.
inline void PD_PString_Assign(PD_PString *dst, const PD_PString *src);
// Moves a `src' tstring to `dst'.  Moving a LARGE `src' to `dst' will result in
// a valid but unspecified `src'.  This function incurs a fixed cost for all
// inputs.
inline void PD_PString_Move(PD_PString *dst, PD_PString *src);

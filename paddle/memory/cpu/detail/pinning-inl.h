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

#ifdef _WIN32

#else
#include <sys/mman.h>

#endif

namespace paddle {
namespace memory {
namespace cpu {

inline void pin_memory(void* address, size_t size) {
#if _WIN32

#else
  mlock(address, size);
#endif
}

inline void unpin_memory(void* address, size_t size) {
#if _WIN32

#else
  munlock(address, size);
#endif
}

} /* cpu */
} /* memory */
} /* paddle */

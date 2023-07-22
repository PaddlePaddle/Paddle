/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// Define sys/mman.h Macros instead of <sys/mman.h> to avoid
// cppcoreguidelines-pro-type-cstyle-cast

/* Return value of `mmap' in case of an error.  */
#define MAP_FAILED_s (reinterpret_cast<void *>(-1))

#ifdef MAP_FAILED
#undef MAP_FAILED
#define MAP_FAILED MAP_FAILED_s
#endif

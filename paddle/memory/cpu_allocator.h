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
namespace cpu {

class SystemAllocator {
 public:
  static void* malloc(size_t& index, size_t size);
  static void free(void* address, size_t size, size_t index);

 public:
  static size_t index_count();

 public:
  static void init();
  static void shutdown();

 public:
  static bool uses_gpu();
};

}  // namespace cpu
}  // namespace memory
}  // namespace paddle

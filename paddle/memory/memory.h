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

#include "paddle/platform/gpu_info.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace memory {

template <typename Place>
void* Alloc(Place, size_t);

template <typename Place>
void Free(Place, void*);

template <typename Place>
size_t Used(Place);

template <typename T, typename Place>
class PODDeleter {
  static_assert(std::is_pod<T>::value, "T must be POD");

 public:
  PODDeleter(Place place) : place_(place) {}
  void operator()(T* ptr) { Free(place_, static_cast<void*>(ptr)); }

 private:
  Place place_;
};

}  // namespace memory
}  // namespace paddle

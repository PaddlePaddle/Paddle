// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/temporay_allocator.h"
#include <gtest/gtest.h>

namespace paddle {
namespace platform {

TEST(temporary_allocator, temporary_allocator) {
  platform::CPUPlace cpu_place;
  TemporaryAllocator alloc(cpu_place);
  auto allocation = alloc.Allocate(100);

#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace gpu_place(0);
  TemporaryAllocator gpu_alloc(gpu_place);
  auto gpu_allocation = gpu_alloc.Allocate(100);

  gpu_alloc.MoveToDeleteQueue();
  gpu_alloc.Release();
#endif
}
}  //  namespace platform
}  //  namespace paddle

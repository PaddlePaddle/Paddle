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

#include "paddle/infrt/common/memory.h"

namespace infrt {

using infrt::common::Target;

namespace {

class X86MemoryMng : public MemoryInterface {
 public:
  void* malloc(size_t nbytes) override { return ::malloc(nbytes); }
  void free(void* data) override {
    if (!data) return;
    ::free(data);
  }
  void* aligned_alloc(size_t alignment, size_t nbytes) override {
    return ::aligned_alloc(alignment, nbytes);
  }
};

}  // namespace

MemoryManager::MemoryManager() {
  Register(Target::Arch::Unk, new X86MemoryMng);
  Register(Target::Arch::X86, new X86MemoryMng);
}

}  // namespace infrt

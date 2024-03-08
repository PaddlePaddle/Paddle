// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/memory.h"

#include "paddle/cinn/runtime/backend_api.h"
using cinn::runtime::BackendAPI;

namespace cinn {
namespace hlir {
namespace framework {

using common::Target;

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


class CudaMemoryMng : public MemoryInterface {
 public:
  void* malloc(size_t nbytes) override {
      return BackendAPI::get_backend(Target::Language::cuda)->malloc(nbytes);
    }
  void free(void* data) override { 
    BackendAPI::get_backend(Target::Language::cuda)->free(data); 
  }
};


class SYCLMemoryMng : public MemoryInterface {
  public:
    void* malloc(size_t nbytes) override {
      return BackendAPI::get_backend(Target::Language::sycl)->malloc(nbytes);
    }
    void free(void* data) override {
      BackendAPI::get_backend(Target::Language::sycl)->free(data);
    }
};

class HIPMemoryMng : public MemoryInterface {
  public:
    void* malloc(size_t nbytes) override {
      return BackendAPI::get_backend(Target::Language::hip)->malloc(nbytes);
    }
    void free(void* data) override {
      BackendAPI::get_backend(Target::Language::hip)->free(data);
    }
};

}  // namespace

MemoryManager::MemoryManager() {
  Register(Target::Language::Unk, new X86MemoryMng);
  Register(Target::Language::llvm, new X86MemoryMng);
  Register(Target::Language::cuda, new CudaMemoryMng);
  Register(Target::Language::sycl, new SYCLMemoryMng);
  Register(Target::Language::hip, new HIPMemoryMng);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

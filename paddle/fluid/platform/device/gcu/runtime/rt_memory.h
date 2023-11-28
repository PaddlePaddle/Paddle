/* Copyright (c) 2023 Enflame. All Rights Reserved.

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
#include <memory>
#include <vector>

#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

struct MemoryImpl;
struct Memory {
  Context *ctx;

  static std::shared_ptr<Memory> CreateMemory(Context *ctx, uint64_t nbytes);

  static std::shared_ptr<Memory> CreateSubMemory(std::shared_ptr<Memory> parent,
                                                 uint64_t offset,
                                                 uint64_t size);

  void SetDims(const std::vector<int64_t> &dims, bool dynamic = false);

  std::vector<int64_t> GetDims(bool dynamic = false) const;

  bool IsValid() const { return mem_impl != nullptr; }

  void *GetDevAddr() const;

  uint64_t Nbytes() const;

  Memory(Context *ctx,
         std::shared_ptr<MemoryImpl> mem,
         uint64_t offset,
         uint64_t nbytes);

  ~Memory();

  Memory() = delete;

 private:
  std::shared_ptr<MemoryImpl> mem_impl;
  uint64_t offset;
  uint64_t nbytes;
  std::vector<int64_t> dims_;

 public:
  RT_DISALLOW_COPY_AND_ASSIGN(Memory);
};

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle

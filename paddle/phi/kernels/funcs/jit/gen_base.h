/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <memory>  // for unique_ptr
#include <string>
#include <vector>

#ifdef _WIN32
#include <malloc.h>  // for _aligned_malloc
#endif

#include "gflags/gflags.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/funcs/jit/kernel_base.h"

PHI_DECLARE_bool(dump_jitcode);

namespace phi {
namespace jit {

class GenBase : public Kernel {
 public:
  virtual ~GenBase() {}
  virtual std::string name() const = 0;
  virtual size_t getSize() const = 0;
  virtual const unsigned char* getCodeInternal() const = 0;
  const char* ImplType() const override { return "JitCode"; }
  template <typename Func>
  Func getCode() const {
    const unsigned char* code = this->getCodeInternal();
    if (FLAGS_dump_jitcode) {
      this->dumpCode(code);
    }
    // Note: failed to cast with reinterpret_cast<const Func> on Mac clang,
    // then workaround with const_cast. Any better idea is appreciated.
    return reinterpret_cast<Func>(const_cast<unsigned char*>(code));
  }

  void* operator new(size_t size);
  void operator delete(void* ptr);
  void* operator new[](size_t size) { return operator new(size); }
  void operator delete[](void* ptr) { operator delete(ptr); }

 protected:
  void dumpCode(const unsigned char* code) const;
};

// Creator is used to creat the jitcode and save in pool.
// Every JitCode should have one creator.
class GenCreator {
 public:
  virtual ~GenCreator() = default;
};

template <typename Attr>
class JitCodeCreator : public GenCreator {
 public:
  virtual ~JitCodeCreator() = default;

  // condition when this jit code can be used.
  virtual bool CanBeUsed(const Attr& attr) const = 0;

  // estimate this code size
  virtual size_t CodeSize(const Attr& attr) const = 0;

  // create this code
  virtual std::unique_ptr<GenBase> CreateJitCode(const Attr& attr) const = 0;
};

// unify the method of packed groups
// output the packed groups which used in weights, the block size and rest size
std::vector<int> packed_groups(int n,
                               int k,
                               int* block = nullptr,
                               int* rest = nullptr);

}  // namespace jit
}  // namespace phi

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <memory>  // for unique_ptr
#include "paddle/fluid/operators/jit/kernel_base.h"

DECLARE_bool(dump_jitcode);

namespace paddle {
namespace operators {
namespace jit {

class GenBase : public Kernel {
 public:
  virtual ~GenBase() = default;
  virtual const char* name() const = 0;
  virtual size_t getSize() const = 0;
  virtual const unsigned char* getCodeInternal() = 0;
  template <typename FUNC>
  const FUNC getCode() {
    const unsigned char* code = this->getCodeInternal();
    if (FLAGS_dump_jitcode) {
      this->dumpCode(code);
    }
    return reinterpret_cast<const FUNC>(code);
  }

 protected:
  void dumpCode(const unsigned char* code) const;
};

// Every JitCode should have a method to get the key from attribution
template <typename Attr>
size_t JitCodeKey(Attr attr);

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
  virtual bool UseMe(const Attr& attr) const = 0;

  // estimate this code size
  virtual size_t CodeSize(const Attr& attr) const = 0;

  // create this code
  virtual std::unique_ptr<GenBase> CreateJitCode(const Attr& attr) const = 0;
};

}  // namespace jit
}  // namespace operators
}  // namespace paddle

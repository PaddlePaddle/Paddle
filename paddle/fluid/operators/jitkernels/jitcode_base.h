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
#include "paddle/fluid/operators/jitkernels/kernel_base.h"
#include "paddle/fluid/platform/macros.h"

DECLARE_bool(dump_jitcode);

namespace paddle {
namespace operators {
namespace jitkernels {

// TODO(TJ): make these functions as virtual of a class

// Every JitCode should estimate the code size itself
template <KernelType KT, typename Attr>
size_t CodeSize(Attr attr) {
  return 4096;
}

// Every JitCode should have a condition when to use this JitCode
template <KernelType KT, typename T, typename Attr>
bool UseJitCode(Attr attr) {
  return false;
}

// Every JitCode should have a method to get the key from attribution
template <typename Attr>
size_t GetKey(Attr attr);

template <>
size_t GetKey<int>(int d) {
  return d;
}

class JitBase {
 public:
  JitBase() = default;
  virtual ~JitBase() = default;
  virtual const char* name() const = 0;
  virtual const unsigned char* getCodeInternal() = 0;

  template <typename FUNC>
  const FUNC getCode() {
    const unsigned char* code = this->getCodeInternal();
    if (FLAGS_dump_jitcode) {
      this->dumpCode(code);
    }
    return reinterpret_cast<const FUNC>(code);
  }
  DISABLE_COPY_AND_ASSIGN(JitBase);

 protected:
  void dumpCode(const unsigned char* code);
};

}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle

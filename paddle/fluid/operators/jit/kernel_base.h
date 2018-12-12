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
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {
namespace jit {

typedef enum {
  vmul = 0,
  vadd = 1,
  vaddrelu,
  vsub,
  vscal,
  vaddbias,
  vexp
} KernelType;

template <typename T>
struct XYZNTuples {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, const T*, T*, int);
};

template <typename T>
struct AXYNTuples : public XYZNTuples<T> {};

// Just for adding to kernel pool without template
class Kernel {
 public:
  Kernel() = default;
  virtual ~Kernel() = default;
  DISABLE_COPY_AND_ASSIGN(Kernel);
};

template <typename KernelTuples>
class KernelImpl : public Kernel {
  using T = typename KernelTuples::data_type;
  using Func = typename KernelTuples::func_type;
  using Attr = typename KernelTuples::attr_type;

 public:
  virtual Func GetFunc() const { return func; }
  virtual bool UseMe(Attr attr) const = 0;

 protected:
  Func func{nullptr};
};

template <typename KernelTuples>
class ReferKernel : public KernelImpl<KernelTuples> {
 public:
  // Refer code can always be used
  bool UseMe(typename KernelTuples::attr_type attr) const override {
    return true;
  }
};

}  // namespace jit
}  // namespace operators
}  // namespace paddle

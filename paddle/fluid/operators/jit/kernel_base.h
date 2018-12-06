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

typedef enum { vmul = 0, vadd = 1, vsub, vexp } KernelType;

template <typename T>
struct VMulTypes {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, const T*, T*, int);
};

// Just for adding to kernel pool without template
class Kernel {
 public:
  Kernel() = default;
  virtual ~Kernel() = default;
  DISABLE_COPY_AND_ASSIGN(Kernel);
};

template <typename T, typename Func, typename Attr>
class KernelImpl : public Kernel {
 public:
  using ELEMENT_TYPE = T;
  virtual Func GetFunc() const { return func; }
  virtual bool UseMe(Attr attr) const = 0;

 protected:
  Func func{nullptr};
};

template <typename T, typename Func, typename Attr>
class ReferKernel : public KernelImpl<T, Func, Attr> {
 public:
  // Refer code can always be used
  bool UseMe(Attr attr) const override { return true; }
};

}  // namespace jit
}  // namespace operators
}  // namespace paddle

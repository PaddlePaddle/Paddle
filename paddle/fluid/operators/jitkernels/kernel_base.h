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
namespace jitkernels {

typedef enum { vmul = 0, vadd = 1, vsub, vexp } KernelType;

// Just for adding to kernel pool without template
class Kernel {
 public:
  Kernel() = default;
  DISABLE_COPY_AND_ASSIGN(Kernel);
};

template <typename T, typename Func, typename Attr>  // TODO(TJ): use tuple
class KernelImpl : public Kernel {
 public:
  using ELEMENT_TYPE = T;  // TODO(TJ): remove me?
  KernelImpl() = default;
  virtual ~KernelImpl() = default;

  virtual Func GetFunc() { return func; }
  virtual bool UseMe(Attr attr) const = 0;

 protected:
  Func func{nullptr};
};

}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle

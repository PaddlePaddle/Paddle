/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/stream.h"

#include "paddle/pten/core/dense_tensor.h"

namespace paddle {

namespace framework {

using LoD = std::vector<paddle::framework::Vector<size_t>>;

/*
 NOTE(liym27): [ What is TensorInplaceVersion used for? ]

 TensorInplaceVersion is a version counter and every Tensor has a version
 counter. It's used to check whether an inplace operation will result in an
 incorrect gradient calculation. Version is incremented when the data of the
 Variable is modified in place.

 - Question: In what scenarios will version counters be shared?
 - Answer: When two Variables/VarBases share the same C++ Tensor(its Allocation
 may change), both of them share the same version counter. For examples:
  1. `z = paddle.assign(input=x, output=y)`, `z` shares the same version counter
    of `y` because z and y is the same VarBase;
  2. `y = x.detach()`, `y` shares the same version counter of `x`.

 - Question: In what scenarios will version counters NOT be shared?
 - Answer: Replacing a `Variable`'s data by calling `Tensor::ShareDataWith(...)`
 or `Tensor::ShareBufferWith(...)`. Because they share the same Allocation but
 not framework::Tensor.

 - Question: Why put the inplace_version_counter_ in framework::Tensor instead
 of Allocation or Variable?
 - Answer:
  1. Tensor can call ResetHolder() to reset the corresponding Allocation so that
  the inplace_version_counter_ changes if it's in Allocation, which will lead to
  confusing information about inplace version.
  2. If inplace_version_counter_ is in Variable, different VariableWrappers
  should be able to share the same Variable. However, a VariableWrapper hold a
  Variable object but not a pointer.
*/

using Tensor = pten::DenseTensor;

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/tensor_impl.h"

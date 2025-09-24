// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/ap/include/code_module/adt.h"
#include "paddle/ap/include/code_module/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

struct TypedBufferImpl {
  void* buffer;
  phi::DataType dtype;
  size_t size;

  bool operator==(const TypedBufferImpl& other) const {
    return other.buffer == this->buffer && other.dtype == this->dtype &&
           other.size == this->size;
  }
};
ADT_DEFINE_RC(TypedBuffer, TypedBufferImpl);

}  // namespace ap::kernel_dispatch

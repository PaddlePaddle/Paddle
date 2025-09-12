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
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/code_module/data_type.h"
#include "paddle/ap/include/kernel_dispatch/typed_buffer.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class DenseTensor;

}

namespace ap::kernel_dispatch {

using MutableTensorDataImpl = std::variant<phi::DenseTensor*, TypedBuffer>;
struct MutableTensorData : public MutableTensorDataImpl {
  using MutableTensorDataImpl::MutableTensorDataImpl;
  ADT_DEFINE_VARIANT_METHODS(MutableTensorDataImpl);

  template <typename T>
  T* data() const {
    return Match(
        [](phi::DenseTensor* tensor) -> T* {
          return reinterpret_cast<T*>(tensor->data());
        },
        [](const TypedBuffer& buffer) -> T* {
          return reinterpret_cast<T*>(buffer->buffer);
        });
  }
  void* data() const {
    return Match(
        [](phi::DenseTensor* tensor) -> void* { return tensor->data(); },
        [](const TypedBuffer& buffer) -> void* { return buffer->buffer; });
  }
  phi::DataType dtype() const {
    return Match([](phi::DenseTensor* tensor) { return tensor->dtype(); },
                 [](const TypedBuffer& buffer) { return buffer->dtype; });
  }
};

template <typename ValueT>
struct MutableTensorImpl {
  MutableTensorData tensor_data;
  adt::List<ValueT> dims;

  bool operator==(const MutableTensorImpl& other) const {
    return other.tensor_data == this->tensor_data && other.dims == this->dims;
  }
};
template <typename ValueT>
ADT_DEFINE_RC(MutableTensor, MutableTensorImpl<ValueT>);

}  // namespace ap::kernel_dispatch

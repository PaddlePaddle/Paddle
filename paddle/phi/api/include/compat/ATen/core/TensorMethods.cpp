// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under   BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <string_view>

namespace at {

void check_type(const TensorBase& tensor,
                ScalarType type,
                std::string_view type_name) {
  PD_CHECK(tensor.scalar_type() == type,
           "expected scalar type ",
           type_name,
           " but found ",
           compat::_PD_AtenScalarTypeToPhiDataType(tensor.scalar_type()));
}

#define DEFINE_CAST(T, name)                                        \
  template <>                                                       \
  PADDLE_API const T* TensorBase::const_data_ptr() const {          \
    check_type(*this, ScalarType::name, #name);                     \
    return const_cast<T*>(tensor_.data<T>());                       \
  }                                                                 \
                                                                    \
  template <>                                                       \
  PADDLE_API const T* TensorBase::const_data_ptr<const T>() const { \
    check_type(*this, ScalarType::name, #name);                     \
    return const_cast<T*>(tensor_.data<std::remove_const_t<T>>());  \
  }                                                                 \
                                                                    \
  template <>                                                       \
  PADDLE_API T* TensorBase::mutable_data_ptr() const {              \
    check_type(*this, ScalarType::name, #name);                     \
    return const_cast<PaddleTensor&>(tensor_).data<T>();            \
  }                                                                 \
                                                                    \
  template <>                                                       \
  PADDLE_API T* TensorBase::data_ptr() const {                      \
    return const_cast<T*>(tensor_.data<T>());                       \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)  // missing half and float16
// AT_FORALL_QINT_TYPES(DEFINE_CAST) // missing qint
DEFINE_CAST(uint16_t, UInt16)
DEFINE_CAST(uint32_t, UInt32)
DEFINE_CAST(uint64_t, UInt64)
#undef DEFINE_CAST

}  // namespace at

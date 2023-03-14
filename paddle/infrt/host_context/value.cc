// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/host_context/value.h"

#include "paddle/infrt/tensor/dense_tensor_view.h"

namespace infrt {
namespace host_context {

ValueRef::ValueRef(int32_t val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(int64_t val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(float val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(double val) : Shared<Value>(new Value(val)) {}
ValueRef::ValueRef(bool val) : Shared<Value>(new Value(val)) {}

const char* Value::type_info() const { return __type_info__; }

void CopyTo(const Value& from, Value* to) {
  CHECK(from.valid()) << "from value is not valid, can't be copied";
  CHECK(to) << "to is not valid";
  visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if (std::is_same<T, int16_t>::value)
          to->data = reinterpret_cast<int16_t const&>(arg);
        else if (std::is_same<T, int32_t>::value)
          to->data = reinterpret_cast<int32_t const&>(arg);
        else if (std::is_same<T, float>::value)
          to->data = reinterpret_cast<float const&>(arg);
        else if (std::is_same<T, double>::value)
          to->data = reinterpret_cast<double const&>(arg);
        else if (std::is_same<T, uint32_t>::value)
          to->data = reinterpret_cast<uint32_t const&>(arg);
        else if (std::is_same<T, uint64_t>::value)
          to->data = reinterpret_cast<uint64_t const&>(arg);
        else if (std::is_same<T, bool>::value)
          to->data = reinterpret_cast<bool const&>(arg);
        else if (std::is_same<T, tensor::TensorShape>::value)
          to->data = reinterpret_cast<tensor::TensorShape const&>(arg);
        else if (std::is_same<T, MlirFunctionExecutable*>::value)
          to->data = reinterpret_cast<MlirFunctionExecutable* const&>(arg);
        else if (std::is_same<T, tensor::DenseHostTensor>::value)
          to->data = reinterpret_cast<tensor::DenseHostTensor const&>(arg);
        else if (std::is_same<T, std::vector<int16_t>>::value)
          to->data = reinterpret_cast<std::vector<int16_t> const&>(arg);
        else if (std::is_same<T, std::vector<int64_t>>::value)
          to->data = reinterpret_cast<std::vector<int64_t> const&>(arg);
        else if (std::is_same<T, tensor::TensorMap>::value)
          to->data = reinterpret_cast<tensor::TensorMap const&>(arg);
#ifdef INFRT_WITH_PHI
        else if (std::is_same<T, ::Tensor>::value)
          to->data = reinterpret_cast<::Tensor const&>(arg);
#endif
        else
          LOG(FATAL) << "Not supported Value copy: " << typeid(T).name();
      },
      from.data);
}

}  // namespace host_context
}  // namespace infrt

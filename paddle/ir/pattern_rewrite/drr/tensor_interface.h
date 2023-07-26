// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <string>
#include <typeindex>

namespace cinn {
namespace hlir {
namespace drr {

class ShapeInterface;
class DtypeInterface;

class TensorInterface final {
 public:
  TensorInterface(const std::string& tensor_name) : tensor_name_(tensor_name) {
    TODO(/*thisjiang*/)
  }

  const ShapeInterface& Shape() const;
  const DtypeInterface& Dtype() const;

 private:
  std::string tensor_name_;
};

class ShapeInterface {
 public:
  template <typename T>
  const T& Value() const {
    CHECK(TypeIndex4Shape() == typeid(T));
    const T* ptr = reinterpret_cast<const T*>(Value());
    CHECK(ptr != nullptr) << "shape should not be null";
    return *ptr;
  }

 protected:
  virtual std::type_index TypeIndex4Shape() const = 0;
  virtual const void* Value() const = 0;
};

class DtypeInterface {
 public:
  template <typename T>
  const T& Value() const {
    CHECK(TypeIndex4Dtype() == typeid(T));
    const T* ptr = reinterpret_cast<const T*>(Value());
    CHECK(ptr != nullptr) << "dtype should not be null";
    return *ptr;
  }

 protected:
  virtual std::type_index TypeIndex4Dtype() const = 0;
  virtual const void* Value() const = 0;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn

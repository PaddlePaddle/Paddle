// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace ir {
namespace drr {

class ShapeInterface;
class DtypeInterface;

class TensorInterface {
 public:
  explicit TensorInterface(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  const ShapeInterface& Shape() const;
  const DtypeInterface& Dtype() const;

 private:
  std::string tensor_name_;
};

class ShapeInterface {
 public:
  bool operator==(const ShapeInterface& other) const;

 protected:
};

class DtypeInterface {
 public:
  bool operator==(const DtypeInterface& other) const;

 protected:
};

}  // namespace drr
}  // namespace ir

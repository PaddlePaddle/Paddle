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

#include <cstdint>

namespace pir {
namespace drr {

class IrValue;
class IrShape;
class IrDtype;

class ShapeInterface final {
 public:
  bool operator==(const ShapeInterface& other) const;

  int size() const;

  int64_t at(int idx) const;

 private:
  explicit ShapeInterface(const IrShape* shape) : shape_(shape) {}

  friend class IrValue;

  const IrShape* shape_;
};

class DtypeInterface final {
 public:
  bool operator==(const DtypeInterface& other) const;

 private:
  explicit DtypeInterface(const IrDtype* dtype) : dtype_(dtype) {}

  friend class IrValue;

  const IrDtype* dtype_;
};

class TensorInterface {
 public:
  virtual ShapeInterface Shape() const = 0;
  virtual DtypeInterface Dtype() const = 0;
};

}  // namespace drr
}  // namespace pir

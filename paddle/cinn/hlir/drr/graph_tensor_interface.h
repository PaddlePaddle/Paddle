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
#include "cinn/hlir/drr/tensor_interface.h"

#include <typeindex>

#include "cinn/common/type.h"

namespace cinn {
namespace hlir {
namespace drr {

class GraphShapeInterface final : public ShapeInterface {
 public:
  GraphShapeInterface(const std::vector<int>& shape) : shape_(shape) {}

 protected:
  std::type_index TypeIndex4Shape() const override;
  const void* Value() const override;

 private:
  std::vector<int> shape_;
};

class GraphDtypeInterface final : public DtypeInterface {
 public:
  GraphDtypeInterface(const cinn::common::Type& dtype) : dtype_(dtype) {}

 protected:
  virtual std::type_index TypeIndex4Dtype() const override;
  virtual const void* Value() const override;

 private:
  cinn::common::Type dtype_;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn

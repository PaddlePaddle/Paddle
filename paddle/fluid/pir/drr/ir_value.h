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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/drr/api/tensor_interface.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/value.h"

namespace paddle {
namespace drr {

class IrShape {
 public:
  explicit IrShape(const phi::DDim& dims) : dims_(dims) {}

  bool operator==(const IrShape& other) const { return dims_ == other.dims_; }

  int size() const { return dims_.size(); }

  int64_t at(int idx) const { return dims_.at(idx); }

 private:
  const phi::DDim dims_;
};

class IrDtype {
 public:
  explicit IrDtype(pir::Type dtype) : dtype_(dtype) {}

  bool operator==(IrDtype other) const { return dtype_ == other.dtype_; }

  template <typename T>
  bool isa() const {
    return dtype_.isa<T>();
  }

  template <typename T>
  T dyn_cast() const {
    return dtype_.dyn_cast<T>();
  }

 private:
  const pir::Type dtype_;
};

class IrValue : public TensorInterface {
 public:
  explicit IrValue(const pir::Value& value)
      : value_(value),
        shape_((value && value.type() &&
                value.type().dyn_cast<paddle::dialect::DenseTensorType>())
                   ? value.type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dims()
                   : phi::DDim{}),
        dtype_((value && value.type() &&
                value.type().dyn_cast<paddle::dialect::DenseTensorType>())
                   ? value.type()
                         .dyn_cast<paddle::dialect::DenseTensorType>()
                         .dtype()
                   : pir::Type{}) {}

  ShapeInterface Shape() const override { return ShapeInterface(&shape_); }
  DtypeInterface Dtype() const override { return DtypeInterface(&dtype_); }

  explicit operator bool() const { return value_.operator bool(); }

  template <typename T>
  bool isa() const {
    return value_.isa<T>();
  }

  template <typename T>
  T dyn_cast() const {
    return value_.dyn_cast<T>();
  }

  template <typename T>
  bool type_isa() const {
    return value_.type().isa<T>();
  }

  template <typename T>
  T type_dyn_cast() const {
    return value_.type().dyn_cast<T>();
  }

  // Don't use it in drr pass!
  const pir::Value& get() const { return value_; }

 private:
  const pir::Value value_;
  const IrShape shape_;
  const IrDtype dtype_;
};

class IrAttr;

}  // namespace drr
}  // namespace paddle

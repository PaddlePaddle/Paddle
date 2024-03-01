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

#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_simplify.h"

namespace symbol {

template <typename T>
class ShapeOrData {
 public:
  explicit ShapeOrData(const std::vector<T>& shape)
      : shape_(shape), data_(std::nullopt) {}
  explicit ShapeOrData(const std::vector<T>& shape, const std::vector<T>& data)
      : shape_(shape), data_(data) {
    // Vaild check
    if (shape.size() == 0) {
      IR_ENFORCE(data.size() == 1,
                 "When shape is 0-D, size of data shoubld be 1, but got %d.",
                 data.size());
    } else if (shape.size() == 1) {
      IR_ENFORCE(shape[0].template Has<int64_t>(),
                 "When shape is 1-D, value of shape shoubld be int");
      IR_ENFORCE(
          shape[0].template Get<int64_t>() == static_cast<int64_t>(data.size()),
          "When shape is 1-D, size of data shoubld be the same as "
          "value[%d] of shape, but got [%d].",
          shape[0].template Get<std::int64_t>(),
          data.size());
    } else {
      IR_THROW("Size of shape shoubld be 0 or 1, but got %d", shape.size());
    }
  }

  ShapeOrData() = default;
  ShapeOrData(const ShapeOrData&) = default;
  ShapeOrData(ShapeOrData&&) = default;
  ShapeOrData& operator=(const ShapeOrData&) = default;
  ShapeOrData& operator=(ShapeOrData&&) = default;

  // Tensor's real shape
  const std::vector<T>& shape() const { return shape_; }
  // Specific for Tensor generated by shape-relevant ops
  const std::optional<std::vector<T>>& data() const { return data_; }
  void SetData(const std::vector<T>& data) { data_ = data; }

  bool operator==(const ShapeOrData<T>& other) const {
    if (data_.has_value() && !other.data_.has_value()) return false;
    if (!data_.has_value() && other.data_.has_value()) return false;
    if (shape_.size() != shape_.size()) return false;

    if (data_.has_value() && other.data_.has_value()) {
      if (data_.value().size() != other.data_.value().size()) return false;

      for (size_t i = 0; i < data_.value().size(); ++i) {
        DimExpr dim0 = symbol::SimplifyDimExpr(data_.value()[i]);
        DimExpr dim1 = symbol::SimplifyDimExpr(other.data_.value()[i]);
        if (dim0 != dim1) return false;
      }
    }

    for (size_t i = 0; i < shape_.size(); ++i) {
      DimExpr dim0 = symbol::SimplifyDimExpr(shape_[i]);
      DimExpr dim1 = symbol::SimplifyDimExpr(other.shape_[i]);
      if (dim0 != dim1) return false;
    }

    return true;
  }

  bool operator!=(const ShapeOrData<T>& other) const {
    return !(*this == other);
  }

 private:
  std::vector<T> shape_;
  std::optional<std::vector<T>> data_;
};

using TensorShapeOrDataDimExprs = ShapeOrData<DimExpr>;
using TensorListShapeOrDataDimExprs = std::vector<TensorShapeOrDataDimExprs>;
using ShapeOrDataDimExprsBase =
    std::variant<TensorShapeOrDataDimExprs, TensorListShapeOrDataDimExprs>;

class ShapeOrDataDimExprs : public ShapeOrDataDimExprsBase {
 public:
  ShapeOrDataDimExprs() = delete;
  ShapeOrDataDimExprs(
      const TensorShapeOrDataDimExprs& tensor_dim_exprs)  // NOLINT
      : ShapeOrDataDimExprsBase(tensor_dim_exprs) {}
  ShapeOrDataDimExprs(
      const TensorListShapeOrDataDimExprs& tensor_list_dim_exprs)
      : ShapeOrDataDimExprsBase(tensor_list_dim_exprs) {}

  template <typename T>
  bool isa() const {
    return std::holds_alternative<T>(*this);
  }

  template <typename T>
  const T& dyn_cast() const {
    return std::get<T>(*this);
  }

  const ShapeOrDataDimExprsBase& variant() const {
    return static_cast<const ShapeOrDataDimExprsBase&>(*this);
  }

  bool operator==(const ShapeOrDataDimExprs& other) const {
    return this->variant() == other.variant();
  }

  bool operator!=(const ShapeOrDataDimExprs& other) const {
    return !(*this == other);
  }

  const std::vector<DimExpr>& shape() const {
    IR_ENFORCE(
        std::holds_alternative<TensorShapeOrDataDimExprs>(*this),
        "Shape of ShapeOrData is not a vector, check wheather the value is a "
        "tensor-list or not.");
    return std::get<TensorShapeOrDataDimExprs>(*this).shape();
  }

  const std::optional<std::vector<DimExpr>>& data() const {
    IR_ENFORCE(
        std::holds_alternative<TensorShapeOrDataDimExprs>(*this),
        "Data of ShapeOrData is not a vector, check wheather the value is a "
        "tensor-list or not.");
    return std::get<TensorShapeOrDataDimExprs>(*this).data();
  }

  void SetData(const std::vector<DimExpr>& data) {
    IR_ENFORCE(
        std::holds_alternative<TensorShapeOrDataDimExprs>(*this),
        "Data of ShapeOrData is not a vector, check wheather the value is a "
        "tensor-list or not.");

    std::get<TensorShapeOrDataDimExprs>(*this).SetData(data);
  }
};

IR_API std::ostream& operator<<(std::ostream&,
                                const ShapeOrDataDimExprs& dim_expr);
}  // namespace symbol

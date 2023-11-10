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

#include <algorithm>
#include <vector>

#include "paddle/phi/core/tensor_base.h"
#include "paddle/pir/core/cast_utils.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/type.h"

namespace pir {

class ShapedTypeInterface : public TypeInterfaceBase<ShapedTypeInterface> {
 public:
  using DDim = phi::DDim;
  using DataType = Type;
  struct Concept {
    /// Defined these methods with the interface.
    explicit Concept(DataType (*get_element_type)(Type),
                     DDim (*get_shape)(Type))
        : get_element_type(get_element_type), get_shape(get_shape) {}

    DataType (*get_element_type)(Type);
    DDim (*get_shape)(Type);
  };

  template <class ConcreteType>
  struct Model : public Concept {
    static inline DataType GetElementType(Type type) {
      return pir::cast<ConcreteType>(type).dtype();
    }

    static inline DDim GetShape(Type type) {
      return pir::cast<ConcreteType>(type).dims();
    }

    Model() : Concept(GetElementType, GetShape) {}
  };

  ///
  /// \brief kDynamic
  ///
  static constexpr int64_t kDynamic = std::numeric_limits<int64_t>::min();

  ShapedTypeInterface(Type type, Concept *impl)
      : TypeInterfaceBase<ShapedTypeInterface>(type), impl_(impl) {
    dy_shape_ = vectorize(impl->get_shape(type));
    std::replace(dy_shape_.begin(), dy_shape_.end(), (int64_t)-1, kDynamic);
  }

  ///
  /// \brief Get the element type.
  ///
  DataType GetElementType() const;

  ///
  /// \brief Get the shape of this type.
  ///
  std::vector<int64_t> GetDyShape() const;

  ///
  /// \brief Check whether this type is ranked, currently return true.
  ///
  bool HasRank() const { return true; }

  ///
  /// If this is a ranked type, return the rank. Otherwise, abort.
  ///
  int64_t GetRank() const {
    IR_ENFORCE((*this).HasRank(), "Cannot query rank of unranked shaped type.");
    return (*this).GetDyShape().size();
  }

  ///
  /// \brief Check whether the given dimension size is a dynamic dimension.
  ///
  static constexpr bool IsDynamic(int64_t dValue) {
    return dValue == kDynamic || dValue == -1;
  }

  ///
  /// \brief Check whether the given shape has any size indicating a dynamic
  /// dimension.
  ///
  bool IsDynamicShape() const {
    return std::any_of(
        dy_shape_.begin(), dy_shape_.end(), [](int64_t size_value) {
          return IsDynamic(size_value);
        });
  }

  ///
  /// \brief Check whether shape has any size indicating a dynamic dimension.
  ///
  bool HasStaticShape() const { return (*this).HasRank() && !IsDynamicShape(); }

  ///
  /// \brief Check whether the given dimension has a dynamic size.Aborts for
  /// unranked types.
  ///
  bool IsDynamicDim(unsigned idx) const {
    IR_ENFORCE(idx < GetRank(), "Invalid index for shaped type.");
    return ShapedTypeInterface::IsDynamic(dy_shape_[idx]);
  }

  ///
  /// \brief Get the number of dimensions with dynamic size for a ranked type.
  /// Aborts for unranked types.
  ///
  int64_t GetNumDynamicDims() const {
    return std::count_if(
        dy_shape_.begin(), dy_shape_.end(), ShapedTypeInterface::IsDynamic);
  }

  ///
  /// \brief Get the size of the specified dimension for a ranked type. Aborts
  /// for unranked types.
  ///
  int64_t GetDimSize(unsigned idx) const {
    IR_ENFORCE(idx < GetRank(), "Invalid index for shaped type.");
    return dy_shape_[idx];
  }

 private:
  Concept *impl_;
  std::vector<int64_t> dy_shape_;
};

}  // namespace pir

IR_DECLARE_EXPLICIT_TYPE_ID(pir::ShapedTypeInterface)

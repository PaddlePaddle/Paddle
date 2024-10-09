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

#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/cast_utils.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/type.h"

namespace pir {

class IR_API ShapedTypeInterface
    : public TypeInterfaceBase<ShapedTypeInterface> {
 public:
  using DDim = pir::DDim;
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
  static constexpr int64_t kDynamic = std::int64_t(-1);

  ShapedTypeInterface(Type type, Concept *impl)
      : TypeInterfaceBase<ShapedTypeInterface>(type), impl_(impl) {}

  ///
  /// \brief Get the element type.
  ///
  DataType GetElementType() const;

  ///
  /// \brief Get the shape of this type.
  ///
  pir::DDim GetShape() const;

  ///
  /// \brief Check whether this type is ranked, currently return true.
  ///
  bool HasRank() const { return true; }

  ///
  /// If this is a ranked type, return the rank. Otherwise, abort.
  ///
  int64_t GetRank() const {
    PADDLE_ENFORCE_EQ((*this).HasRank(),
                      true,
                      common::errors::InvalidArgument(
                          "Cannot query rank of unranked shaped type."));
    return (*this).GetShape().size();
  }

  ///
  /// \brief Check whether the given dimension size is a dynamic dimension.
  ///
  static constexpr bool IsDynamic(int64_t dValue) { return dValue == kDynamic; }

  ///
  /// \brief Check whether the given shape has any size indicating a dynamic
  /// dimension.
  ///
  bool IsDynamicShape() const {
    auto size_vec = common::vectorize(impl_->get_shape(*this));
    return std::any_of(size_vec.begin(), size_vec.end(), [](int64_t size_val) {
      return IsDynamic(size_val);
    });
  }

  ///
  /// \brief Check whether shape has any size indicating a dynamic dimension.
  ///
  bool IsStaticShape() const { return (*this).HasRank() && !IsDynamicShape(); }

  ///
  /// \brief Check whether the given dimension has a dynamic size.Aborts for
  /// unranked types.
  ///
  bool IsDynamicDim(unsigned idx) const {
    PADDLE_ENFORCE_LT(
        idx,
        GetRank(),
        common::errors::InvalidArgument("Invalid index for shaped type."));
    return ShapedTypeInterface::IsDynamic((*this).GetShape()[idx]);
  }

  ///
  /// \brief Get the number of dimensions with dynamic size for a ranked type.
  /// Aborts for unranked types.
  ///
  int64_t GetNumDynamicDims() const {
    auto shape_vec = vectorize((*this).GetShape());
    return std::count_if(
        shape_vec.begin(), shape_vec.end(), ShapedTypeInterface::IsDynamic);
  }

  ///
  /// \brief Get the size of the specified dimension for a ranked type. Aborts
  /// for unranked types.
  ///
  int64_t GetDimSize(unsigned idx) const {
    PADDLE_ENFORCE_LT(
        idx,
        GetRank(),
        common::errors::InvalidArgument("Invalid index for shaped type."));
    return (*this).GetShape()[idx];
  }

 private:
  Concept *impl_;
};

class IR_API WrapTypeInterface : public TypeInterfaceBase<WrapTypeInterface> {
 public:
  struct Concept {
    /// Defined these methods with the interface.
    explicit Concept(Type (*prim_type)(Type)) : prim_type(prim_type) {}
    Type (*prim_type)(Type);
  };

  template <class ConcreteType>
  struct Model : public Concept {
    static Type prim_type(Type type) {
      return pir::cast<ConcreteType>(type).prim_type();
    }
    Model() : Concept(prim_type) {}
  };

  WrapTypeInterface(Type type, Concept *impl)
      : TypeInterfaceBase<WrapTypeInterface>(type), impl_(impl) {}

  Type prim_type() { return impl_->prim_type(*this); }

 private:
  Concept *impl_;
};
}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ShapedTypeInterface)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::WrapTypeInterface)

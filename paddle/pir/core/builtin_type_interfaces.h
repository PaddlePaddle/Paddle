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

#include <vector>
#include "paddle/phi/core/tensor_base.h"
#include "paddle/pir/core/cast_utils.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/type.h"

namespace pir {

namespace detail {

template <typename RangeT>
constexpr auto begin_impl(RangeT &&range)
    -> decltype(std::begin(std::forward<RangeT>(range))) {
  return std::begin(std::forward<RangeT>(range));
}

template <typename RangeT>
constexpr auto end_impl(RangeT &&range)
    -> decltype(std::end(std::forward<RangeT>(range))) {
  return std::end(std::forward<RangeT>(range));
}

template <typename RangeT>
constexpr auto adl_begin(RangeT &&range)
    -> decltype(begin_impl(std::forward<RangeT>(range))) {
  return begin_impl(std::forward<RangeT>(range));
}

template <typename RangeT>
constexpr auto adl_end(RangeT &&range)
    -> decltype(end_impl(std::forward<RangeT>(range))) {
  return end_impl(std::forward<RangeT>(range));
}

template <typename R, typename UnaryPredicate>
bool any_of(R &&Range, UnaryPredicate P) {
  return std::any_of(adl_begin(Range), adl_end(Range), P);
}

template <typename R, typename UnaryPredicate>
auto count_if(R &&Range, UnaryPredicate P) {
  return std::count_if(adl_begin(Range), adl_end(Range), P);
}

}  // namespace detail

class ShapedTypeInterface : public TypeInterfaceBase<ShapedTypeInterface> {
 public:
  using DDim = phi::DDim;
  using DataType = Type;
  struct Concept {
    /// Defined these methods with the interface.
    explicit Concept(DataType (*get_element_type)(Type),
                     DDim (*get_shape)(Type))
        : get_element_type_(get_element_type), get_shape_(get_shape) {}

    DataType (*get_element_type_)(Type);
    DDim (*get_shape_)(Type);
  };

  template <class ConcreteType>
  struct Model : public Concept {
    static inline DataType getElementType(Type type) {
      return pir::cast<ConcreteType>(type).dtype();
    }

    static inline DDim getShape(Type type) {
      return pir::cast<ConcreteType>(type).dims();
    }

    Model() : Concept(getElementType, getShape) {}
  };

  /// Constructor
  ShapedTypeInterface(Type type, Concept *impl)
      : TypeInterfaceBase<ShapedTypeInterface>(type), impl_(impl) {}

  ///
  /// \brief Get the element type.
  ///
  DataType getElementType() const;

  ///
  /// \brief Get the shape of this type.
  ///
  DDim getShape() const;

  ///
  /// \brief kDynamic
  ///
  static constexpr int64_t kDynamic = std::numeric_limits<int64_t>::min();

  ///
  /// \brief Check whether this type is ranked, currently return true.
  ///
  bool hasRank() const { return true; }

  ///
  /// If this is a ranked type, return the rank. Otherwise, abort.
  ///
  int64_t getRank() const {
    IR_ENFORCE((*this).hasRank(), "Cannot query rank of unranked shaped type.");
    return (*this).getShape().size();
  }

  ///
  /// \brief Check whether the given dimension size is a dynamic dimension.
  ///
  static constexpr bool isDynamic(int64_t dValue) { return dValue == kDynamic; }

  ///
  /// \brief Check whether the given shape has any size indicating a dynamic
  /// dimension.
  ///
  static bool isDynamicShape(DDim dSizes) {
    return detail::any_of(vectorize(dSizes),
                          [](int64_t dSize) { return isDynamic(dSize); });
  }

  ///
  /// \brief Check whether shape has any size indicating a dynamic dimension.
  ///
  bool hasStaticShape() const {
    return (*this).hasRank() && !isDynamicShape((*this).getShape());
  }

  ///
  /// \brief Check whether the given dimension has a dynamic size.Aborts for
  /// unranked types.
  ///
  bool isDynamicDim(unsigned idx) const {
    IR_ENFORCE(idx < getRank(), "Invalid index for shaped type.");
    return ShapedTypeInterface::isDynamic((*this).getShape()[idx]);
  }

  ///
  /// \brief Get the number of dimensions with dynamic size for a ranked type.
  /// Aborts for unranked types.
  ///
  int64_t getNumDynamicDims() const {
    return detail::count_if(vectorize((*this).getShape()),
                            ShapedTypeInterface::isDynamic);
  }

  ///
  /// \brief Get the size of the specified dimension for a ranked type. Aborts
  /// for unranked types.
  ///
  int64_t getDimSize(unsigned idx) const {
    IR_ENFORCE(idx < getRank(), "Invalid index for shaped type.");
    return (*this).getShape()[idx];
  }

 private:
  Concept *impl_;
};

}  // namespace pir

IR_DECLARE_EXPLICIT_TYPE_ID(pir::ShapedTypeInterface)

// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <ATen/core/jit_type_base.h>

#include <c10/util/Exception.h>

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace c10 {

inline bool operator!=(const Type& lhs, const Type& rhs) {
  return !(lhs == rhs);
}

namespace detail {

// Lightweight runtime types used only by the compat schema parser.
class SchemaAtomicType final : public SharedType {
 public:
  static std::shared_ptr<SchemaAtomicType> create(TypeKind kind,
                                                  std::string repr) {
    return std::shared_ptr<SchemaAtomicType>(
        new SchemaAtomicType(kind, std::move(repr)));
  }

  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return repr_; }

 private:
  SchemaAtomicType(TypeKind kind, std::string repr)
      : SharedType(kind), repr_(std::move(repr)) {}

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return repr_;
  }

  std::string repr_;
};

class SchemaOptionalType final : public SharedType {
 public:
  static const TypeKind Kind = TypeKind::OptionalType;

  static std::shared_ptr<SchemaOptionalType> create(TypePtr elem) {
    return std::shared_ptr<SchemaOptionalType>(
        new SchemaOptionalType(std::move(elem)));
  }

  bool equals(const Type& rhs) const override {
    if (rhs.kind() != kind()) {
      return false;
    }
    const auto contained = rhs.containedTypes();
    return contained.size() == 1 && *elem_.front() == *contained.front();
  }

  std::string str() const override { return elem_.front()->str() + "?"; }

  at::ArrayRef<TypePtr> containedTypes() const override { return elem_; }

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    TORCH_CHECK(contained_types.size() == 1,
                "Optional type expects exactly one contained type");
    return create(std::move(contained_types.front()));
  }

 private:
  explicit SchemaOptionalType(TypePtr elem)
      : SharedType(Kind), elem_{std::move(elem)} {}

  std::string annotation_str_impl(
      const TypePrinter& printer = nullptr) const override {
    return "Optional[" + elem_.front()->annotation_str(printer) + "]";
  }

  std::vector<TypePtr> elem_;
};

class SchemaTupleType final : public SharedType {
 public:
  static const TypeKind Kind = TypeKind::TupleType;

  static std::shared_ptr<SchemaTupleType> create(
      std::vector<TypePtr> elements) {
    return std::shared_ptr<SchemaTupleType>(
        new SchemaTupleType(std::move(elements)));
  }

  bool equals(const Type& rhs) const override {
    if (rhs.kind() != kind()) {
      return false;
    }
    const auto rhs_elems = rhs.containedTypes();
    if (rhs_elems.size() != elements_.size()) {
      return false;
    }
    for (size_t i = 0; i < elements_.size(); ++i) {
      if (*elements_[i] != *rhs_elems[i]) {
        return false;
      }
    }
    return true;
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < elements_.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << elements_[i]->str();
    }
    ss << ")";
    return ss.str();
  }

  at::ArrayRef<TypePtr> containedTypes() const override { return elements_; }

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types));
  }

 private:
  explicit SchemaTupleType(std::vector<TypePtr> elements)
      : SharedType(Kind), elements_(std::move(elements)) {}

  std::string annotation_str_impl(
      const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "Tuple[";
    for (size_t i = 0; i < elements_.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << elements_[i]->annotation_str(printer);
    }
    ss << "]";
    return ss.str();
  }

  std::vector<TypePtr> elements_;
};

}  // namespace detail

inline TypePtr makeSchemaAtomicType(TypeKind kind, std::string repr) {
  return detail::SchemaAtomicType::create(kind, std::move(repr));
}

inline TypePtr makeSchemaOptionalType(TypePtr elem) {
  return detail::SchemaOptionalType::create(std::move(elem));
}

inline TypePtr makeSchemaTupleType(std::vector<TypePtr> elements) {
  return detail::SchemaTupleType::create(std::move(elements));
}

struct TensorType;
using TensorTypePtr = SingletonTypePtr<TensorType>;
struct PADDLE_API TensorType : public Type {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return "Tensor"; }

  bool isInferredType() const { return is_inferred_; }

  static const TypeKind Kind = TypeKind::TensorType;

  static TensorTypePtr get() {
    static TensorType value(/*inferred=*/false);
    return TensorTypePtr(&value);
  }

  static TensorTypePtr getInferred() {
    static TensorType value(/*inferred=*/true);
    return TensorTypePtr(&value);
  }

 private:
  explicit TensorType(bool inferred)
      : Type(TypeKind::TensorType), is_inferred_(inferred) {}

  bool is_inferred_;
};

struct NumberType;
using NumberTypePtr = SingletonTypePtr<NumberType>;
struct PADDLE_API NumberType : public Type {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    return rhs.kind() == TypeKind::NumberType ||
           Type::isSubtypeOfExt(rhs, why_not);
  }

  std::string str() const override { return "Scalar"; }

  static const TypeKind Kind = TypeKind::NumberType;

  static NumberTypePtr get() {
    static NumberType value;
    return NumberTypePtr(&value);
  }

 protected:
  explicit NumberType(TypeKind kind = TypeKind::NumberType) : Type(kind) {}

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "number";
  }
};

struct FloatType;
using FloatTypePtr = SingletonTypePtr<FloatType>;
struct PADDLE_API FloatType : public NumberType {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return "float"; }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    return rhs.kind() == TypeKind::NumberType ||
           Type::isSubtypeOfExt(rhs, why_not);
  }

  static const TypeKind Kind = TypeKind::FloatType;

  static FloatTypePtr get() {
    static FloatType value;
    return FloatTypePtr(&value);
  }

 private:
  FloatType() : NumberType(TypeKind::FloatType) {}

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "float";
  }
};

struct IntType;
using IntTypePtr = SingletonTypePtr<IntType>;
struct PADDLE_API IntType : public NumberType {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return "int"; }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    return rhs.kind() == TypeKind::NumberType ||
           Type::isSubtypeOfExt(rhs, why_not);
  }

  static const TypeKind Kind = TypeKind::IntType;

  static IntTypePtr get() {
    static IntType value;
    return IntTypePtr(&value);
  }

 private:
  IntType() : NumberType(TypeKind::IntType) {}

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "int";
  }
};

struct BoolType;
using BoolTypePtr = SingletonTypePtr<BoolType>;
struct PADDLE_API BoolType : public Type {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return "bool"; }

  static const TypeKind Kind = TypeKind::BoolType;

  static BoolTypePtr get() {
    static BoolType value;
    return BoolTypePtr(&value);
  }

 private:
  BoolType() : Type(TypeKind::BoolType) {}
};

struct StringType;
using StringTypePtr = SingletonTypePtr<StringType>;
struct PADDLE_API StringType : public Type {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return annotation_str(); }

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "str";
  }

  static const TypeKind Kind = TypeKind::StringType;

  static StringTypePtr get() {
    static StringType value;
    return StringTypePtr(&value);
  }

 private:
  StringType() : Type(TypeKind::StringType) {}
};

struct NoneType;
using NoneTypePtr = SingletonTypePtr<NoneType>;
struct PADDLE_API NoneType : public Type {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return "NoneType"; }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    return rhs.kind() == TypeKind::OptionalType ||
           Type::isSubtypeOfExt(rhs, why_not);
  }

  static const TypeKind Kind = TypeKind::NoneType;

  static NoneTypePtr get() {
    static NoneType value;
    return NoneTypePtr(&value);
  }

 private:
  NoneType() : Type(TypeKind::NoneType) {}
};

struct DeviceObjType;
using DeviceObjTypePtr = SingletonTypePtr<DeviceObjType>;
struct PADDLE_API DeviceObjType : public Type {
  bool equals(const Type& rhs) const override { return rhs.kind() == kind(); }

  std::string str() const override { return "Device"; }

  static const TypeKind Kind = TypeKind::DeviceObjType;

  static DeviceObjTypePtr get() {
    static DeviceObjType value;
    return DeviceObjTypePtr(&value);
  }

 private:
  DeviceObjType() : Type(TypeKind::DeviceObjType) {}
};

inline std::ostream& operator<<(std::ostream& out, const Type& t) {
  out << t.str();
  return out;
}

}  // namespace c10

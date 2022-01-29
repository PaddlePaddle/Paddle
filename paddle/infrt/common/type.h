// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>

#include <memory>
#include <string>

#include "paddle/infrt/common/macros.h"

//! Much of the concepts are borrowed from Halide project.

namespace infrt {
namespace common {

/**
 * Types in the INFRT type system. They can be ints, unsigned ints, or floats of
 * various bit-widths.
 * They can also be vectors of the same (by setting the `lanes` field to
 * something larger than one).
 * NOTE: Front-end code other than vectorize shouldn't use vector types.
 */
struct Type {
  enum class type_t {
    Unk = -1,
    Int,
    UInt,
    Float,
    String,
    Void,
    // stupid idea to mix the Customized with other primitive types, large
    // refactor needs here.
    Customized,  // Customized type
  };

  //! type decorators in C++, the different code can used together.
  enum class cpp_type_t : uint8_t {
    None = 0,               // None information.
    Const = 1,              // const.
    Handle = 1 << 1,        // pointer type, such as `infrt_buffer_t*`.
    HandleHandle = 1 << 2,  // pointer of pointer, such as `infrt_buffer_t**`.
  };

  Type();
  Type(type_t t, int b, int w);
  Type(const Type& other);
  explicit Type(Type&& other);
  Type& operator=(const Type& other);

  INFRT_NODISCARD bool is_primitive() const;
  INFRT_NODISCARD bool is_customized() const;
  INFRT_NODISCARD bool valid() const;

  //! Some helper functions to check a type.
  // @{
  INFRT_NODISCARD bool is_unk() const;
  INFRT_NODISCARD bool is_void() const;
  INFRT_NODISCARD bool is_bool() const;
  INFRT_NODISCARD bool is_vector() const;
  INFRT_NODISCARD bool is_scalar() const;
  INFRT_NODISCARD bool is_float(int bits = -1) const;
  INFRT_NODISCARD bool is_int(int bits = -1) const;
  INFRT_NODISCARD bool is_integer(int bits = -1) const;
  INFRT_NODISCARD bool is_uint(int bits = -1) const;
  INFRT_NODISCARD bool is_string() const;
  INFRT_NODISCARD bool is_index_type();
  // @}

  Type& set_cpp_handle(bool x = true);
  INFRT_NODISCARD bool is_cpp_handle() const;

  Type& set_cpp_handle2(bool x = true);
  INFRT_NODISCARD bool is_cpp_handle2() const;

  Type& set_cpp_const(bool is_const = true);
  INFRT_NODISCARD bool is_cpp_const() const;

  Type& set_customized_type(const std::string& t);
  const std::string& customized_type() const;
  INFRT_NODISCARD bool is_customized_type() const;

  // Get a new type with bits set to \p x.
  Type with_bits(int x) const;
  // Get a new type with type set to \p x.
  Type with_type(type_t x) const;
  // Get a new type with lanes set to \p x.
  Type with_lanes(int x) const;
  // Get a new type with cpp_const set to \p x.
  Type with_cpp_const(bool x = true) const;

  //! Getters
  // @{
  type_t type() const;
  int bits() const;
  int lanes() const;
  cpp_type_t cpp_type() const;
  // @}

  //! Compare two types for equality.
  bool operator==(const Type& other) const;

  //! Compare two types for inequality.
  bool operator!=(const Type& other) const { return !(*this == other); }

  //! Generate a vector of this type, with `w` elements.
  Type VectorOf(int w) const;
  //! Generate a element type of this type.
  Type ElementOf() const;
  //! Generate the address type.
  Type PointerOf() const;
  //! Ignore const.
  Type IgnoreConst() const;
  //! Add const.
  Type ConstOf() const;

  friend std::ostream& operator<<(std::ostream& os, const Type& t);

  ~Type();

 private:
  void CheckTypeValid() const;

  struct Storage;
  Storage& GetStorage();
  const Storage& GetStorage() const;

  std::unique_ptr<Storage> storage_;
};  // namespace common

inline Type Void() { return Type(Type::type_t::Void, 1, 0); }
inline Type Int(int bits, int lanes = 1) {
  return Type(Type::type_t::Int, bits, lanes);
}
inline Type UInt(int bits, int lanes = 1) {
  return Type(Type::type_t::UInt, bits, lanes);
}
inline Type Float(int bits, int lanes = 1) {
  return Type(Type::type_t::Float, bits, lanes);
}
inline Type Bool(int lanes = 1) { return Type(Type::type_t::UInt, 1, lanes); }
inline Type String() { return Type(Type::type_t::String, 1, 1); }

//! Builtin native types as global singletons.
// @{
const Type& F16();
const Type& F32();
const Type& F64();
const Type& I8();
const Type& I16();
const Type& I32();
const Type& I64();
const Type& UI8();
const Type& UI16();
const Type& UI32();
const Type& UI64();
const Type& I1();
const Type& UI1();
// @}

template <typename T>
Type type_of();

// clang-format off
template <> inline Type type_of<float>() { return F32(); }
template <> inline Type type_of<double>() { return F64(); }
template <> inline Type type_of<unsigned char>() { return UI8(); }
template <> inline Type type_of<int16_t>() { return UI16(); }
template <> inline Type type_of<int32_t>() { return I32(); }
template <> inline Type type_of<uint32_t>() { return UI32(); }
template <> inline Type type_of<bool>() { return UI1(); }
template <> inline Type type_of<char>() { return I8(); }
template <> inline Type type_of<int64_t>() { return I64(); }
template <> inline Type type_of<uint64_t>() { return UI64(); }
template <> inline Type type_of<signed char>() { return I8(); }
template <> inline Type type_of<void>() { return Void(); }
// clang-format on
template <>
inline Type type_of<int8_t*>() {
  Type x = Int(8);
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<void*>() {
  Type x = type_of<void>();
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<void**>() {
  Type x = type_of<void>();
  x.set_cpp_handle2();
  return x;
}
template <>
inline Type type_of<float*>() {
  Type x = type_of<float>();
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<double*>() {
  Type x = type_of<double>();
  x.set_cpp_handle();
  return x;
}

std::ostream& operator<<(std::ostream& os, Type::type_t t);

}  // namespace common
}  // namespace infrt

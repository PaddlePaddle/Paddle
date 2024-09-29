// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"
#include "paddle/cinn/common/float16_bfloat16_utils.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/common/enforce.h"
//! Much of the concepts are borrowed from Halide project.

namespace cinn {
namespace common {

/**
 * Types in the CINN type system. They can be ints, unsigned ints, or floats of
 * various bit-widths. They can also be vectors of the same (by setting the
 * `lanes` field to something larger than one). NOTE: Front-end code other than
 * vectorize shouldn't use vector types.
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

  // CINN use type_t and bits to distinguish data types, like is_float(64) for
  // double, is_float(32) for float, but for Float16 and BFloat16, the bits are
  // both 16, so we need some other info to distinguish them.
  enum class specific_type_t {
    // None for some cases we only care about the bits, e.g. vectorize for
    // hardwares
    None = -1,
    FP16,
    BF16,
    // for FP8 in future
    // E5M2,
    // E4M3,
  };

  //! type decorators in C++, the different code can used together.
  enum class cpp_type_t : uint8_t {
    None = 0,               // None information.
    Const = 1,              // const.
    Handle = 1 << 1,        // pointer type, such as `cinn_buffer_t*`.
    HandleHandle = 1 << 2,  // pointer of pointer, such as `cinn_buffer_t**`.
  };

  Type();
  Type(type_t t, int b, int w, specific_type_t st = specific_type_t::None);
  Type(const Type& other);
  explicit Type(Type&& other);
  Type& operator=(const Type& other);

  CINN_NODISCARD bool is_primitive() const;
  CINN_NODISCARD bool is_customized() const;
  CINN_NODISCARD bool valid() const;

  //! Some helper functions to check a type.
  // @{
  CINN_NODISCARD bool is_unk() const;
  CINN_NODISCARD bool is_void() const;
  CINN_NODISCARD bool is_bool() const;
  CINN_NODISCARD bool is_vector() const;
  CINN_NODISCARD bool is_scalar() const;
  CINN_NODISCARD bool is_float(
      int bits = -1, specific_type_t st = specific_type_t::None) const;
  CINN_NODISCARD bool is_float16() const;
  CINN_NODISCARD bool is_bfloat16() const;
  CINN_NODISCARD bool is_int(int bits = -1) const;
  CINN_NODISCARD bool is_integer(int bits = -1) const;
  CINN_NODISCARD bool is_uint(int bits = -1) const;
  CINN_NODISCARD bool is_string() const;
  CINN_NODISCARD bool is_index_type();
  // @}

  Type& set_cpp_handle(bool x = true);
  CINN_NODISCARD bool is_cpp_handle() const;

  Type& set_cpp_handle2(bool x = true);
  CINN_NODISCARD bool is_cpp_handle2() const;

  Type& set_cpp_const(bool is_const = true);
  CINN_NODISCARD bool is_cpp_const() const;

  Type& set_customized_type(const std::string& t);
  const std::string& customized_type() const;
  CINN_NODISCARD bool is_customized_type() const;

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
  specific_type_t specific_type() const;
  int bits() const;
  int lanes() const;
  cpp_type_t cpp_type() const;
  int bytes() const;
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
  //! Check if a dtype is supported in CINN yet.
  bool is_supported() const;

  std::string to_string() const;

  friend std::ostream& operator<<(std::ostream& os, const Type& t);

  ~Type();

 private:
  void CheckTypeValid() const;

  struct Storage;
  Storage& GetStorage();
  const Storage& GetStorage() const;

  std::unique_ptr<Storage> storage_;
};  // namespace common

inline Type Void() { return Type(Type::type_t ::Void, 1, 0); }
inline Type Int(int bits, int lanes = 1) {
  return Type(Type::type_t ::Int, bits, lanes);
}
inline Type UInt(int bits, int lanes = 1) {
  return Type(Type::type_t ::UInt, bits, lanes);
}
inline Type BFloat16(int lanes = 1) {
  return Type(Type::type_t ::Float, 16, lanes, Type::specific_type_t::BF16);
}
inline Type Float16(int lanes = 1) {
  return Type(Type::type_t ::Float, 16, lanes, Type::specific_type_t::FP16);
}
inline Type Float(int bits,
                  int lanes = 1,
                  Type::specific_type_t st = Type::specific_type_t::None) {
  if (bits == 16) {
    PADDLE_ENFORCE_EQ((st == Type::specific_type_t::FP16 ||
                       st == Type::specific_type_t::BF16),
                      true,
                      ::common::errors::InvalidArgument(
                          "When creating a 16-bit Float, the specific_type_t "
                          "must be FP16 or BF16."));
  }
  return Type(Type::type_t ::Float, bits, lanes, st);
}
inline Type Bool(int lanes = 1) { return Type(Type::type_t ::UInt, 1, lanes); }
inline Type String() { return Type(Type::type_t::String, 1, 1); }

//! Builtin native types as global singletons.
// @{
const Type& BF16();
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
template <> inline Type type_of<void>() { return Void(); }

template <> inline Type type_of<bfloat16>() { return BF16(); }
template <> inline Type type_of<float16>() { return F16(); }
template <> inline Type type_of<float>() { return F32(); }
template <> inline Type type_of<double>() { return F64(); }

template <> inline Type type_of<bool>() { return UI1(); }
template <> inline Type type_of<char>() { return I8(); }
// template <> inline Type type_of<signed char>() { return I8(); }
// template <> inline Type type_of<unsigned char>() { return UI8(); }
template <> inline Type type_of<std::string>() { return String(); }

template <> inline Type type_of<int8_t>() { return I8(); }
template <> inline Type type_of<int16_t>() { return I16(); }
template <> inline Type type_of<int32_t>() { return I32(); }
template <> inline Type type_of<int64_t>() { return I64(); }

template <> inline Type type_of<uint8_t>() { return UI8(); }
template <> inline Type type_of<uint16_t>() { return UI16(); }
template <> inline Type type_of<uint32_t>() { return UI32(); }
template <> inline Type type_of<uint64_t>() { return UI64(); }

// clang-format on
template <>
inline Type type_of<int8_t*>() {
  Type x = Int(8);
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<uint8_t*>() {
  Type x = UInt(8);
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<int32_t*>() {
  Type x = Int(32);
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<int32_t**>() {
  Type x = Int(32);
  x.set_cpp_handle2();
  return x;
}
template <>
inline Type type_of<int64_t**>() {
  Type x = Int(64);
  x.set_cpp_handle2();
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
inline Type type_of<bfloat16*>() {
  Type x = type_of<float16>();
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<float16*>() {
  Type x = type_of<float16>();
  x.set_cpp_handle();
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

namespace customized_type {

static const char* kArgs_type_repr = "Args";
static const char* kArgValue_type_repr = "ArgValue";
static const char* kbuffer_t = "cinn_buffer_t";
static const char* kpod_value_t = "cinn_pod_value_t";
static const char* kcuda_builtin_vector_t = "CudaVectorType::";

}  // namespace customized_type

template <>
inline Type type_of<cinn_buffer_t>() {
  return Type().set_customized_type(customized_type::kbuffer_t);
}
template <>
inline Type type_of<cinn_buffer_t*>() {
  return Type()
      .set_customized_type(customized_type::kbuffer_t)
      .set_cpp_handle();
}
template <>
inline Type type_of<const cinn_buffer_t*>() {
  return Type()
      .set_customized_type(customized_type::kbuffer_t)
      .set_cpp_handle()
      .set_cpp_const();
}
template <>
inline Type type_of<cinn_pod_value_t>() {
  return Type().set_customized_type(customized_type::kpod_value_t);
}
template <>
inline Type type_of<cinn_pod_value_t*>() {
  return Type()
      .set_customized_type(customized_type::kpod_value_t)
      .set_cpp_handle();
}

Type Str2Type(const std::string& type);

std::string Type2Str(const Type& type);

enum class Layout {
  kUnk = 0,
  kNCHW,
  kNHWC,
};

}  // namespace common
}  // namespace cinn

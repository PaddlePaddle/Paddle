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

#include "paddle/infrt/common/type.h"

#include <utility>

namespace infrt {
namespace common {

struct Type::Storage {
  Storage() = default;
  Storage(type_t t, int b, int w) : type_(t), bits_(b), lanes_(w) {}

  type_t type_{type_t::Unk};
  cpp_type_t cpp_type_{cpp_type_t::None};

  //! How many bits per element.
  int bits_{};

  //! How many elements(if a vector type), for scalar types, it should be 1.
  int lanes_{1};

  //! Name of the customized type.
  std::string customized_type_;
};

Type::~Type() {}

std::ostream &operator<<(std::ostream &os, const Type &t) {
  if (t.is_cpp_const()) os << "const ";
  switch (t.type()) {
    case Type::type_t::Int:
      if (t.bits() == 1) {
        os << "bool";
      } else {
        os << "int" << t.bits();
      }

      break;
    case Type::type_t::UInt:
      os << "uint" << t.bits();
      break;

    case Type::type_t::Float:
      os << "float" << t.bits();
      break;
    case Type::type_t::Void:
      os << "void";
      break;
    case Type::type_t::Customized:
      os << t.customized_type();
      break;
    case Type::type_t::String:
      os << "string";
      break;
    case Type::type_t::Unk:
      os << "unk";
      break;
  }

  if (t.lanes() > 1) os << "<" << t.lanes() << ">";
  if (t.is_cpp_handle()) os << "*";
  if (t.is_cpp_handle2()) os << "**";

  return os;
}

std::ostream &operator<<(std::ostream &os, Type::type_t t) {
  switch (t) {
    case Type::type_t::String:
      os << "String";
      break;
    case Type::type_t::Void:
      os << "Void";
      break;
    case Type::type_t::UInt:
      os << "UInt";
      break;
    case Type::type_t::Int:
      os << "Int";
      break;
    case Type::type_t::Float:
      os << "Float";
      break;
    case Type::type_t::Unk:
      os << "Unk";
      break;
    case Type::type_t::Customized:
      os << "Customized";
  }
  return os;
}

Type &Type::set_cpp_handle(bool x) {
  // unset the other handle-related bits.
  set_cpp_handle2(false);

  auto &v = (*reinterpret_cast<uint8_t *>(&GetStorage().cpp_type_));
  // unset the other handle-related bits.
  v &= ~static_cast<uint8_t>(cpp_type_t::Handle);
  v &= ~static_cast<uint8_t>(cpp_type_t::HandleHandle);

  if (x)
    v |= static_cast<uint8_t>(cpp_type_t::Handle);
  else
    v &= ~static_cast<uint8_t>(cpp_type_t::Handle);

  return *this;
}

Type &Type::set_cpp_handle2(bool x) {
  auto &v = (*reinterpret_cast<uint8_t *>(&GetStorage().cpp_type_));

  // unset the other handle-related bits.
  v &= ~static_cast<uint8_t>(cpp_type_t::Handle);
  v &= ~static_cast<uint8_t>(cpp_type_t::HandleHandle);

  if (x)
    v |= static_cast<uint8_t>(cpp_type_t::HandleHandle);
  else
    v &= ~static_cast<uint8_t>(cpp_type_t::HandleHandle);

  return *this;
}

Type Type::VectorOf(int w) const {
  CheckTypeValid();
  return Type(type(), w, bits());
}

Type::Type(const Type &other) {
  if (other.storage_) storage_.reset(new Storage(*other.storage_));
}

Type Type::ElementOf() const {
  CheckTypeValid();
  auto type = *this;
  type.storage_->lanes_ = 1;
  return type;
}

void Type::CheckTypeValid() const { CHECK_NE(GetStorage().type_, type_t::Unk); }

Type Type::PointerOf() const {
  CheckTypeValid();
  auto x = *this;
  CHECK(!x.is_cpp_handle2()) << "Not support three level of PointerOf";
  if (x.is_cpp_handle())
    x.set_cpp_handle2();
  else
    x.set_cpp_handle();
  return x;
}

Type Type::ConstOf() const {
  CheckTypeValid();
  auto x = *this;
  x.set_cpp_const();
  return x;
}

Type Type::IgnoreConst() const {
  CheckTypeValid();
  auto x = *this;
  x.set_cpp_const(false);
  return x;
}

Type Type::with_bits(int x) const {
  CHECK(is_primitive());
  Type type = *this;
  type.GetStorage().bits_ = x;
  return type;
}

Type Type::with_type(Type::type_t x) const {
  Type type = *this;
  type.GetStorage().type_ = x;
  return type;
}

Type Type::with_lanes(int x) const {
  CHECK(valid());
  Type type = *this;
  type.GetStorage().lanes_ = x;
  return type;
}

Type Type::with_cpp_const(bool x) const {
  Type type = *this;
  type.set_cpp_const(x);
  return type;
}

Type &Type::set_cpp_const(bool is_const) {
  uint8_t &data = *reinterpret_cast<uint8_t *>(&GetStorage().cpp_type_);
  if (is_const) {
    data |= static_cast<uint8_t>(cpp_type_t::Const);
  } else {
    data &= ~(static_cast<uint8_t>(cpp_type_t::Const));
  }

  return *this;
}
Type &Type::set_customized_type(const std::string &t) {
  GetStorage().type_ = type_t::Customized;
  GetStorage().customized_type_ = t;

  return *this;
}

bool Type::valid() const {
  if (is_unk()) return false;
  if (is_customized()) {
    return !GetStorage().customized_type_.empty();
  }
  if (is_primitive()) {
    return bits() != 0;
  }
  return true;
}

Type::Type(Type::type_t t, int b, int w) : storage_(new Storage(t, b, w)) {}
bool Type::is_primitive() const {
  return !is_unk() && type() != type_t::Customized;
}
bool Type::is_customized() const {
  return !is_unk() && type() == type_t::Customized;
}
bool Type::is_unk() const { return type() == type_t::Unk; }
bool Type::is_bool() const { return type() == type_t::UInt && bits() == 1; }
bool Type::is_void() const { return type() == type_t::Void; }
bool Type::is_vector() const { return lanes() > 1; }
bool Type::is_scalar() const { return lanes() == 1; }
bool Type::is_float(int bits) const {
  return type() == type_t::Float && (bits < 0 || bits == this->bits());
}
bool Type::is_uint(int bits) const {
  return type() == type_t::UInt && (bits < 0 || bits == this->bits());
}
bool Type::is_int(int bits) const {
  return type() == type_t::Int && (bits < 0 || bits == this->bits());
}
bool Type::is_integer(int bits) const {
  return (type() == type_t::Int || type() == type_t::UInt) &&
         (bits < 0 || bits == this->bits());
}
bool Type::is_index_type() {
  return is_int() && lanes() == 1 && (bits() == 32 || bits() == 64);
}
bool Type::is_cpp_handle() const {
  return static_cast<uint8_t>(GetStorage().cpp_type_) &
         static_cast<uint8_t>(cpp_type_t::Handle);
}
bool Type::is_cpp_handle2() const {
  return static_cast<uint8_t>(GetStorage().cpp_type_) &
         static_cast<uint8_t>(cpp_type_t::HandleHandle);
}
bool Type::is_cpp_const() const {
  return static_cast<uint8_t>(cpp_type_t::Const) &
         static_cast<uint8_t>(GetStorage().cpp_type_);
}
const std::string &Type::customized_type() const {
  return GetStorage().customized_type_;
}
bool Type::is_customized_type() const {
  return !GetStorage().customized_type_.empty();
}
Type::type_t Type::type() const { return GetStorage().type_; }
int Type::bits() const { return GetStorage().bits_; }
int Type::lanes() const { return GetStorage().lanes_; }
Type::cpp_type_t Type::cpp_type() const { return GetStorage().cpp_type_; }
bool Type::operator==(const Type &other) const {
  return type() == other.type() && bits() == other.bits() &&
         lanes() == other.lanes() &&
         GetStorage().cpp_type_ == other.GetStorage().cpp_type_ &&
         customized_type() == other.customized_type();
}
bool Type::is_string() const { return type() == type_t::String; }

Type &Type::operator=(const Type &other) {
  if (other.storage_) storage_.reset(new Storage(*other.storage_));
  return *this;
}

Type::Storage &Type::GetStorage() { return *storage_; }
const Type::Storage &Type::GetStorage() const { return *storage_; }

Type::Type() : storage_(new Storage) {}
Type::Type(Type &&other) : storage_(std::move(other.storage_)) {}

const Type &F16() {
  static auto t = Float(16);
  return t;
}
const Type &F32() {
  static auto t = Float(32);
  return t;
}
const Type &F64() {
  static auto t = Float(64);
  return t;
}
const Type &I8() {
  static auto t = Int(8);
  return t;
}
const Type &I16() {
  static auto t = Int(16);
  return t;
}
const Type &I32() {
  static auto t = Int(32);
  return t;
}
const Type &I64() {
  static auto t = Int(64);
  return t;
}
const Type &UI8() {
  static auto t = UInt(8);
  return t;
}
const Type &UI16() {
  static auto t = UInt(16);
  return t;
}
const Type &UI32() {
  static auto t = UInt(32);
  return t;
}
const Type &UI64() {
  static auto t = UInt(64);
  return t;
}
const Type &I1() {
  static auto t = Int(1);
  return t;
}
const Type &UI1() {
  static auto t = UInt(1);
  return t;
}

}  // namespace common
}  // namespace infrt

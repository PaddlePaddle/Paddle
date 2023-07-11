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

#include "paddle/cinn/common/cinn_value.h"

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {

namespace ir {

class Expr;
class Var;

}  // namespace ir

namespace common {

//! Implement the type_code for all the supported types.
// @{
#define __m(T, code__)           \
  template <>                    \
  int CINNValue::TypeCode<T>() { \
    return code__;               \
  }
__m(std::nullptr_t, -1);
__m(char *, 20);  // start from a larger number to avoid duplicate id with
                  // cinn_pod_value_t
__m(char const *, 21);
__m(ir::Expr, 22);
__m(ir::Var, 23);
__m(CINNValuePack, 24);
__m(poly::StageMap, 25);
__m(std::string, 26);
#undef __m
//@}

//! Implement ToValue.
// @{
template <>
cinn_value_t ToValue<bool>(bool v) {
  cinn_value_t val;
  val.v_int64 = v;
  return val;
}
template <>
cinn_value_t ToValue<int>(int v) {
  cinn_value_t val;
  val.v_int64 = v;
  return val;
}
template <>
cinn_value_t ToValue<int64_t>(int64_t v) {
  cinn_value_t val;
  val.v_int64 = v;
  return val;
}
template <>
cinn_value_t ToValue<float>(float v) {
  cinn_value_t val;
  val.v_float64 = v;
  return val;
}
template <>
cinn_value_t ToValue<double>(double v) {
  cinn_value_t val;
  val.v_float64 = v;
  return val;
}
template <>
cinn_value_t ToValue<bfloat16>(bfloat16 v) {
  cinn_value_t val;
  val.v_float64 = static_cast<double>(v);
  return val;
}
template <>
cinn_value_t ToValue<float16>(float16 v) {
  cinn_value_t val;
  val.v_float64 = static_cast<double>(v);
  return val;
}
template <>
cinn_value_t ToValue<char *>(char *v) {
  cinn_value_t val;
  val.v_str = v;
  return val;
}
template <>
cinn_value_t ToValue<char const *>(char const *v) {
  cinn_value_t val;
  val.v_str = const_cast<char *>(v);
  return val;
}
// @}

bool CINNValue::is_string() const {
  return type_code_ == TypeCode<std::string>();
}

bool CINNValue::is_var() const { return type_code_ == TypeCode<ir::Var>(); }

bool CINNValue::is_expr() const {
  return type_code_ == TypeCode<ir::Expr>() &&
         !absl::any_cast<Expr>(shared_).as_tensor();
}

bool CINNValue::is_stagemap() const {
  return type_code_ == TypeCode<poly::StageMap>();
}

bool CINNValue::is_tensor() const {
  return type_code_ == TypeCode<ir::Expr>() &&
         absl::any_cast<Expr>(shared_).as_tensor();
}

CINNValue::operator std::string() const {
  CHECK_EQ(type_code_, TypeCode<std::string>());
  return absl::any_cast<std::string>(shared_);
}
CINNValue::operator ir::Var() const {
  CHECK_EQ(type_code_, TypeCode<ir::Var>());
  return absl::any_cast<ir::Var>(shared_);
}
CINNValue::operator ir::Expr() const {
  CHECK_EQ(type_code_, TypeCode<ir::Expr>());
  return absl::any_cast<Expr>(shared_);
}
CINNValue::operator CINNValuePack() const {
  CHECK_EQ(type_code_, TypeCode<CINNValuePack>());
  return absl::any_cast<CINNValuePack>(shared_);
}
CINNValue::operator poly::StageMap() const {
  CHECK_EQ(type_code(), TypeCode<poly::StageMap>());
  return absl::any_cast<poly::StageMap>(shared_);
}
CINNValue::CINNValue(char *value)
    : cinn_pod_value_t(ToValue(value), TypeCode<char *>()) {}

CINNValue::CINNValue(const std::string &value)
    : cinn_pod_value_t(cinn_value_t(), TypeCode<std::string>()) {
  shared_ = value;
}
CINNValue::CINNValue(const Var &value)
    : cinn_pod_value_t(cinn_value_t(), TypeCode<Var>()) {
  CHECK(value.defined());
  shared_ = value;
}
CINNValue::CINNValue(const Expr &value)
    : cinn_pod_value_t(cinn_value_t(), TypeCode<Expr>()) {
  CHECK(value.defined());
  shared_ = value;
}
CINNValue::CINNValue(const CINNValuePack &value)
    : cinn_pod_value_t(cinn_value_t(), TypeCode<CINNValuePack>()) {
  CHECK(value.defined());
  shared_ = value;
}
CINNValue::CINNValue(const poly::StageMap &value)
    : cinn_pod_value_t(cinn_value_t(), TypeCode<poly::StageMap>()) {
  CHECK(value.defined());
  shared_ = value;
}

CINNValuePack _CINNValuePack_::Make(const std::vector<CINNValue> &array) {
  auto *node = new _CINNValuePack_;
  for (auto &item : array) node->AddValue(item);
  return CINNValuePack(node);
}
CINNValue &_CINNValuePack_::operator[](int offset) {
  CHECK_LT(offset, size());
  return values_[offset];
}
const CINNValue &_CINNValuePack_::operator[](int offset) const {
  CHECK_LT(offset, size());
  return values_[offset];
}
void _CINNValuePack_::AddValue(const CINNValue &value) {
  CHECK(value.defined());
  values_.push_back(value);
}
void _CINNValuePack_::Clear() { values_.clear(); }
const char *_CINNValuePack_::type_info() const { return __type_info__; }

CINNValue &CINNValue::operator=(bool value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(int32_t value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(int64_t value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(float value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(double value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(bfloat16 value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(float16 value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(char *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(cinn_buffer_t *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(void *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const char *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const CINNValuePack &value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const std::string &value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const ir::Var &value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const ir::Expr &value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const poly::StageMap &value) {
  *this = CINNValue(value);
  return *this;
}

}  // namespace common
}  // namespace cinn

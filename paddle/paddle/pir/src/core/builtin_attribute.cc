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

#include "paddle/pir/include/core/builtin_attribute.h"

namespace pir {

bool BoolAttribute::data() const { return storage()->data(); }

float FloatAttribute::data() const { return storage()->data(); }

double DoubleAttribute::data() const { return storage()->data(); }

int32_t Int32Attribute::data() const { return storage()->data(); }

int64_t IndexAttribute::data() const { return storage()->data(); }

int64_t Int64Attribute::data() const { return storage()->data(); }

void* PointerAttribute::data() const { return storage()->data(); }

Type TypeAttribute::data() const { return storage()->data(); }

phi::dtype::complex<float> Complex64Attribute::data() const {
  return storage()->data();
}

phi::dtype::complex<double> Complex128Attribute::data() const {
  return storage()->data();
}

bool StrAttribute::operator<(const StrAttribute& right) const {
  return storage() < right.storage();
}
std::string StrAttribute::AsString() const { return storage()->AsString(); }

size_t StrAttribute::size() const { return storage()->size(); }

StrAttribute StrAttribute::get(pir::IrContext* ctx, const std::string& value) {
  return AttributeManager::get<StrAttribute>(ctx, value);
}

std::vector<Attribute> ArrayAttribute::AsVector() const {
  return storage()->AsVector();
}

size_t ArrayAttribute::size() const { return storage()->size(); }

bool ArrayAttribute::empty() const { return storage()->empty(); }

Attribute ArrayAttribute::at(size_t index) const {
  return storage()->at(index);
}
Attribute ArrayAttribute::operator[](size_t index) const {
  return storage()->operator[](index);
}

ArrayAttribute ArrayAttribute::get(IrContext* ctx,
                                   const std::vector<Attribute>& value) {
  return AttributeManager::get<ArrayAttribute>(ctx, value);
}

ArrayAttributeStorage::ArrayAttributeStorage(const ParamKey& key)
    : size_(key.size()) {
  constexpr size_t align = alignof(Attribute);
  if (align > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    data_ = static_cast<Attribute*>(
        ::operator new(size_ * sizeof(Attribute), std::align_val_t(align)));
  } else {
    data_ = static_cast<Attribute*>(::operator new(size_ * sizeof(Attribute)));
  }
  memcpy(data_, key.data(), sizeof(Attribute) * size_);
}

ArrayAttributeStorage::~ArrayAttributeStorage() {
  constexpr size_t align = alignof(Attribute);
  if (align > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    ::operator delete(data_, std::align_val_t(align));
  } else {
    ::operator delete(data_);
  }
}

bool TensorNameAttribute::operator<(const TensorNameAttribute& right) const {
  return storage() < right.storage();
}
std::string TensorNameAttribute::data() const { return storage()->AsString(); }

size_t TensorNameAttribute::size() const { return storage()->size(); }

TensorNameAttribute TensorNameAttribute::get(pir::IrContext* ctx,
                                             const std::string& tensor_name) {
  return AttributeManager::get<TensorNameAttribute>(ctx, tensor_name);
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::StrAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::BoolAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::FloatAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::DoubleAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Int32Attribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::IndexAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Int64Attribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ArrayAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::PointerAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::TypeAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::TensorNameAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Complex64Attribute)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Complex128Attribute)

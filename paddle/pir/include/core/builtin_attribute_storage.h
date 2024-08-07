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
#include <map>
#include <type_traits>

#include "paddle/common/enforce.h"
#include "paddle/phi/common/complex.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/attribute_base.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/utils.h"

namespace pir {

#define DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(ConcreteStorage, BaseType) \
  struct ConcreteStorage : public AttributeStorage {                   \
    using ParamKey = BaseType;                                         \
                                                                       \
    explicit ConcreteStorage(ParamKey key) { data_ = key; }            \
                                                                       \
    static ConcreteStorage *Construct(ParamKey key) {                  \
      return new ConcreteStorage(key);                                 \
    }                                                                  \
                                                                       \
    static size_t HashValue(ParamKey key) {                            \
      return std::hash<ParamKey>{}(key);                               \
    }                                                                  \
                                                                       \
    bool operator==(ParamKey key) const { return data_ == key; }       \
                                                                       \
    BaseType data() const { return data_; }                            \
                                                                       \
   private:                                                            \
    BaseType data_;                                                    \
  }

DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(BoolAttributeStorage, bool);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(FloatAttributeStorage, float);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(DoubleAttributeStorage, double);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(Int32AttributeStorage, int32_t);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(IndexAttributeStorage, int64_t);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(Int64AttributeStorage, int64_t);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(PointerAttributeStorage, void *);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(TypeAttributeStorage, Type);

///
/// \brief Define Parametric AttributeStorage for StrAttribute.
///
struct StrAttributeStorage : public AttributeStorage {
  using ParamKey = std::string;

  explicit StrAttributeStorage(const ParamKey &key) : size_(key.size()) {
    if (size_ > kLocalSize) {
      data_ = static_cast<char *>(::operator new(size_));
      memcpy(data_, key.c_str(), size_);
    } else {
      memcpy(buff_, key.c_str(), size_);
    }
  }

  ~StrAttributeStorage() {
    if (size_ > kLocalSize) ::operator delete(data_);
  }

  static StrAttributeStorage *Construct(const ParamKey &key) {
    return new StrAttributeStorage(key);
  }

  static size_t HashValue(const ParamKey &key) {
    return std::hash<std::string>{}(key);
  }

  bool operator==(const ParamKey &key) const {
    if (size_ != key.size()) return false;
    const char *data = size_ > kLocalSize ? data_ : buff_;
    return std::equal(data, data + size_, key.c_str());
  }

  // Note: The const char* is not end with '\0'.
  const char *data() const { return size_ > kLocalSize ? data_ : buff_; }
  size_t size() const { return size_; }
  std::string AsString() const { return std::string(data(), size_); }

 private:
  static constexpr size_t kLocalSize = sizeof(void *) / sizeof(char);
  union {
    char *data_;
    char buff_[kLocalSize];
  };
  const size_t size_;
};

struct ArrayAttributeStorage : public AttributeStorage {
  using ParamKey = std::vector<Attribute>;

  explicit ArrayAttributeStorage(const ParamKey &key);

  ~ArrayAttributeStorage();

  static ArrayAttributeStorage *Construct(const ParamKey &key) {
    return new ArrayAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey &key) {
    std::size_t hash_value = 0;
    for (size_t i = 0; i < key.size(); ++i) {
      hash_value =
          detail::hash_combine(hash_value, std::hash<Attribute>()(key[i]));
    }
    return hash_value;
  }

  bool operator==(const ParamKey &key) const {
    return key.size() == size_ && std::equal(key.begin(), key.end(), data_);
  }

  std::vector<Attribute> AsVector() const {
    return std::vector<Attribute>(data_, data_ + size_);
  }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0u; }

  Attribute at(size_t index) const {
    PADDLE_ENFORCE_LT(
        index,
        size_,
        common::errors::InvalidArgument(
            "The index (%d) must be less than size (%d).", index, size_));
    return data_[index];
  }
  Attribute operator[](size_t index) const { return data_[index]; }

 private:
  Attribute *data_;
  const size_t size_;
};

struct Complex64AttributeStorage : public AttributeStorage {
  using ParamKey = phi::dtype::complex<float>;
  explicit Complex64AttributeStorage(const ParamKey &key) { data_ = key; }
  static Complex64AttributeStorage *Construct(const ParamKey &key) {
    return new Complex64AttributeStorage(key);
  }
  static std::size_t HashValue(const ParamKey &key) {
    std::stringstream complex_str;
    complex_str << key.real << "+" << key.imag << "i";
    return std::hash<std::string>{}(complex_str.str());
  }

  bool operator==(ParamKey key) const { return data_ == key; }

  phi::dtype::complex<float> data() const { return data_; }

 private:
  phi::dtype::complex<float> data_;
};

struct Complex128AttributeStorage : public AttributeStorage {
  using ParamKey = phi::dtype::complex<double>;
  explicit Complex128AttributeStorage(const ParamKey &key) { data_ = key; }
  static Complex128AttributeStorage *Construct(const ParamKey &key) {
    return new Complex128AttributeStorage(key);
  }
  static std::size_t HashValue(const ParamKey &key) {
    std::stringstream complex_str;
    complex_str << key.real << "+" << key.imag << "i";
    return std::hash<std::string>{}(complex_str.str());
  }

  bool operator==(ParamKey key) const { return data_ == key; }

  phi::dtype::complex<double> data() const { return data_; }

 private:
  phi::dtype::complex<double> data_;
};
}  // namespace pir

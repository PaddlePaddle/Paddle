/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
   Copyright 2019 The TensorFlow Authors. All Rights Reserved.

This file is inspired by

    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/tstring.h

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <assert.h>

#include <ostream>
#include <string>

#include "paddle/phi/common/cpstring_impl.h"

namespace phi {
namespace dtype {

// Pstring is an only dtype of StringTensor, which is
// used to manage string data. It provides almost same
// interfaces compared to std::string, including data(),
// length() and so on. Besides, pstring data can be
// manipulated in GPU.

class pstring {
  PD_PString pstr_;

 public:
  enum Type {
    SMALL = PD_PSTR_SMALL,
    LARGE = PD_PSTR_LARGE,
    OFFSET = PD_PSTR_OFFSET,
    VIEW = PD_PSTR_VIEW,
  };

  typedef const char* const_iterator;

  // Ctor
  HOSTDEVICE pstring();
  HOSTDEVICE pstring(const std::string& str);  // NOLINT
  HOSTDEVICE pstring(const char* str, size_t len);
  HOSTDEVICE pstring(const char* str);  // NOLINT
  HOSTDEVICE pstring(size_t n, char c);

  // Copy
  HOSTDEVICE pstring(const pstring& str);

  // Move
  HOSTDEVICE pstring(pstring&& str) noexcept;

  // Dtor
  HOSTDEVICE ~pstring();

  // Copy Assignment
  HOSTDEVICE pstring& operator=(const pstring& str);
  HOSTDEVICE pstring& operator=(const std::string& str);
  HOSTDEVICE pstring& operator=(const char* str);
  HOSTDEVICE pstring& operator=(char ch);

  // Move Assignment
  HOSTDEVICE pstring& operator=(pstring&& str);

  // Comparison
  HOSTDEVICE int compare(const char* str, size_t len) const;
  HOSTDEVICE bool operator<(const pstring& o) const;
  HOSTDEVICE bool operator>(const pstring& o) const;
  HOSTDEVICE bool operator==(const char* str) const;
  HOSTDEVICE bool operator==(const pstring& o) const;
  HOSTDEVICE bool operator!=(const char* str) const;
  HOSTDEVICE bool operator!=(const pstring& o) const;

  // Conversion Operators
  HOSTDEVICE operator std::string() const;  // NOLINT

  // Attributes
  HOSTDEVICE size_t size() const;
  HOSTDEVICE size_t length() const;
  HOSTDEVICE size_t capacity() const;
  HOSTDEVICE bool empty() const;
  HOSTDEVICE Type type() const;

  // Allocation
  HOSTDEVICE void resize(size_t new_size, char c = 0);
  // Similar to resize, but will leave the newly grown region uninitialized.
  HOSTDEVICE void resize_uninitialized(size_t new_size);
  HOSTDEVICE void clear() noexcept;
  HOSTDEVICE void reserve(size_t n);

  // Iterators
  HOSTDEVICE const_iterator begin() const;
  HOSTDEVICE const_iterator end() const;

  // Const Element Access
  HOSTDEVICE const char* c_str() const;
  HOSTDEVICE const char* data() const;
  HOSTDEVICE const char& operator[](size_t i) const;
  HOSTDEVICE const char& back() const;

  // Mutable Element Access
  HOSTDEVICE char* mdata();
  HOSTDEVICE char& operator[](size_t i);

  // Assignment
  HOSTDEVICE pstring& assign(const char* str, size_t len);
  HOSTDEVICE pstring& assign(const char* str);

  // View Assignment
  HOSTDEVICE pstring& assign_as_view(const pstring& str);
  HOSTDEVICE pstring& assign_as_view(const std::string& str);
  HOSTDEVICE pstring& assign_as_view(const char* str, size_t len);
  HOSTDEVICE pstring& assign_as_view(const char* str);

  // Modifiers
  // NOTE: Invalid input will result in undefined behavior.
  HOSTDEVICE pstring& append(const pstring& str);
  HOSTDEVICE pstring& append(const char* str, size_t len);
  HOSTDEVICE pstring& append(const char* str);
  HOSTDEVICE pstring& append(size_t n, char c);

  HOSTDEVICE pstring& erase(size_t pos, size_t len);

  HOSTDEVICE pstring& insert(size_t pos,
                             const pstring& str,
                             size_t subpos,
                             size_t sublen);
  HOSTDEVICE pstring& insert(size_t pos, size_t n, char c);
  HOSTDEVICE void swap(pstring& str);
  HOSTDEVICE void push_back(char ch);

  // Friends
  HOSTDEVICE friend bool operator==(const char* a, const pstring& b);
  HOSTDEVICE friend bool operator==(const std::string& a, const pstring& b);
  HOSTDEVICE friend pstring operator+(const pstring& a, const pstring& b);
  HOSTDEVICE friend std::ostream& operator<<(std::ostream& o,
                                             const pstring& str);
};

// Non-member function overloads

HOSTDEVICE bool operator==(const char* a, const pstring& b);
HOSTDEVICE bool operator==(const std::string& a, const pstring& b);
HOSTDEVICE pstring operator+(const pstring& a, const pstring& b);
HOSTDEVICE std::ostream& operator<<(std::ostream& o, const pstring& str);
HOSTDEVICE size_t strlen(const char* start);

// Implementations

// Ctor

HOSTDEVICE inline pstring::pstring() { PD_PString_Init(&pstr_); }

HOSTDEVICE inline pstring::pstring(const char* str, size_t len) {
  PD_PString_Init(&pstr_);
  PD_PString_Copy(&pstr_, str, len);
}

HOSTDEVICE inline pstring::pstring(const char* str)
    : pstring(str, strlen(str)) {}

HOSTDEVICE inline pstring::pstring(size_t n, char c) {
  PD_PString_Init(&pstr_);
  PD_PString_Resize(&pstr_, n, c);
}

HOSTDEVICE inline pstring::pstring(const std::string& str)
    : pstring(str.data(), str.size()) {}

HOSTDEVICE inline pstring::pstring(const pstring& str) {
  PD_PString_Init(&pstr_);
  PD_PString_Assign(&pstr_, &str.pstr_);
}

// Move

HOSTDEVICE inline pstring::pstring(pstring&& str) noexcept {
  PD_PString_Init(&pstr_);
  PD_PString_Move(&pstr_, &str.pstr_);
}

// Dtor

HOSTDEVICE inline pstring::~pstring() { PD_PString_Dealloc(&pstr_); }

// Copy Assignment

HOSTDEVICE inline pstring& pstring::operator=(const pstring& str) {
  PD_PString_Assign(&pstr_, &str.pstr_);

  return *this;
}

HOSTDEVICE inline pstring& pstring::operator=(const std::string& str) {
  PD_PString_Copy(&pstr_, str.data(), str.size());
  return *this;
}

HOSTDEVICE inline pstring& pstring::operator=(const char* str) {
  PD_PString_Copy(&pstr_, str, strlen(str));

  return *this;
}

HOSTDEVICE inline pstring& pstring::operator=(char c) {
  resize_uninitialized(1);
  (*this)[0] = c;

  return *this;
}

// Move Assignment

HOSTDEVICE inline pstring& pstring::operator=(pstring&& str) {
  PD_PString_Move(&pstr_, &str.pstr_);

  return *this;
}

// Comparison

HOSTDEVICE inline int pstring::compare(const char* str, size_t len) const {
  int ret = PD_Memcmp(data(), str, (len < size()) ? len : size());

  if (ret < 0) return -1;
  if (ret > 0) return +1;

  if (size() < len) return -1;
  if (size() > len) return +1;

  return 0;
}

HOSTDEVICE inline bool pstring::operator<(const pstring& o) const {
  return compare(o.data(), o.size()) < 0;
}

HOSTDEVICE inline bool pstring::operator>(const pstring& o) const {
  return compare(o.data(), o.size()) > 0;
}

HOSTDEVICE inline bool pstring::operator==(const char* str) const {
  return strlen(str) == size() && PD_Memcmp(data(), str, size()) == 0;
}

HOSTDEVICE inline bool pstring::operator==(const pstring& o) const {
  return o.size() == size() && PD_Memcmp(data(), o.data(), size()) == 0;
}

HOSTDEVICE inline bool pstring::operator!=(const char* str) const {
  return !(*this == str);
}

HOSTDEVICE inline bool pstring::operator!=(const pstring& o) const {
  return !(*this == o);
}

// Conversion Operators

HOSTDEVICE inline pstring::operator std::string() const {
  return std::string(data(), size());
}

// Attributes

HOSTDEVICE inline size_t pstring::size() const {
  return PD_PString_GetSize(&pstr_);
}

HOSTDEVICE inline size_t pstring::length() const { return size(); }

HOSTDEVICE inline size_t pstring::capacity() const {
  return PD_PString_GetCapacity(&pstr_);
}

HOSTDEVICE inline bool pstring::empty() const { return size() == 0; }

HOSTDEVICE inline pstring::Type pstring::type() const {
  return static_cast<pstring::Type>(PD_PString_GetType(&pstr_));
}

// Allocation

HOSTDEVICE inline void pstring::resize(size_t new_size, char c) {
  PD_PString_Resize(&pstr_, new_size, c);
}

HOSTDEVICE inline void pstring::resize_uninitialized(size_t new_size) {
  PD_PString_ResizeUninitialized(&pstr_, new_size);
}

HOSTDEVICE inline void pstring::clear() noexcept {
  PD_PString_ResizeUninitialized(&pstr_, 0);
}

HOSTDEVICE inline void pstring::reserve(size_t n) {
  PD_PString_Reserve(&pstr_, n);
}

// Iterators

HOSTDEVICE inline pstring::const_iterator pstring::begin() const {
  return &(*this)[0];
}
HOSTDEVICE inline pstring::const_iterator pstring::end() const {
  return &(*this)[size()];
}

// Element Access

HOSTDEVICE inline const char* pstring::c_str() const { return data(); }

HOSTDEVICE inline const char* pstring::data() const {
  return PD_PString_GetDataPointer(&pstr_);
}

HOSTDEVICE inline const char& pstring::operator[](size_t i) const {
  return data()[i];
}

HOSTDEVICE inline const char& pstring::back() const {
  return (*this)[size() - 1];
}

HOSTDEVICE inline char* pstring::mdata() {
  return PD_PString_GetMutableDataPointer(&pstr_);
}

HOSTDEVICE inline char& pstring::operator[](size_t i) { return mdata()[i]; }

// Assignment

HOSTDEVICE inline pstring& pstring::assign(const char* str, size_t len) {
  PD_PString_Copy(&pstr_, str, len);

  return *this;
}

HOSTDEVICE inline pstring& pstring::assign(const char* str) {
  assign(str, strlen(str));

  return *this;
}

// View Assignment

HOSTDEVICE inline pstring& pstring::assign_as_view(const pstring& str) {
  assign_as_view(str.data(), str.size());

  return *this;
}

HOSTDEVICE inline pstring& pstring::assign_as_view(const std::string& str) {
  assign_as_view(str.data(), str.size());

  return *this;
}

HOSTDEVICE inline pstring& pstring::assign_as_view(const char* str,
                                                   size_t len) {
  PD_PString_AssignView(&pstr_, str, len);
  return *this;
}

HOSTDEVICE inline pstring& pstring::assign_as_view(const char* str) {
  assign_as_view(str, strlen(str));

  return *this;
}

// Modifiers

HOSTDEVICE inline pstring& pstring::append(const pstring& str) {
  PD_PString_Append(&pstr_, &str.pstr_);

  return *this;
}

HOSTDEVICE inline pstring& pstring::append(const char* str, size_t len) {
  PD_PString_AppendN(&pstr_, str, len);

  return *this;
}

HOSTDEVICE inline pstring& pstring::append(const char* str) {
  append(str, strlen(str));

  return *this;
}

HOSTDEVICE inline pstring& pstring::append(size_t n, char c) {
  // For append use cases, we want to ensure amortized growth.
  const size_t new_size = size() + n;
  PD_PString_ReserveAmortized(&pstr_, new_size);
  resize(new_size, c);

  return *this;
}

HOSTDEVICE inline pstring& pstring::erase(size_t pos, size_t len) {
  PD_Memmove(mdata() + pos, data() + pos + len, size() - len - pos);

  resize(size() - len);

  return *this;
}

HOSTDEVICE inline pstring& pstring::insert(size_t pos,
                                           const pstring& str,
                                           size_t subpos,
                                           size_t sublen) {
  size_t orig_size = size();
  PD_PString_ResizeUninitialized(&pstr_, orig_size + sublen);

  PD_Memmove(mdata() + pos + sublen, data() + pos, orig_size - pos);
  PD_Memmove(mdata() + pos, str.data() + subpos, sublen);

  return *this;
}

HOSTDEVICE inline pstring& pstring::insert(size_t pos, size_t n, char c) {
  size_t size_ = size();
  PD_PString_ResizeUninitialized(&pstr_, size_ + n);

  PD_Memmove(mdata() + pos + n, data() + pos, size_ - pos);
  PD_Memset(mdata() + pos, c, n);

  return *this;
}

HOSTDEVICE inline void pstring::swap(pstring& str) {
  std::swap(pstr_, str.pstr_);
}

HOSTDEVICE inline void pstring::push_back(char ch) { append(1, ch); }

// Friends

HOSTDEVICE inline bool operator==(const char* a, const pstring& b) {
  return strlen(a) == b.size() && PD_Memcmp(a, b.data(), b.size()) == 0;
}

HOSTDEVICE inline bool operator==(const std::string& a, const pstring& b) {
  return a.size() == b.size() && PD_Memcmp(a.data(), b.data(), b.size()) == 0;
}

HOSTDEVICE inline pstring operator+(const pstring& a, const pstring& b) {
  pstring r;
  r.reserve(a.size() + b.size());
  r.append(a);
  r.append(b);

  return r;
}

HOSTDEVICE inline std::ostream& operator<<(std::ostream& o,
                                           const pstring& str) {
  return o.write(str.data(), str.size());
}

HOSTDEVICE inline size_t strlen(const char* start) {
  const char* end = start;
  for (; *end != '\0'; ++end) {
  }
  return end - start;
}

}  // namespace dtype
}  // namespace phi

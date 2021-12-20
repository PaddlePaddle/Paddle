/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <iostream>
#include <ostream>
#include <string>

#include "paddle/pten/core/platform/cpstring.h"

namespace pten {
namespace platform {

class pstring {
  PD_PString pstr_;

 public:
  enum Type {
    // See cstring.h
    SMALL = PD_PSTR_SMALL,
    LARGE = PD_PSTR_LARGE,
    OFFSET = PD_PSTR_OFFSET,
    VIEW = PD_PSTR_VIEW,
  };

  // Assignment to a pstring object with a pstring::view type will create a VIEW
  // type pstring.
  class view {
    const char* data_;
    size_t size_;

   public:
    explicit view(const char* data, size_t size) : data_(data), size_(size) {}
    explicit view(const char* data) : data_(data), size_(::strlen(data)) {}

    const char* data() const { return data_; }

    size_t size() const { return size_; }

    view() = delete;
    view(const view&) = delete;
    view& operator=(const view&) = delete;
  };

  typedef const char* const_iterator;

  // Ctor
  pstring();
  pstring(const std::string& str);  // NOLINT TODO(b/147740521): Make explicit.
  pstring(const char* str, size_t len);
  pstring(const char* str);  // NOLINT TODO(b/147740521): Make explicit.
  pstring(size_t n, char c);

  // Copy
  pstring(const pstring& str);

  // Move
  pstring(pstring&& str) noexcept;

  // Dtor
  ~pstring();

  // Copy Assignment
  pstring& operator=(const pstring& str);
  pstring& operator=(const std::string& str);
  pstring& operator=(const char* str);
  pstring& operator=(char ch);

  // View Assignment
  pstring& operator=(const view& tsv);

  // Move Assignment
  pstring& operator=(pstring&& str);

  // Comparison
  int compare(const char* str, size_t len) const;
  bool operator<(const pstring& o) const;
  bool operator>(const pstring& o) const;
  bool operator==(const char* str) const;
  bool operator==(const pstring& o) const;
  bool operator!=(const char* str) const;
  bool operator!=(const pstring& o) const;

  // Conversion Operators
  // TODO(b/147740521): Make explicit.
  operator std::string() const;  // NOLINT

  // Attributes
  size_t size() const;
  size_t length() const;
  size_t capacity() const;
  bool empty() const;
  Type type() const;

  // Allocation
  void resize(size_t new_size, char c = 0);
  // Similar to resize, but will leave the newly grown region uninitialized.
  void resize_uninitialized(size_t new_size);
  void clear() noexcept;
  void reserve(size_t n);

  // Iterators
  const_iterator begin() const;
  const_iterator end() const;

  // Const Element Access
  const char* c_str() const;
  const char* data() const;
  const char& operator[](size_t i) const;
  const char& back() const;

  // Mutable Element Access
  // NOTE: For VIEW/OFFSET types, calling these methods will result in the
  // conversion to a SMALL or heap allocated LARGE type.  As a result,
  // previously obtained pointers, references, or iterators to the underlying
  // buffer will point to the original VIEW/OFFSET and not the new allocation.
  char* mdata();
  char* data();  // DEPRECATED: Use mdata().
  char& operator[](size_t i);

  // Assignment
  pstring& assign(const char* str, size_t len);
  pstring& assign(const char* str);

  // View Assignment
  pstring& assign_as_view(const pstring& str);
  pstring& assign_as_view(const std::string& str);
  pstring& assign_as_view(const char* str, size_t len);
  pstring& assign_as_view(const char* str);

  // Modifiers
  // NOTE: Invalid input will result in undefined behavior.
  pstring& append(const pstring& str);
  pstring& append(const char* str, size_t len);
  pstring& append(const char* str);
  pstring& append(size_t n, char c);

  pstring& erase(size_t pos, size_t len);

  pstring& insert(size_t pos, const pstring& str, size_t subpos, size_t sublen);
  pstring& insert(size_t pos, size_t n, char c);
  void swap(pstring& str);
  void push_back(char ch);

  // Friends
  friend bool operator==(const char* a, const pstring& b);
  friend bool operator==(const std::string& a, const pstring& b);
  friend pstring operator+(const pstring& a, const pstring& b);
  friend std::ostream& operator<<(std::ostream& o, const pstring& str);
  friend std::hash<pstring>;
};

// Non-member function overloads

bool operator==(const char* a, const pstring& b);
bool operator==(const std::string& a, const pstring& b);
pstring operator+(const pstring& a, const pstring& b);
std::ostream& operator<<(std::ostream& o, const pstring& str);

// Implementations

// Ctor

inline pstring::pstring() { PD_PString_Init(&pstr_); }

inline pstring::pstring(const char* str, size_t len) {
  PD_PString_Init(&pstr_);
  PD_PString_Copy(&pstr_, str, len);
}

inline pstring::pstring(const char* str) : pstring(str, ::strlen(str)) {}

inline pstring::pstring(size_t n, char c) {
  PD_PString_Init(&pstr_);
  PD_PString_Resize(&pstr_, n, c);
}

inline pstring::pstring(const std::string& str)
    : pstring(str.data(), str.size()) {}

inline pstring::pstring(const pstring& str) {
  PD_PString_Init(&pstr_);
  PD_PString_Assign(&pstr_, &str.pstr_);
}

// Move

inline pstring::pstring(pstring&& str) noexcept {
  PD_PString_Init(&pstr_);
  PD_PString_Move(&pstr_, &str.pstr_);
}

// Dtor

inline pstring::~pstring() { PD_PString_Dealloc(&pstr_); }

// Copy Assignment

inline pstring& pstring::operator=(const pstring& str) {
  PD_PString_Assign(&pstr_, &str.pstr_);

  return *this;
}

inline pstring& pstring::operator=(const std::string& str) {
  PD_PString_Copy(&pstr_, str.data(), str.size());
  return *this;
}

inline pstring& pstring::operator=(const char* str) {
  PD_PString_Copy(&pstr_, str, ::strlen(str));

  return *this;
}

inline pstring& pstring::operator=(char c) {
  resize_uninitialized(1);
  (*this)[0] = c;

  return *this;
}

// View Assignment

inline pstring& pstring::operator=(const pstring::view& tsv) {
  assign_as_view(tsv.data(), tsv.size());

  return *this;
}

// Move Assignment

inline pstring& pstring::operator=(pstring&& str) {
  PD_PString_Move(&pstr_, &str.pstr_);

  return *this;
}

// Comparison

inline int pstring::compare(const char* str, size_t len) const {
  int ret = ::memcmp(data(), str, std::min(len, size()));

  if (ret < 0) return -1;
  if (ret > 0) return +1;

  if (size() < len) return -1;
  if (size() > len) return +1;

  return 0;
}

inline bool pstring::operator<(const pstring& o) const {
  return compare(o.data(), o.size()) < 0;
}

inline bool pstring::operator>(const pstring& o) const {
  return compare(o.data(), o.size()) > 0;
}

inline bool pstring::operator==(const char* str) const {
  return ::strlen(str) == size() && ::memcmp(data(), str, size()) == 0;
}

inline bool pstring::operator==(const pstring& o) const {
  return o.size() == size() && ::memcmp(data(), o.data(), size()) == 0;
}

inline bool pstring::operator!=(const char* str) const {
  return !(*this == str);
}

inline bool pstring::operator!=(const pstring& o) const {
  return !(*this == o);
}

// Conversion Operators

inline pstring::operator std::string() const {
  return std::string(data(), size());
}

// Attributes

inline size_t pstring::size() const { return PD_PString_GetSize(&pstr_); }

inline size_t pstring::length() const { return size(); }

inline size_t pstring::capacity() const {
  return PD_PString_GetCapacity(&pstr_);
}

inline bool pstring::empty() const { return size() == 0; }

inline pstring::Type pstring::type() const {
  return static_cast<pstring::Type>(PD_PString_GetType(&pstr_));
}

// Allocation

inline void pstring::resize(size_t new_size, char c) {
  PD_PString_Resize(&pstr_, new_size, c);
}

inline void pstring::resize_uninitialized(size_t new_size) {
  PD_PString_ResizeUninitialized(&pstr_, new_size);
}

inline void pstring::clear() noexcept {
  PD_PString_ResizeUninitialized(&pstr_, 0);
}

inline void pstring::reserve(size_t n) { PD_PString_Reserve(&pstr_, n); }

// Iterators

inline pstring::const_iterator pstring::begin() const { return &(*this)[0]; }
inline pstring::const_iterator pstring::end() const { return &(*this)[size()]; }

// Element Access

inline const char* pstring::c_str() const { return data(); }

inline const char* pstring::data() const {
  return PD_PString_GetDataPointer(&pstr_);
}

inline const char& pstring::operator[](size_t i) const { return data()[i]; }

inline const char& pstring::back() const { return (*this)[size() - 1]; }

inline char* pstring::mdata() {
  return PD_PString_GetMutableDataPointer(&pstr_);
}

inline char* pstring::data() {
  // Deprecated
  return mdata();
}

inline char& pstring::operator[](size_t i) { return mdata()[i]; }

// Assignment

inline pstring& pstring::assign(const char* str, size_t len) {
  PD_PString_Copy(&pstr_, str, len);

  return *this;
}

inline pstring& pstring::assign(const char* str) {
  assign(str, ::strlen(str));

  return *this;
}

// View Assignment

inline pstring& pstring::assign_as_view(const pstring& str) {
  assign_as_view(str.data(), str.size());

  return *this;
}

inline pstring& pstring::assign_as_view(const std::string& str) {
  assign_as_view(str.data(), str.size());

  return *this;
}

inline pstring& pstring::assign_as_view(const char* str, size_t len) {
  PD_PString_AssignView(&pstr_, str, len);
  std::cout << "call PD_PString_AssignView" << std::endl;
  return *this;
}

inline pstring& pstring::assign_as_view(const char* str) {
  assign_as_view(str, ::strlen(str));

  return *this;
}

// Modifiers

inline pstring& pstring::append(const pstring& str) {
  PD_PString_Append(&pstr_, &str.pstr_);

  return *this;
}

inline pstring& pstring::append(const char* str, size_t len) {
  PD_PString_AppendN(&pstr_, str, len);

  return *this;
}

inline pstring& pstring::append(const char* str) {
  append(str, ::strlen(str));

  return *this;
}

inline pstring& pstring::append(size_t n, char c) {
  // For append use cases, we want to ensure amortized growth.
  const size_t new_size = size() + n;
  PD_PString_ReserveAmortized(&pstr_, new_size);
  resize(new_size, c);

  return *this;
}

inline pstring& pstring::erase(size_t pos, size_t len) {
  memmove(mdata() + pos, data() + pos + len, size() - len - pos);

  resize(size() - len);

  return *this;
}

inline pstring& pstring::insert(size_t pos,
                                const pstring& str,
                                size_t subpos,
                                size_t sublen) {
  size_t orig_size = size();
  PD_PString_ResizeUninitialized(&pstr_, orig_size + sublen);

  memmove(mdata() + pos + sublen, data() + pos, orig_size - pos);
  memmove(mdata() + pos, str.data() + subpos, sublen);

  return *this;
}

inline pstring& pstring::insert(size_t pos, size_t n, char c) {
  size_t size_ = size();
  PD_PString_ResizeUninitialized(&pstr_, size_ + n);

  memmove(mdata() + pos + n, data() + pos, size_ - pos);
  memset(mdata() + pos, c, n);

  return *this;
}

inline void pstring::swap(pstring& str) {
  // TODO(dero): Invalid for OFFSET (unimplemented).
  std::swap(pstr_, str.pstr_);
}

inline void pstring::push_back(char ch) { append(1, ch); }

// Friends

inline bool operator==(const char* a, const pstring& b) {
  return ::strlen(a) == b.size() && ::memcmp(a, b.data(), b.size()) == 0;
}

inline bool operator==(const std::string& a, const pstring& b) {
  return a.size() == b.size() && ::memcmp(a.data(), b.data(), b.size()) == 0;
}

inline pstring operator+(const pstring& a, const pstring& b) {
  pstring r;
  r.reserve(a.size() + b.size());
  r.append(a);
  r.append(b);

  return r;
}

inline std::ostream& operator<<(std::ostream& o, const pstring& str) {
  return o.write(str.data(), str.size());
}

}  // namespace platform
}  // namespace pten

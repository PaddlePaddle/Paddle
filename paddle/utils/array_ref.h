// This file copy from llvm/ADT/ArrayRef.h, version: 12.0.0
// Modified the following points
// 1. remove hash_value functions
// 2. replace with the llvm::NoneType with paddle::none_t
// 3. remove drop_while, drop_until, take_while, take_until methods
// 4. change ArrayRef to array_ref to unify naming style of utils

//===- ArrayRef.h - Array Reference Wrapper ---------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include "paddle/utils/none.h"
#include "paddle/utils/small_vector.h"

namespace paddle {

/// array_ref - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the array_ref. For this reason, it is not in general
/// safe to store an array_ref.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
template <typename T>
class array_ref {
 public:
  using iterator = const T *;
  using const_iterator = const T *;
  using size_type = size_t;
  using reverse_iterator = std::reverse_iterator<iterator>;

 private:
  /// The start of the array, in an external buffer.
  const T *Data = nullptr;

  /// The number of elements.
  size_type Length = 0;

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty array_ref.
  /*implicit*/ array_ref() = default;

  /// Construct an empty array_ref from None.
  /*implicit*/ array_ref(none_t) {}

  /// Construct an array_ref from a single element.
  /*implicit*/ array_ref(const T &OneElt) : Data(&OneElt), Length(1) {}

  /// Construct an array_ref from a pointer and length.
  /*implicit*/ array_ref(const T *data, size_t length)
      : Data(data), Length(length) {}

  /// Construct an array_ref from a range.
  array_ref(const T *begin, const T *end) : Data(begin), Length(end - begin) {}

  /// Construct an array_ref from a small_vector. This is templated in order to
  /// avoid instantiating small_vector_template_common<T> whenever we
  /// copy-construct an array_ref.
  template <typename U>
  /*implicit*/ array_ref(const small_vector_template_common<T, U> &Vec)
      : Data(Vec.data()), Length(Vec.size()) {}

  /// Construct an array_ref from a std::vector.
  template <typename A>
  /*implicit*/ array_ref(const std::vector<T, A> &Vec)
      : Data(Vec.data()), Length(Vec.size()) {}

  /// Construct an array_ref from a std::array
  template <size_t N>
  /*implicit*/ constexpr array_ref(const std::array<T, N> &Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct an array_ref from a C array.
  template <size_t N>
  /*implicit*/ constexpr array_ref(const T (&Arr)[N]) : Data(Arr), Length(N) {}

/// Construct an array_ref from a std::initializer_list.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 9
// Disable gcc's warning in this constructor as it generates an enormous
// amount
// of messages. Anyone using array_ref should already be aware of the fact that
// it does not do lifetime extension.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif
  /*implicit*/ array_ref(const std::initializer_list<T> &Vec)
      : Data(Vec.begin() == Vec.end() ? (T *)nullptr : Vec.begin()),
        Length(Vec.size()) {}
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 9
#pragma GCC diagnostic pop
#endif

  /// Construct an array_ref<const T*> from array_ref<T*>. This uses SFINAE to
  /// ensure that only ArrayRefs of pointers can be converted.
  template <typename U>
  array_ref(const array_ref<U *> &A,
            std::enable_if_t<std::is_convertible<U *const *, T const *>::value>
                * = nullptr)
      : Data(A.data()), Length(A.size()) {}

  /// Construct an array_ref<const T*> from a small_vector<T*>. This is
  /// templated in order to avoid instantiating small_vector_template_common<T>
  /// whenever we copy-construct an array_ref.
  template <typename U, typename DummyT>
  /*implicit*/ array_ref(
      const small_vector_template_common<U *, DummyT> &Vec,
      std::enable_if_t<std::is_convertible<U *const *, T const *>::value> * =
          nullptr)
      : Data(Vec.data()), Length(Vec.size()) {}

  /// Construct an array_ref<const T*> from std::vector<T*>. This uses SFINAE
  /// to ensure that only vectors of pointers can be converted.
  template <typename U, typename A>
  array_ref(
      const std::vector<U *, A> &Vec,
      std::enable_if_t<std::is_convertible<U *const *, T const *>::value> * = 0)
      : Data(Vec.data()), Length(Vec.size()) {}

  /// @}
  /// @name Simple Operations
  /// @{

  iterator begin() const { return Data; }
  iterator end() const { return Data + Length; }

  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  /// empty - Check if the array is empty.
  bool empty() const { return Length == 0; }

  const T *data() const { return Data; }

  /// size - Get the array size.
  size_t size() const { return Length; }

  /// front - Get the first element.
  const T &front() const {
    assert(!empty());
    return Data[0];
  }

  /// back - Get the last element.
  const T &back() const {
    assert(!empty());
    return Data[Length - 1];
  }

  // copy - Allocate copy in Allocator and return array_ref<T> to it.
  template <typename Allocator>
  array_ref<T> copy(Allocator &A) {
    T *Buff = A.template Allocate<T>(Length);
    std::uninitialized_copy(begin(), end(), Buff);
    return array_ref<T>(Buff, Length);
  }

  /// equals - Check for element-wise equality.
  bool equals(array_ref RHS) const {
    if (Length != RHS.Length) return false;
    return std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Chop off the first N elements of the array, and keep M
  /// elements in the array.
  array_ref<T> slice(size_t N, size_t M) const {
    assert(N + M <= size() && "Invalid specifier");
    return array_ref<T>(data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  array_ref<T> slice(size_t N) const { return slice(N, size() - N); }

  /// Drop the first \p N elements of the array.
  array_ref<T> drop_front(size_t N = 1) const {
    assert(size() >= N && "Dropping more elements than exist");
    return slice(N, size() - N);
  }

  /// Drop the last \p N elements of the array.
  array_ref<T> drop_back(size_t N = 1) const {
    assert(size() >= N && "Dropping more elements than exist");
    return slice(0, size() - N);
  }

  /// Return a copy of *this with only the first \p N elements.
  array_ref<T> take_front(size_t N = 1) const {
    if (N >= size()) return *this;
    return drop_back(size() - N);
  }

  /// Return a copy of *this with only the last \p N elements.
  array_ref<T> take_back(size_t N = 1) const {
    if (N >= size()) return *this;
    return drop_front(size() - N);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  const T &operator[](size_t Index) const {
    assert(Index < Length && "Invalid index!");
    return Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same<U, T>::value, array_ref<T>> &operator=(
      U &&Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same<U, T>::value, array_ref<T>> &operator=(
      std::initializer_list<U>) = delete;

  /// @}
  /// @name Expensive Operations
  /// @{
  std::vector<T> vec() const { return std::vector<T>(Data, Data + Length); }

  /// @}
  /// @name Conversion operators
  /// @{
  operator std::vector<T>() const {
    return std::vector<T>(Data, Data + Length);
  }

  /// @}
};

/// @name array_ref Convenience constructors
/// @{

/// Construct an array_ref from a single element.
template <typename T>
array_ref<T> make_array_ref(const T &OneElt) {
  return OneElt;
}

/// Construct an array_ref from a pointer and length.
template <typename T>
array_ref<T> make_array_ref(const T *data, size_t length) {
  return array_ref<T>(data, length);
}

/// Construct an array_ref from a range.
template <typename T>
array_ref<T> make_array_ref(const T *begin, const T *end) {
  return array_ref<T>(begin, end);
}

/// Construct an array_ref from a small_vector.
template <typename T>
array_ref<T> make_array_ref(const small_vector_impl<T> &Vec) {
  return Vec;
}

/// Construct an array_ref from a small_vector.
template <typename T, unsigned N>
array_ref<T> make_array_ref(const small_vector<T, N> &Vec) {
  return Vec;
}

/// Construct an array_ref from a std::vector.
template <typename T>
array_ref<T> make_array_ref(const std::vector<T> &Vec) {
  return Vec;
}

/// Construct an array_ref from a std::array.
template <typename T, std::size_t N>
array_ref<T> make_array_ref(const std::array<T, N> &Arr) {
  return Arr;
}

/// Construct an array_ref from an array_ref (no-op) (const)
template <typename T>
array_ref<T> make_array_ref(const array_ref<T> &Vec) {
  return Vec;
}

/// Construct an array_ref from an array_ref (no-op)
template <typename T>
array_ref<T> &make_array_ref(array_ref<T> &Vec) {
  return Vec;
}

/// Construct an array_ref from a C array.
template <typename T, size_t N>
array_ref<T> make_array_ref(const T (&Arr)[N]) {
  return array_ref<T>(Arr);
}

/// @}
/// @name array_ref Comparison Operators
/// @{

template <typename T>
inline bool operator==(array_ref<T> LHS, array_ref<T> RHS) {
  return LHS.equals(RHS);
}

template <typename T>
inline bool operator==(small_vector_impl<T> &LHS, array_ref<T> RHS) {
  return array_ref<T>(LHS).equals(RHS);
}

template <typename T>
inline bool operator!=(array_ref<T> LHS, array_ref<T> RHS) {
  return !(LHS == RHS);
}

template <typename T>
inline bool operator!=(small_vector_impl<T> &LHS, array_ref<T> RHS) {
  return !(LHS == RHS);
}

}  // namespace paddle

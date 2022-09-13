/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Statically sized array of elements that accommodates all CUTLASS-supported numeric types
           and is safe to use in a union.
*/

#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically sized array for any data type
template <
  typename T,
  int N,
  bool RegisterSized = sizeof_bits<T>::value >= 32
>
class Array;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an Array<> in bits
template <typename T, int N, bool RegisterSized>
struct sizeof_bits<Array<T, N, RegisterSized> > {
  static int const value =
    int(sizeof(typename Array<T, N, RegisterSized>::Storage)) * 8 * int(Array<T, N, RegisterSized>::kStorageElements);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if the argument is a power of 2
CUTLASS_HOST_DEVICE
constexpr bool ispow2(unsigned x) {
  return x && (!(x & (x - 1)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the largest power of two not greater than the argument.
CUTLASS_HOST_DEVICE
constexpr unsigned floor_pow_2(unsigned x) {
  return (x == 0 || ispow2(x)) ? x : ((floor_pow_2(x >> 1)) << 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically sized array for any data type
template <
  typename T,
  int N
>
class Array<T, N, true> {
public:

  /// Storage type
  using Storage = T;

  /// Element type
  using Element = T;

  /// Number of storage elements
  //static std::size_t const kStorageElements = N;
  static size_t const kStorageElements = N;

  /// Number of logical elements
  static size_t const kElements = N;

  //
  // C++ standard members
  //

  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type &reference;
  typedef value_type const & const_reference;
  typedef value_type *pointer;
  typedef value_type const * const_pointer;

  //
  // Iterators
  //

  /// Bidirectional iterator over elements
  class iterator {

    /// Pointer to object
    T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    iterator(T *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    iterator &operator++() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator &operator--() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator operator++(int) {
      iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    iterator operator--(int) {
      iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T &operator*() const {
      return *ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator==(iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional constant iterator over elements
  class const_iterator {

    /// Pointer to object
    const T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    const_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    const_iterator(T const *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    const_iterator &operator++() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_iterator &operator--() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_iterator operator++(int) {
      const_iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_iterator operator--(int) {
      const_iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T const &operator*() const {
      return *ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator==(const_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(const_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional iterator over elements
  class reverse_iterator {

    /// Pointer to object
    T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    reverse_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    reverse_iterator(T *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    reverse_iterator &operator++() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator &operator--() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator operator++(int) {
      iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator operator--(int) {
      iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T &operator*() const {
      return *(ptr_ - 1);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(reverse_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(reverse_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional constant iterator over elements
  class const_reverse_iterator {

    /// Pointer to object
    T const *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(T const *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator &operator++() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator &operator--() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator operator++(int) {
      const_reverse_iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator operator--(int) {
      const_reverse_iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T const &operator*() const {
      return *(ptr_ - 1);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(const_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(const_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

private:

  /// Internal storage
  Storage storage[kElements];

public:

  #if 0
  CUTLASS_HOST_DEVICE
  Array() { }

  CUTLASS_HOST_DEVICE
  Array(Array const &x) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElements; ++i) {
      storage[i] = x.storage[i];
    }
  }
  #endif

  /// Efficient clear method
  CUTLASS_HOST_DEVICE
  void clear() {
    fill(T(0));
  }

  CUTLASS_HOST_DEVICE
  reference at(size_type pos) {
    return reinterpret_cast<reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  const_reference at(size_type pos) const {
    return reinterpret_cast<const_reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  reference operator[](size_type pos) {
    return reinterpret_cast<reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  const_reference operator[](size_type pos) const {
    return reinterpret_cast<const_reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  reference front() {
    return reinterpret_cast<reference>(storage[0]);
  }

  CUTLASS_HOST_DEVICE
  const_reference front() const {
    return reinterpret_cast<const_reference>(storage[0]);
  }

  CUTLASS_HOST_DEVICE
  reference back() {
    return reinterpret_cast<reference>(storage[kStorageElements - 1]);
  }

  CUTLASS_HOST_DEVICE
  const_reference back() const {
    return reinterpret_cast<const_reference>(storage[kStorageElements - 1]);
  }

  CUTLASS_HOST_DEVICE
  pointer data() {
    return reinterpret_cast<pointer>(storage);
  }

  CUTLASS_HOST_DEVICE
  const_pointer data() const {
    return reinterpret_cast<const_pointer>(storage);
  }
  
  CUTLASS_HOST_DEVICE
  pointer raw_data() {
    return reinterpret_cast<pointer>(storage);
  }

  CUTLASS_HOST_DEVICE
  const_pointer raw_data() const {
    return reinterpret_cast<const_pointer>(storage);
  }


  CUTLASS_HOST_DEVICE
  constexpr bool empty() const {
    return !kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type max_size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  void fill(T const &value) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElements; ++i) {
      storage[i] = static_cast<Storage>(value);
    }
  }

  CUTLASS_HOST_DEVICE
  iterator begin() {
    return iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  const_iterator cbegin() const {
    return const_iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  iterator end() {
    return iterator(reinterpret_cast<pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  const_iterator cend() const {
    return const_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rbegin() {
    return reverse_iterator(reinterpret_cast<pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rend() {
    return reverse_iterator(reinterpret_cast<pointer>(storage));
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crend() const {
    return const_reverse_iterator(reinterpret_cast<const_pointer>(storage));
  }

  //
  // Comparison operators
  //

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 1> make_Array(Element x) {
  Array<Element, 1> m;
  m[0] = x;
  return m;
}

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 2> make_Array(Element x, Element y) {
  Array<Element, 2> m;
  m[0] = x;
  m[1] = y;
  return m;
}

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 3> make_Array(Element x, Element y, Element z) {
  Array<Element, 3> m;
  m[0] = x;
  m[1] = y;
  m[2] = z;
  return m;
}

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 4> make_Array(Element x, Element y, Element z, Element w) {
  Array<Element, 4> m;
  m[0] = x;
  m[1] = y;
  m[2] = z;
  m[3] = w;
  return m;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/array_subbyte.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Aligned array type
template <
  /// Element type
  typename T,
  /// Number of elements in the array
  int N,
  /// Alignment requirement in bytes
  int Alignment = sizeof_bits<T>::value * N / 8
>
class alignas(Alignment) AlignedArray: public Array<T, N> {
public:

};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////


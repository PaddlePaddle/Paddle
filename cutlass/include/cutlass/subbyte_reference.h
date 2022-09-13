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
    \brief Provides a mechanism for packing and unpacking elements smaller than one byte
*/
#pragma once

#include "cutlass/numeric_types.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This class provides a mechanism for packing and unpacking elements smaller than one byte. It
/// assumes these sub-byte elements are packed in a traditional C++ numeric type.
///
/// The intended application is to provide a mechanism to indirectly reference elements in
/// memory or Array<> objects whose addresses cannot otherwise be taken since they are smaller
/// than one byte.
/// 
/// Supports basic pointer arithmetic:
///
/// Example:
///
///   int4b_t *ptr = ...;
///
///   SubbyteReference<int4b_t> ref = ptr;
///   ref += 15;
///
///   int4b_t x = ref;      // load an int4b_t
///   ref = x + 2_s4;      // perform arithmetic on int4b_t and then store
///
template <
  typename Element_,              /// CUTLASS numeric element type.
  typename Storage_ = uint8_t     /// Underlying storage type. Must be able to hold an integer 
                                  ///   number of objects of type Element.
>
class ConstSubbyteReference {
public:

  using Element = Element_;
  using Storage = Storage_;
  using StoragePointer = Storage const *;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
    "Size of Element must not be greater than Storage.");

  static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
    "Storage must be divisible by Element");

private:

  ///! Number of elements per storage vector
  int const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

  ///! Bit mask 
  Storage const kMask = 
    ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? 
      (Storage(1) << sizeof_bits<Element>::value) - Storage(1) :
      ~Storage(0));

private:

  /// Pointer to array containing element
  StoragePointer ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

public:

  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(): ptr_(nullptr), offset_(0) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(
    Element const *ptr,           /// pointer to memory
    int64_t offset          /// logical offset in units of Element
  ): 
    ptr_(reinterpret_cast<StoragePointer>(ptr)),
    offset_(0) {

    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference(
    Element *ptr = nullptr
  ): ConstSubbyteReference(ptr, 0) { }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  StoragePointer storage_pointer() const {
    return ptr_;
  }

  /// Gets element offset within storage vector
  CUTLASS_HOST_DEVICE
  int element_offset() const {
    return offset_;
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  Element get() const {
    Storage item = Storage((*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
    return reinterpret_cast<Element const &>(item);
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  operator Element() const {
    return get();
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator+=(int offset) {

    offset += offset_;
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator+=(long long offset) {

    offset += offset_;
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator-=(int offset) {
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference &operator-=(long long offset) {
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator+(int offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator+(long long offset) const {
    
    ConstSubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator-(int offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  ConstSubbyteReference operator-=(long long offset) const {

    ConstSubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Computes the difference in elements between references
  CUTLASS_HOST_DEVICE
  ptrdiff_t operator-(ConstSubbyteReference ref) const {
    return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
  }

  /// Explicit cast to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(get());
  }

  /// Explicit cast to signed 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator int64_t() const {
    return int64_t(get());
  }

  /// Explicit cast to unsigned 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const {
    return uint64_t(get());
  }

  /// Explicit cast to float
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }

  /// Explicit cast to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(get());
  }
};

template <
  typename Element_,              /// CUTLASS numeric element type.
  typename Storage_ = uint8_t     /// Underlying storage type. Must be able to hold an integer 
                                  ///   number of objects of type Element.
>
class SubbyteReference {
public:

  using Element = Element_;
  using Storage = Storage_;
  using StoragePointer = Storage *;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
    "Size of Element must not be greater than Storage.");

  static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
    "Storage must be divisible by Element");

private:

  ///! Number of elements per storage vector
  int const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

  ///! Bit mask 
  Storage const kMask = 
    ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? 
      (Storage(1) << sizeof_bits<Element>::value) - Storage(1) :
      ~Storage(0));

private:

  /// Pointer to array containing element
  StoragePointer ptr_;

  /// Offset (in units of elements) from pointer.
  ///
  /// Invariant: must always be in range [0, kElementsPerVector)
  int offset_;

public:

  CUTLASS_HOST_DEVICE
  SubbyteReference(): ptr_(nullptr), offset_(0) { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  SubbyteReference(
    Element *ptr,           /// pointer to memory
    int64_t offset          /// logical offset in units of Element
  ): 
    ptr_(reinterpret_cast<StoragePointer>(ptr)),
    offset_(0) {

    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);
  }

  /// Constructor
  CUTLASS_HOST_DEVICE
  SubbyteReference(
    Element *ptr = nullptr
  ): SubbyteReference(ptr, 0) { }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  StoragePointer storage_pointer() const {
    return ptr_;
  }

  /// Gets storage pointer
  CUTLASS_HOST_DEVICE
  Element * operator&() const {
    return reinterpret_cast<Element *>(ptr_);
  }

  /// Gets element offset within storage vector
  CUTLASS_HOST_DEVICE
  int element_offset() const {
    return offset_;
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  Element get() const {
    Storage item = Storage((*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
    return reinterpret_cast<Element const &>(item);
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference & set(Element const &x) {

    Storage item = (reinterpret_cast<Storage const &>(x) & kMask);

    Storage kUpdateMask = Storage(~(kMask << (offset_ * sizeof_bits<Element>::value)));
    *ptr_ = Storage((*ptr_ & kUpdateMask) | Storage(item << (offset_ * sizeof_bits<Element>::value)));

    return *this;
  }

  /// Unpacks an element from memory
  CUTLASS_HOST_DEVICE
  operator Element() const {
    return get();
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(Element const & x) {
    return set(x);
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(SubbyteReference const & x) {
    return set(x.get());
  }

  /// Stores an element to memory
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator=(
      ConstSubbyteReference<Element, Storage> const &x) {
    return set(x.get());
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator+=(int offset) {

    offset += offset_;
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator+=(long long offset) {

    offset += offset_;
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ += offset_in_vectors;
    offset_ = offset_in_elements;

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator-=(int offset) {
    
    int offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = offset % kElementsPerVector;

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Adds an offset in units of elements to the reference
  CUTLASS_HOST_DEVICE
  SubbyteReference &operator-=(long long offset) {
    
    long long offset_in_vectors = offset / kElementsPerVector;
    int offset_in_elements = int(offset % kElementsPerVector);

    ptr_ -= offset_in_vectors;
    offset_ -= offset_in_elements;

    if (offset_ < 0) {
      offset_ += kElementsPerVector;
      --ptr_;
    }

    return *this;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator+(int offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator+(long long offset) const {
    
    SubbyteReference ref(ptr_, offset_);
    ref += offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator-(int offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Returns a reference to an element with a given offset from the current reference
  CUTLASS_HOST_DEVICE
  SubbyteReference operator-=(long long offset) const {

    SubbyteReference ref(ptr_, offset_);
    ref -= offset;

    return ref;
  }

  /// Computes the difference in elements between references
  CUTLASS_HOST_DEVICE
  ptrdiff_t operator-(SubbyteReference ref) const {
    return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
  }

  /// Explicit cast to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(get());
  }

  /// Explicit cast to signed 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator int64_t() const {
    return int64_t(get());
  }

  /// Explicit cast to unsigned 64-bit integer
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const {
    return uint64_t(get());
  }

  /// Explicit cast to float
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }

  /// Explicit cast to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(get());
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, bool subbyte = (sizeof_bits<Element>::value < 8)>
struct ReferenceFactory;

template <typename Element>
struct ReferenceFactory<Element, false> {
  CUTLASS_HOST_DEVICE
  static Element &get(Element *ptr, int64_t offset) {
    return ptr[offset];
  }

  CUTLASS_HOST_DEVICE
  static Element const &get(Element const *ptr, int64_t offset) {
    return ptr[offset];
  }
};

template <typename Element>
struct ReferenceFactory<Element, true> {
  CUTLASS_HOST_DEVICE
  static SubbyteReference<Element> get(Element *ptr, int64_t offset) {
    return SubbyteReference<Element>(ptr, offset);
  }

  CUTLASS_HOST_DEVICE
  static ConstSubbyteReference<Element> get(Element const *ptr,
                                             int64_t offset) {
    return ConstSubbyteReference<Element>(ptr, offset);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

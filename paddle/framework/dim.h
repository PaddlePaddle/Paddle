#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "paddle/platform/assert.h"
#include "paddle/platform/hostdevice.h"

namespace paddle {
namespace framework {

// Statically sized, statically indexed dimension
template <int i>
struct Dim {
  static constexpr int dimensions = i;

  template <typename... Args>
  HOSTDEVICE Dim(int _head, Args... _tail) : head(_head), tail(_tail...) {
    static_assert(sizeof...(_tail) == i - 1,
                  "Dim initialized with the wrong number of parameters");
  }

  HOSTDEVICE
  Dim(int _head, const Dim<i - 1>& _tail) : head(_head), tail(_tail) {}

  HOSTDEVICE
  Dim() : head(0), tail() {}

  /** Construct a Dim from a linear index and size.  Uses Fortran order
   * indexing. */
  HOSTDEVICE
  Dim(int idx, const Dim<i>& size)
      : head(idx % size.head), tail(idx / size.head, size.tail) {}

  /** Construct a Dim with each dimension set to the given index */
  HOSTDEVICE
  Dim(int idx) : head(idx), tail(idx) {}

  HOSTDEVICE
  bool operator==(const Dim<i>& o) const {
    return (head == o.head) && (tail == o.tail);
  }

  HOSTDEVICE
  bool operator!=(const Dim<i>& o) const { return !(*this == o); }

  HOSTDEVICE
  int& operator[](int idx);
  HOSTDEVICE
  int operator[](int idx) const;

  HOST std::string to_string() const;

  int head;
  Dim<i - 1> tail;
};

// Base case specialization
template <>
struct Dim<1> {
  static constexpr int dimensions = 1;

  HOSTDEVICE
  Dim(int _head) : head(_head) {}

  HOSTDEVICE
  Dim() : head(0) {}

  HOSTDEVICE
  Dim(int idx, const Dim<1>& size) : head(idx) {
#ifndef __CUDA_ARCH__
    if (idx >= size.head) {
      throw std::invalid_argument("Index out of range.");
    }
#else
    PADDLE_ASSERT(idx < size.head);
#endif
  }

  HOSTDEVICE
  bool operator==(const Dim<1>& o) const { return (head == o.head); }

  HOSTDEVICE
  bool operator!=(const Dim<1>& o) const { return !(*this == o); }

  HOSTDEVICE
  int& operator[](int idx);
  HOSTDEVICE
  int operator[](int idx) const;

  int head;
};

namespace {

// Helper for accessing Dim classes
template <int i>
struct DimGetter {
  // Return a copy if Dim is const
  template <typename D>
  HOSTDEVICE static int impl(const D& d) {
    return DimGetter<i - 1>::impl(d.tail);
  }
  // Return a reference if Dim is mutable
  template <typename D>
  HOSTDEVICE static int& impl(D& d) {
    return DimGetter<i - 1>::impl(d.tail);
  }
};

// Eureka! We found the element!
template <>
struct DimGetter<0> {
  // Return a copy if Dim is const
  template <typename D>
  HOSTDEVICE static int impl(const D& d) {
    return d.head;
  }
  // Return a reference if Dim is mutable
  template <typename D>
  HOSTDEVICE static int& impl(D& d) {
    return d.head;
  }
};

template <int D>
HOSTDEVICE int& indexer(Dim<D>& dim, int idx) {
#ifndef __CUDA_ARCH__
  if (idx < 0) {
    throw std::invalid_argument("Tried to access a negative dimension");
  }
#else
  PADDLE_ASSERT(idx >= 0);
#endif
  if (idx == 0) {
    return dim.head;
  }
  return indexer(dim.tail, idx - 1);
}

template <>
HOSTDEVICE int& indexer<1>(Dim<1>& dim, int idx) {
#ifndef __CUDA_ARCH__
  if (idx != 0) {
    throw std::invalid_argument("Invalid index");
  }
#else
  PADDLE_ASSERT(idx == 0);
#endif
  return dim.head;
}

template <int D>
HOSTDEVICE int indexer(const Dim<D>& dim, int idx) {
#ifndef __CUDA_ARCH__
  if (idx < 0) {
    throw std::invalid_argument("Tried to access a negative dimension");
  }
#else
  PADDLE_ASSERT(idx >= 0);
#endif
  if (idx == 0) {
    return dim.head;
  }
  return indexer(dim.tail, idx - 1);
}

template <>
HOSTDEVICE int indexer<1>(const Dim<1>& dim, int idx) {
#ifndef __CUDA_ARCH__
  if (idx != 0) {
    throw std::invalid_argument("Invalid index");
  }
#else
  PADDLE_ASSERT(idx == 0);
#endif
  return dim.head;
}

}  // namespace
// Static access to constant Dim
template <int i, int l>
HOSTDEVICE int get(const Dim<l>& d) {
  return DimGetter<i>::impl(d);
}

// Static access to mutable Dim
template <int i, int l>
HOSTDEVICE int& get(Dim<l>& d) {
  return DimGetter<i>::impl(d);
}

// Dynamic access to constant Dim
template <int l>
HOSTDEVICE int Dim<l>::operator[](int i) const {
  return indexer(*this, i);
}

// Dynamic access to mutable Dim
template <int l>
HOSTDEVICE int& Dim<l>::operator[](int i) {
  return indexer(*this, i);
}

// Dynamic access to constant Dim
inline HOSTDEVICE int Dim<1>::operator[](int i) const {
  return indexer(*this, i);
}

// Dynamic access to mutable Dim
inline HOSTDEVICE int& Dim<1>::operator[](int i) { return indexer(*this, i); }

// Dynamic access to constant Dim
// without std::enable_if will try to instantiate this on get<0>(d)
template <int l>
HOSTDEVICE typename std::enable_if<(l > 0), int>::type get(const Dim<l>& d,
                                                           int i) {
  return d[i];
}

// Dynamic access to mutable Dim
template <int l>
HOSTDEVICE typename std::enable_if<(l > 0), int&>::type get(Dim<l>& d, int i) {
  return d[i];
}

// Dot product of two dims
template <int i>
HOSTDEVICE int linearize(const Dim<i>& a, const Dim<i>& b) {
  return a.head * b.head + linearize(a.tail, b.tail);
}

// Base case dot product of two Dims
// Notice it is inline because it is no longer a template
template <>
HOSTDEVICE inline int linearize(const Dim<1>& a, const Dim<1>& b) {
  return a.head * b.head;
}

// Product of a Dim
template <int i>
HOSTDEVICE int product(const Dim<i>& a, int prod = 1) {
  return prod * a.head * product(a.tail);
}

// Base case product of a Dim
// Notice it is inline because it is no longer a template
template <>
HOSTDEVICE inline int product(const Dim<1>& a, int prod) {
  return prod * a.head;
}

// Is 0 <= idx_i < size_i for all i?
template <int i>
HOSTDEVICE bool contained(const Dim<i>& idx, const Dim<i>& size) {
  return ((0 <= idx.head) && (idx.head < size.head) &&
          contained(idx.tail, size.tail));
}

// Base case of is 0 <= idx_i < size_i ?
// Notice it is inline because it is no longer a template
template <>
HOSTDEVICE inline bool contained(const Dim<1>& idx, const Dim<1>& size) {
  return ((0 <= idx.head) && (idx.head < size.head));
}

/**
 * \brief Check if a size and a stride create a Fortran order contiguous
 * block of memory.
 */
template <int i>
HOST bool contiguous(const Dim<i>& size, const Dim<i>& stride, int mul = 1) {
  if (product(size) == 0) return true;
  int contiguous_stride = get<0>(size) == 1 ? 0 : mul;
  return (get<0>(stride) == contiguous_stride &&
          contiguous(size.tail, stride.tail, mul * get<0>(size)));
}

///\cond HIDDEN
// Base case of contiguous, check the nth stride is the size of
// the prefix multiply of n-1 dims.
template <>
inline bool contiguous(const Dim<1>& size, const Dim<1>& stride, int mul) {
  if (get<0>(size) == 0) return true;
  int contiguous_stride = get<0>(size) == 1 ? 0 : mul;
  return get<0>(stride) == contiguous_stride;
}
///\endcond

/**
 * \brief Compute exclusive prefix-multiply of a Dim.
 */
template <int i>
HOSTDEVICE Dim<i> ex_prefix_mul(const Dim<i>& src, int mul = 1) {
  return Dim<i>(mul, ex_prefix_mul(src.tail, mul * src.head));
}

///\cond HIDDEN
// Base case of ex_prefix_mul
// Notice it is inline because it is no longer a template
template <>
HOSTDEVICE inline Dim<1> ex_prefix_mul(const Dim<1>& src, int mul) {
  return Dim<1>(mul);
}
///\endcond

/**
 * \brief Calculate strides of a contiguous array of the given size
 *
 * Sets the stride for any dimension with an extent of 1 to 0.
 * \param size Dim object containing the size of the array.
 * \param base The base stride to use.
 * \return Dim object the same size as \p size with the strides.
 */
template <int i>
HOSTDEVICE Dim<i> contiguous_strides(const Dim<i>& size, int base = 1) {
  int stride = size.head == 1 ? 0 : base;
  return Dim<i>(stride, contiguous_strides(size.tail, base * size.head));
}

///\cond HIDDEN

// Base case of contiguous_strides
template <>
HOSTDEVICE inline Dim<1> contiguous_strides(const Dim<1>& size, int base) {
  int stride = size.head == 1 ? 0 : base;
  return Dim<1>(stride);
}

///\endcond

/**
 * Add two dimensions together
 */
template <int i>
HOSTDEVICE Dim<i> dim_plus(const Dim<i>& a, const Dim<i>& b) {
  return Dim<i>(a.head + b.head, dim_plus(a.tail, b.tail));
}

// Base case
template <>
HOSTDEVICE inline Dim<1> dim_plus(const Dim<1>& a, const Dim<1>& b) {
  return Dim<1>(a.head + b.head);
}

template <int i>
HOSTDEVICE Dim<i> operator+(const Dim<i>& lhs, const Dim<i>& rhs) {
  return dim_plus(lhs, rhs);
}

/**
 * Multiply two dimensions together
 */
template <int i>
HOSTDEVICE Dim<i> dim_mult(const Dim<i>& a, const Dim<i>& b) {
  return Dim<i>(a.head * b.head, dim_mult(a.tail, b.tail));
}

// Base case
template <>
HOSTDEVICE inline Dim<1> dim_mult(const Dim<1>& a, const Dim<1>& b) {
  return Dim<1>(a.head * b.head);
}

template <int i>
HOSTDEVICE Dim<i> operator*(const Dim<i>& lhs, const Dim<i>& rhs) {
  return dim_mult(lhs, rhs);
}

/**
 * \brief Normalize strides to ensure any dimension with extent 1
 * has stride 0.
 *
 * \param size Dim object containing the size of an array
 * \param stride Dim object containing stride of an array
 * \return Dim object the same size as \p size with normalized strides
 *
 */

template <int i>
HOSTDEVICE Dim<i> normalize_strides(const Dim<i>& size, const Dim<i>& stride) {
  int norm_stride = size.head == 1 ? 0 : stride.head;
  return Dim<i>(norm_stride, normalize_strides(size.tail, stride.tail));
}

///\cond HIDDEN

template <>
HOSTDEVICE inline Dim<1> normalize_strides(const Dim<1>& size,
                                           const Dim<1>& stride) {
  int norm_stride = size.head == 1 ? 0 : stride.head;
  return Dim<1>(norm_stride);
}

///\endcond

/**
 * Helper function to create a Dim
 *
 * \param idxes The type of Dim constructed depends on the number of params
 *
 */

template <typename... Args>
HOSTDEVICE Dim<sizeof...(Args)> make_dim(Args... idxes) {
  return Dim<sizeof...(Args)>(idxes...);
}

// Allows us to output a Dim
// XXX For some reason, overloading fails to resolve this correctly
template <int i>
typename std::enable_if<(i > 1), std::ostream&>::type operator<<(
    std::ostream& os, const Dim<i>& d) {
  os << d.head << ", " << d.tail;
  return os;
}

// Base case that allows us to output a Dim
// XXX I wish this could be an overload instead of a template
template <int i>
typename std::enable_if<(i == 1), std::ostream&>::type operator<<(
    std::ostream& os, const Dim<i>& d) {
  os << d.head;
  return os;
}

template <int i>
HOST std::string Dim<i>::to_string() const {
  std::stringstream stream;

  stream << *this;

  return stream.str();
}

template <int D>
HOSTDEVICE Dim<D> linear_to_dimension(int linear_index, Dim<D> extents) {
  Dim<D> result;

  for (int i = 0; i < D - 1; ++i) {
    result[i] = linear_index % extents[i];
    linear_index /= extents[i];
  }

  result[D - 1] = linear_index;

  return result;
}

}  // namespace framework
}  // namespace paddle

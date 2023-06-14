// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// This file copy from https://github.com/tcbrindle/span
// Modified the following points
// 1. remove macros for backward compatibility with pre-C++17 standards
// 2. instantiated namespace name with paddle

/*
This is an implementation of C++20's std::span
http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/n4820.pdf
*/
//          Copyright Tristan Brindle 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../../LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <type_traits>

#ifdef SPAN_THROW_ON_CONTRACT_VIOLATION
#include <cstdio>
#include <stdexcept>
#endif

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#define NO_DISCARD [[nodiscard]]
#else
#define NO_DISCARD
#endif

namespace paddle {

#if defined(SPAN_THROW_ON_CONTRACT_VIOLATION)
struct contract_violation_error : std::logic_error {
  explicit contract_violation_error(const char* msg) : std::logic_error(msg) {}
};

inline void contract_violation(const char* msg) {
  throw contract_violation_error(msg);
}
#else
[[noreturn]] void contract_violation(const char* /*unused*/) {
  std::terminate();
}
#endif

#if !defined(SPAN_NO_CONTRACT_CHECKING)
#define SPAN_STRINGIFY(cond) #cond
#define SPAN_EXPECT(cond) \
  cond ? (void)0 : contract_violation("Expected " SPAN_STRINGIFY(cond))
#else
#define SPAN_EXPECT(cond)
#endif

#ifdef __cpp_inline_variables
inline constexpr std::size_t dynamic_extent =
    std::numeric_limits<std::size_t>::max();
#else
constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();
#endif

template <typename ElementType, std::size_t Extent = dynamic_extent>
class span;

namespace detail {

#ifdef __cpp_lib_byte
using byte = std::byte;
#else
using byte = unsigned char;
#endif

#ifdef __cpp_lib_nonmember_container_access
using std::data;
using std::size;
#else
template <class C>
constexpr auto size(const C& c) -> decltype(c.size()) {
  return c.size();
}

template <class T, std::size_t N>
constexpr std::size_t size(const T (&)[N]) noexcept {
  return N;
}

template <class C>
constexpr auto data(C& c) -> decltype(c.data()) {  // NOLINT
  return c.data();
}

template <class C>
constexpr auto data(const C& c) -> decltype(c.data()) {
  return c.data();
}

template <class T, std::size_t N>
constexpr T* data(T (&array)[N]) noexcept {
  return array;
}

template <class E>
constexpr const E* data(std::initializer_list<E> il) noexcept {
  return il.begin();
}
#endif

#ifdef __cpp_lib_void_t
using std::void_t;
#else
template <typename...>
using void_t = void;
#endif

template <typename E, std::size_t S>
struct span_storage {
  constexpr span_storage() noexcept = default;
  constexpr span_storage(E* ptr, std::size_t /*unused*/) noexcept : ptr{ptr} {}
  E* ptr{};
  static constexpr std::size_t size{S};
};

template <typename E>
struct span_storage<E, dynamic_extent> {
  constexpr span_storage() noexcept = default;
  constexpr span_storage(E* ptr, std::size_t size) noexcept
      : ptr{ptr}, size{size} {}
  E* ptr{};
  std::size_t size{};
};

template <typename>
struct is_span : std::false_type {};

template <typename T, std::size_t S>
struct is_span<span<T, S>> : std::true_type {};

template <typename>
struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename, typename = void>
struct has_size_and_data : std::false_type {};

template <typename T>
struct has_size_and_data<
    T,
    detail::void_t<decltype(detail::size(std::declval<T>())),
                   decltype(detail::data(std::declval<T>()))>>
    : std::true_type {};

template <typename C,
          typename U = typename std::remove_cv<
              typename std::remove_reference<C>::type>::type>
struct is_container {
  static constexpr bool value = !is_span<U>::value && !is_std_array<U>::value &&
                                !std::is_array<U>::value &&
                                has_size_and_data<C>::value;
};

template <typename, typename, typename = void>
struct is_container_element_type_compatible : std::false_type {};

template <typename T, typename E>
struct is_container_element_type_compatible<
    T,
    E,
    typename std::enable_if<
        !std::is_same<typename std::remove_cv<decltype(detail::data(
                          std::declval<T>()))>::type,
                      void>::value &&
        std::is_convertible<typename std::remove_pointer<decltype(detail::data(
                                std::declval<T>()))>::type (*)[],
                            E (*)[]

                            >::value>::type> : std::true_type {};

template <typename, typename = std::size_t>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, decltype(sizeof(T))> : std::true_type {};
}  // namespace detail

template <typename ElementType, std::size_t Extent>
class span {
  static_assert(std::is_object<ElementType>::value,
                "A span's ElementType must be an object type (not a "
                "reference type or void)");
  static_assert(detail::is_complete<ElementType>::value,
                "A span's ElementType must be a complete type (not a forward "
                "declaration)");
  static_assert(!std::is_abstract<ElementType>::value,
                "A span's ElementType cannot be an abstract class type");
  using storage_type = detail::span_storage<ElementType, Extent>;

 public:
  using element_type = ElementType;
  using value_type = typename std::remove_cv<ElementType>::type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = element_type*;
  using const_pointer = const element_type*;
  using reference = element_type&;
  using const_reference = const element_type&;
  using iterator = pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;

  static constexpr size_type extent = Extent;

  // [span.cons], span constructors, copy, assignment, and destructor
  template <
      std::size_t E = Extent,
      typename std::enable_if<E == dynamic_extent || E == 0, int>::type = 0>
  constexpr span() noexcept {}

  constexpr span(pointer ptr, size_type count) : storage_(ptr, count) {
    SPAN_EXPECT(extent == dynamic_extent || count == extent);
  }

  constexpr span(pointer first_elem, pointer last_elem)
      : storage_(first_elem, last_elem - first_elem) {
    SPAN_EXPECT(extent == dynamic_extent ||
                last_elem - first_elem == static_cast<std::ptrdiff_t>(extent));
  }

  template <
      std::size_t N,
      std::size_t E = Extent,
      typename std::enable_if<
          (E == dynamic_extent || N == E) &&
              detail::is_container_element_type_compatible<element_type (&)[N],
                                                           ElementType>::value,
          int>::type = 0>
  constexpr span(element_type (&arr)[N]) noexcept  // NOLINT
      : storage_(arr, N) {}

  template <
      typename T,
      std::size_t N,
      std::size_t E = Extent,
      typename std::enable_if<
          (E == dynamic_extent || N == E) &&
              detail::is_container_element_type_compatible<std::array<T, N>&,
                                                           ElementType>::value,
          int>::type = 0>
  constexpr span(std::array<T, N>& arr) noexcept  // NOLINT
      : storage_(arr.data(), N) {}

  template <
      typename T,
      std::size_t N,
      std::size_t E = Extent,
      typename std::enable_if<(E == dynamic_extent || N == E) &&
                                  detail::is_container_element_type_compatible<
                                      const std::array<T, N>&,
                                      ElementType>::value,
                              int>::type = 0>
  constexpr span(const std::array<T, N>& arr) noexcept  // NOLINT
      : storage_(arr.data(), N) {}

  template <
      typename Container,
      std::size_t E = Extent,
      typename std::enable_if<
          E == dynamic_extent && detail::is_container<Container>::value &&
              detail::is_container_element_type_compatible<Container&,
                                                           ElementType>::value,
          int>::type = 0>
  constexpr span(Container& cont)  // NOLINT
      : storage_(detail::data(cont), detail::size(cont)) {}

  template <
      typename Container,
      std::size_t E = Extent,
      typename std::enable_if<
          E == dynamic_extent && detail::is_container<Container>::value &&
              detail::is_container_element_type_compatible<const Container&,
                                                           ElementType>::value,
          int>::type = 0>
  constexpr span(const Container& cont)  // NOLINT
      : storage_(detail::data(cont), detail::size(cont)) {}

  constexpr span(const span& other) noexcept = default;

  template <typename OtherElementType,
            std::size_t OtherExtent,
            typename std::enable_if<
                (Extent == dynamic_extent || OtherExtent == dynamic_extent ||
                 Extent == OtherExtent) &&
                    std::is_convertible<OtherElementType (*)[],
                                        ElementType (*)[]>::value,
                int>::type = 0>
  constexpr span(const span<OtherElementType, OtherExtent>& other) noexcept
      : storage_(other.data(), other.size()) {}

  ~span() noexcept = default;

  constexpr span& operator=(const span& other) noexcept = default;

  // [span.sub], span subviews
  template <std::size_t Count>
  constexpr span<element_type, Count> first() const {
    SPAN_EXPECT(Count <= size());
    return {data(), Count};
  }

  template <std::size_t Count>
  constexpr span<element_type, Count> last() const {
    SPAN_EXPECT(Count <= size());
    return {data() + (size() - Count), Count};
  }

  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  using subspan_return_t =
      span<ElementType,
           Count != dynamic_extent
               ? Count
               : (Extent != dynamic_extent ? Extent - Offset : dynamic_extent)>;

  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  constexpr subspan_return_t<Offset, Count> subspan() const {
    SPAN_EXPECT(Offset <= size() &&
                (Count == dynamic_extent || Offset + Count <= size()));
    return {data() + Offset, Count != dynamic_extent ? Count : size() - Offset};
  }

  constexpr span<element_type, dynamic_extent> first(size_type count) const {
    SPAN_EXPECT(count <= size());
    return {data(), count};
  }

  constexpr span<element_type, dynamic_extent> last(size_type count) const {
    SPAN_EXPECT(count <= size());
    return {data() + (size() - count), count};
  }

  constexpr span<element_type, dynamic_extent> subspan(
      size_type offset, size_type count = dynamic_extent) const {
    SPAN_EXPECT(offset <= size() &&
                (count == dynamic_extent || offset + count <= size()));
    return {data() + offset, count == dynamic_extent ? size() - offset : count};
  }

  // [span.obs], span observers
  constexpr size_type size() const noexcept { return storage_.size; }

  constexpr size_type size_bytes() const noexcept {
    return size() * sizeof(element_type);
  }

  NO_DISCARD constexpr bool empty() const noexcept { return size() == 0; }

  // [span.elem], span element access
  constexpr reference operator[](size_type idx) const {
    SPAN_EXPECT(idx < size());
    return *(data() + idx);
  }

  constexpr reference front() const {
    SPAN_EXPECT(!empty());
    return *data();
  }

  constexpr reference back() const {
    SPAN_EXPECT(!empty());
    return *(data() + (size() - 1));
  }

  constexpr pointer data() const noexcept { return storage_.ptr; }

  // [span.iterators], span iterator support
  constexpr iterator begin() const noexcept { return data(); }

  constexpr iterator end() const noexcept { return data() + size(); }

  constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }

  constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

 private:
  storage_type storage_{};
};

#ifdef __cpp_deduction_guides
/* Deduction Guides */
template <typename T, size_t N>
span(T (&)[N]) -> span<T, N>;

template <typename T, size_t N>
span(std::array<T, N>&) -> span<T, N>;

template <typename T, size_t N>
span(const std::array<T, N>&) -> span<const T, N>;

template <typename Container>
span(Container&) -> span<typename std::remove_reference<  // NOLINT
    decltype(*detail::data(std::declval<Container&>()))>::type>;

template <typename Container>
span(const Container&) -> span<const typename Container::value_type>;

#endif

template <typename ElementType, std::size_t Extent>
span<const detail::byte,
     (Extent == dynamic_extent ? dynamic_extent : sizeof(ElementType) * Extent)>
as_bytes(span<ElementType, Extent> s) noexcept {
  return {reinterpret_cast<const detail::byte*>(s.data()), s.size_bytes()};
}

template <
    typename ElementType,
    std::size_t Extent,
    typename std::enable_if<!std::is_const<ElementType>::value, int>::type = 0>
span<detail::byte,
     (Extent == dynamic_extent ? dynamic_extent : sizeof(ElementType) * Extent)>
as_writable_bytes(span<ElementType, Extent> s) noexcept {
  return {reinterpret_cast<detail::byte*>(s.data()), s.size_bytes()};
}

}  // namespace paddle

namespace std {

template <typename ElementType, std::size_t Extent>
class tuple_size<::paddle::span<ElementType, Extent>>
    : public integral_constant<size_t, Extent> {};

template <typename ElementType>
class tuple_size<
    ::paddle::span<ElementType, ::paddle::dynamic_extent>>;  // not defined

template <size_t I, typename ElementType, size_t Extent>
class tuple_element<I, ::paddle::span<ElementType, Extent>> {
 public:
  static_assert(Extent != ::paddle::dynamic_extent && I < Extent, "");
  using type = ElementType;
};

}  // namespace std

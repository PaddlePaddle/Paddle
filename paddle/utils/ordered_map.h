/**
 * Copy from https://github.com/Tessil/ordered-map
 * Modified the following points:
 * 1. modify namespace from `tsl` to `paddle`
 * 2. modify some naming prefixes from `tsl` to `paddle`
 * 3. refine code-format by pre-commit hook
 */

/**
 * MIT License
 *
 * Copyright (c) 2017 Thibaut Goetghebuer-Planchon <tessil@gmx.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "paddle/utils/ordered_hash.h"

namespace paddle {

/**
 * Implementation of an hash map using open addressing with robin hood with
 * backshift delete to resolve collisions.
 *
 * The particularity of this hash map is that it remembers the order in which
 * the elements were added and provide a way to access the structure which
 * stores these values through the 'values_container()' method. The used
 * container is defined by ValueTypeContainer, by default a std::deque is used
 * (grows faster) but a std::vector may be used. In this case the map provides a
 * 'data()' method which give a direct access to the memory used to store the
 * values (which can be useful to communicate with C API's).
 *
 * The Key and T must be copy constructible and/or move constructible. To use
 * `unordered_erase` they both must be swappable.
 *
 * The behaviour of the hash map is undefined if the destructor of Key or T
 * throws an exception.
 *
 * By default the maximum size of a map is limited to 2^32 - 1 values, if needed
 * this can be changed through the IndexType template parameter. Using an
 * `uint64_t` will raise this limit to 2^64 - 1 values but each bucket will use
 * 16 bytes instead of 8 bytes in addition to the space needed to store the
 * values.
 *
 * Iterators invalidation:
 *  - clear, operator=, reserve, rehash: always invalidate the iterators (also
 * invalidate end()).
 *  - insert, emplace, emplace_hint, operator[]: when a std::vector is used as
 * ValueTypeContainer and if size() < capacity(), only end(). Otherwise all the
 * iterators are invalidated if an insert occurs.
 *  - erase, unordered_erase: when a std::vector is used as ValueTypeContainer
 * invalidate the iterator of the erased element and all the ones after the
 * erased element (including end()). Otherwise all the iterators are invalidated
 * if an erase occurs.
 */
template <class Key,
          class T,
          class Hash = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class Allocator = std::allocator<std::pair<Key, T>>,
          class ValueTypeContainer = std::deque<std::pair<Key, T>, Allocator>,
          class IndexType = std::uint_least32_t>
class ordered_map {
 private:
  template <typename U>
  using has_is_transparent = paddle::detail_ordered_hash::has_is_transparent<U>;

  class KeySelect {
   public:
    using key_type = Key;

    const key_type& operator()(const std::pair<Key, T>& key_value) const
        noexcept {
      return key_value.first;
    }

    key_type& operator()(std::pair<Key, T>& key_value) noexcept {  // NOLINT
      return key_value.first;
    }
  };

  class ValueSelect {
   public:
    using value_type = T;

    const value_type& operator()(const std::pair<Key, T>& key_value) const
        noexcept {
      return key_value.second;
    }

    value_type& operator()(std::pair<Key, T>& key_value) noexcept {  // NOLINT
      return key_value.second;
    }
  };

  using ht = detail_ordered_hash::ordered_hash<std::pair<Key, T>,
                                               KeySelect,
                                               ValueSelect,
                                               Hash,
                                               KeyEqual,
                                               Allocator,
                                               ValueTypeContainer,
                                               IndexType>;

 public:
  using key_type = typename ht::key_type;
  using mapped_type = T;
  using value_type = typename ht::value_type;
  using size_type = typename ht::size_type;
  using difference_type = typename ht::difference_type;
  using hasher = typename ht::hasher;
  using key_equal = typename ht::key_equal;
  using allocator_type = typename ht::allocator_type;
  using reference = typename ht::reference;
  using const_reference = typename ht::const_reference;
  using pointer = typename ht::pointer;
  using const_pointer = typename ht::const_pointer;
  using iterator = typename ht::iterator;
  using const_iterator = typename ht::const_iterator;
  using reverse_iterator = typename ht::reverse_iterator;
  using const_reverse_iterator = typename ht::const_reverse_iterator;

  using values_container_type = typename ht::values_container_type;

  /*
   * Constructors
   */
  ordered_map() : ordered_map(ht::DEFAULT_INIT_BUCKETS_SIZE) {}

  explicit ordered_map(size_type bucket_count,
                       const Hash& hash = Hash(),
                       const KeyEqual& equal = KeyEqual(),
                       const Allocator& alloc = Allocator())
      : m_ht(bucket_count, hash, equal, alloc, ht::DEFAULT_MAX_LOAD_FACTOR) {}

  ordered_map(size_type bucket_count, const Allocator& alloc)
      : ordered_map(bucket_count, Hash(), KeyEqual(), alloc) {}

  ordered_map(size_type bucket_count, const Hash& hash, const Allocator& alloc)
      : ordered_map(bucket_count, hash, KeyEqual(), alloc) {}

  explicit ordered_map(const Allocator& alloc)
      : ordered_map(ht::DEFAULT_INIT_BUCKETS_SIZE, alloc) {}

  template <class InputIt>
  ordered_map(InputIt first,
              InputIt last,
              size_type bucket_count = ht::DEFAULT_INIT_BUCKETS_SIZE,
              const Hash& hash = Hash(),
              const KeyEqual& equal = KeyEqual(),
              const Allocator& alloc = Allocator())
      : ordered_map(bucket_count, hash, equal, alloc) {
    insert(first, last);
  }

  template <class InputIt>
  ordered_map(InputIt first,
              InputIt last,
              size_type bucket_count,
              const Allocator& alloc)
      : ordered_map(first, last, bucket_count, Hash(), KeyEqual(), alloc) {}

  template <class InputIt>
  ordered_map(InputIt first,
              InputIt last,
              size_type bucket_count,
              const Hash& hash,
              const Allocator& alloc)
      : ordered_map(first, last, bucket_count, hash, KeyEqual(), alloc) {}

  ordered_map(std::initializer_list<value_type> init,
              size_type bucket_count = ht::DEFAULT_INIT_BUCKETS_SIZE,
              const Hash& hash = Hash(),
              const KeyEqual& equal = KeyEqual(),
              const Allocator& alloc = Allocator())
      : ordered_map(
            init.begin(), init.end(), bucket_count, hash, equal, alloc) {}

  ordered_map(std::initializer_list<value_type> init,
              size_type bucket_count,
              const Allocator& alloc)
      : ordered_map(
            init.begin(), init.end(), bucket_count, Hash(), KeyEqual(), alloc) {
  }

  ordered_map(std::initializer_list<value_type> init,
              size_type bucket_count,
              const Hash& hash,
              const Allocator& alloc)
      : ordered_map(
            init.begin(), init.end(), bucket_count, hash, KeyEqual(), alloc) {}

  ordered_map& operator=(std::initializer_list<value_type> ilist) {
    m_ht.clear();

    m_ht.reserve(ilist.size());
    m_ht.insert(ilist.begin(), ilist.end());

    return *this;
  }

  allocator_type get_allocator() const { return m_ht.get_allocator(); }

  /*
   * Iterators
   */
  iterator begin() noexcept { return m_ht.begin(); }
  const_iterator begin() const noexcept { return m_ht.begin(); }
  const_iterator cbegin() const noexcept { return m_ht.cbegin(); }

  iterator end() noexcept { return m_ht.end(); }
  const_iterator end() const noexcept { return m_ht.end(); }
  const_iterator cend() const noexcept { return m_ht.cend(); }

  reverse_iterator rbegin() noexcept { return m_ht.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return m_ht.rbegin(); }
  const_reverse_iterator rcbegin() const noexcept { return m_ht.rcbegin(); }

  reverse_iterator rend() noexcept { return m_ht.rend(); }
  const_reverse_iterator rend() const noexcept { return m_ht.rend(); }
  const_reverse_iterator rcend() const noexcept { return m_ht.rcend(); }

  /*
   * Capacity
   */
  bool empty() const noexcept { return m_ht.empty(); }
  size_type size() const noexcept { return m_ht.size(); }
  size_type max_size() const noexcept { return m_ht.max_size(); }

  /*
   * Modifiers
   */
  void clear() noexcept { m_ht.clear(); }

  std::pair<iterator, bool> insert(const value_type& value) {
    return m_ht.insert(value);
  }

  template <class P,
            typename std::enable_if<
                std::is_constructible<value_type, P&&>::value>::type* = nullptr>
  std::pair<iterator, bool> insert(P&& value) {
    return m_ht.emplace(std::forward<P>(value));
  }

  std::pair<iterator, bool> insert(value_type&& value) {
    return m_ht.insert(std::move(value));
  }

  iterator insert(const_iterator hint, const value_type& value) {
    return m_ht.insert_hint(hint, value);
  }

  template <class P,
            typename std::enable_if<
                std::is_constructible<value_type, P&&>::value>::type* = nullptr>
  iterator insert(const_iterator hint, P&& value) {
    return m_ht.emplace_hint(hint, std::forward<P>(value));
  }

  iterator insert(const_iterator hint, value_type&& value) {
    return m_ht.insert_hint(hint, std::move(value));
  }

  template <class InputIt>
  void insert(InputIt first, InputIt last) {
    m_ht.insert(first, last);
  }
  void insert(std::initializer_list<value_type> ilist) {
    m_ht.insert(ilist.begin(), ilist.end());
  }

  template <class M>
  std::pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj) {
    return m_ht.insert_or_assign(k, std::forward<M>(obj));
  }

  template <class M>
  std::pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj) {
    return m_ht.insert_or_assign(std::move(k), std::forward<M>(obj));
  }

  template <class M>
  iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj) {
    return m_ht.insert_or_assign(hint, k, std::forward<M>(obj));
  }

  template <class M>
  iterator insert_or_assign(const_iterator hint, key_type&& k, M&& obj) {
    return m_ht.insert_or_assign(hint, std::move(k), std::forward<M>(obj));
  }

  /**
   * Due to the way elements are stored, emplace will need to move or copy the
   * key-value once. The method is equivalent to
   * insert(value_type(std::forward<Args>(args)...));
   *
   * Mainly here for compatibility with the std::unordered_map interface.
   */
  template <class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return m_ht.emplace(std::forward<Args>(args)...);
  }

  /**
   * Due to the way elements are stored, emplace_hint will need to move or copy
   * the key-value once. The method is equivalent to insert(hint,
   * value_type(std::forward<Args>(args)...));
   *
   * Mainly here for compatibility with the std::unordered_map interface.
   */
  template <class... Args>
  iterator emplace_hint(const_iterator hint, Args&&... args) {
    return m_ht.emplace_hint(hint, std::forward<Args>(args)...);
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(const key_type& k, Args&&... args) {
    return m_ht.try_emplace(k, std::forward<Args>(args)...);
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(key_type&& k, Args&&... args) {
    return m_ht.try_emplace(std::move(k), std::forward<Args>(args)...);
  }

  template <class... Args>
  iterator try_emplace(const_iterator hint, const key_type& k, Args&&... args) {
    return m_ht.try_emplace_hint(hint, k, std::forward<Args>(args)...);
  }

  template <class... Args>
  iterator try_emplace(const_iterator hint, key_type&& k, Args&&... args) {
    return m_ht.try_emplace_hint(
        hint, std::move(k), std::forward<Args>(args)...);
  }

  /**
   * When erasing an element, the insert order will be preserved and no holes
   * will be present in the container returned by 'values_container()'.
   *
   * The method is in O(n), if the order is not important 'unordered_erase(...)'
   * method is faster with an O(1) average complexity.
   */
  iterator erase(iterator pos) { return m_ht.erase(pos); }

  /**
   * @copydoc erase(iterator pos)
   */
  iterator erase(const_iterator pos) { return m_ht.erase(pos); }

  /**
   * @copydoc erase(iterator pos)
   */
  iterator erase(const_iterator first, const_iterator last) {
    return m_ht.erase(first, last);
  }

  /**
   * @copydoc erase(iterator pos)
   */
  size_type erase(const key_type& key) { return m_ht.erase(key); }

  /**
   * @copydoc erase(iterator pos)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup to the value if you already have the hash.
   */
  size_type erase(const key_type& key, std::size_t precalculated_hash) {
    return m_ht.erase(key, precalculated_hash);
  }

  /**
   * @copydoc erase(iterator pos)
   *
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type erase(const K& key) {
    return m_ht.erase(key);
  }

  /**
   * @copydoc erase(const key_type& key, std::size_t precalculated_hash)
   *
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type erase(const K& key, std::size_t precalculated_hash) {
    return m_ht.erase(key, precalculated_hash);
  }

  void swap(ordered_map& other) { other.m_ht.swap(m_ht); }

  /*
   * Lookup
   */
  T& at(const Key& key) { return m_ht.at(key); }

  /**
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  T& at(const Key& key, std::size_t precalculated_hash) {
    return m_ht.at(key, precalculated_hash);
  }

  const T& at(const Key& key) const { return m_ht.at(key); }

  /**
   * @copydoc at(const Key& key, std::size_t precalculated_hash)
   */
  const T& at(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.at(key, precalculated_hash);
  }

  /**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  T& at(const K& key) {
    return m_ht.at(key);
  }

  /**
   * @copydoc at(const K& key)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  T& at(const K& key, std::size_t precalculated_hash) {
    return m_ht.at(key, precalculated_hash);
  }

  /**
   * @copydoc at(const K& key)
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const T& at(const K& key) const {
    return m_ht.at(key);
  }

  /**
   * @copydoc at(const K& key, std::size_t precalculated_hash)
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const T& at(const K& key, std::size_t precalculated_hash) const {
    return m_ht.at(key, precalculated_hash);
  }

  T& operator[](const Key& key) { return m_ht[key]; }
  T& operator[](Key&& key) { return m_ht[std::move(key)]; }

  size_type count(const Key& key) const { return m_ht.count(key); }

  /**
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  size_type count(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.count(key, precalculated_hash);
  }

  /**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type count(const K& key) const {
    return m_ht.count(key);
  }

  /**
   * @copydoc count(const K& key) const
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type count(const K& key, std::size_t precalculated_hash) const {
    return m_ht.count(key, precalculated_hash);
  }

  iterator find(const Key& key) { return m_ht.find(key); }

  /**
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  iterator find(const Key& key, std::size_t precalculated_hash) {
    return m_ht.find(key, precalculated_hash);
  }

  const_iterator find(const Key& key) const { return m_ht.find(key); }

  /**
   * @copydoc find(const Key& key, std::size_t precalculated_hash)
   */
  const_iterator find(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.find(key, precalculated_hash);
  }

  /**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  iterator find(const K& key) {
    return m_ht.find(key);
  }

  /**
   * @copydoc find(const K& key)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  iterator find(const K& key, std::size_t precalculated_hash) {
    return m_ht.find(key, precalculated_hash);
  }

  /**
   * @copydoc find(const K& key)
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const_iterator find(const K& key) const {
    return m_ht.find(key);
  }

  /**
   * @copydoc find(const K& key)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const_iterator find(const K& key, std::size_t precalculated_hash) const {
    return m_ht.find(key, precalculated_hash);
  }

  bool contains(const Key& key) const { return m_ht.contains(key); }

  /**
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  bool contains(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.contains(key, precalculated_hash);
  }

  /**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  bool contains(const K& key) const {
    return m_ht.contains(key);
  }

  /**
   * @copydoc contains(const K& key) const
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  bool contains(const K& key, std::size_t precalculated_hash) const {
    return m_ht.contains(key, precalculated_hash);
  }

  std::pair<iterator, iterator> equal_range(const Key& key) {
    return m_ht.equal_range(key);
  }

  /**
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  std::pair<iterator, iterator> equal_range(const Key& key,
                                            std::size_t precalculated_hash) {
    return m_ht.equal_range(key, precalculated_hash);
  }

  std::pair<const_iterator, const_iterator> equal_range(const Key& key) const {
    return m_ht.equal_range(key);
  }

  /**
   * @copydoc equal_range(const Key& key, std::size_t precalculated_hash)
   */
  std::pair<const_iterator, const_iterator> equal_range(
      const Key& key, std::size_t precalculated_hash) const {
    return m_ht.equal_range(key, precalculated_hash);
  }

  /**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<iterator, iterator> equal_range(const K& key) {
    return m_ht.equal_range(key);
  }

  /**
   * @copydoc equal_range(const K& key)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<iterator, iterator> equal_range(const K& key,
                                            std::size_t precalculated_hash) {
    return m_ht.equal_range(key, precalculated_hash);
  }

  /**
   * @copydoc equal_range(const K& key)
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<const_iterator, const_iterator> equal_range(const K& key) const {
    return m_ht.equal_range(key);
  }

  /**
   * @copydoc equal_range(const K& key, std::size_t precalculated_hash)
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<const_iterator, const_iterator> equal_range(
      const K& key, std::size_t precalculated_hash) const {
    return m_ht.equal_range(key, precalculated_hash);
  }

  /*
   * Bucket interface
   */
  size_type bucket_count() const { return m_ht.bucket_count(); }
  size_type max_bucket_count() const { return m_ht.max_bucket_count(); }

  /*
   * Hash policy
   */
  float load_factor() const { return m_ht.load_factor(); }
  float max_load_factor() const { return m_ht.max_load_factor(); }
  void max_load_factor(float ml) { m_ht.max_load_factor(ml); }

  void rehash(size_type count) { m_ht.rehash(count); }
  void reserve(size_type count) { m_ht.reserve(count); }

  /*
   * Observers
   */
  hasher hash_function() const { return m_ht.hash_function(); }
  key_equal key_eq() const { return m_ht.key_eq(); }

  /*
   * Other
   */

  /**
   * Convert a const_iterator to an iterator.
   */
  iterator mutable_iterator(const_iterator pos) {
    return m_ht.mutable_iterator(pos);
  }

  /**
   * Requires index <= size().
   *
   * Return an iterator to the element at index. Return end() if index ==
   * size().
   */
  iterator nth(size_type index) { return m_ht.nth(index); }

  /**
   * @copydoc nth(size_type index)
   */
  const_iterator nth(size_type index) const { return m_ht.nth(index); }

  /**
   * Return const_reference to the first element. Requires the container to not
   * be empty.
   */
  const_reference front() const { return m_ht.front(); }

  /**
   * Return const_reference to the last element. Requires the container to not
   * be empty.
   */
  const_reference back() const { return m_ht.back(); }

  /**
   * Only available if ValueTypeContainer is a std::vector. Same as calling
   * 'values_container().data()'.
   */
  template <class U = values_container_type,
            typename std::enable_if<paddle::detail_ordered_hash::is_vector<
                U>::value>::type* = nullptr>
  const typename values_container_type::value_type* data() const noexcept {
    return m_ht.data();
  }

  /**
   * Return the container in which the values are stored. The values are in the
   * same order as the insertion order and are contiguous in the structure, no
   * holes (size() == values_container().size()).
   */
  const values_container_type& values_container() const noexcept {
    return m_ht.values_container();
  }

  template <class U = values_container_type,
            typename std::enable_if<paddle::detail_ordered_hash::is_vector<
                U>::value>::type* = nullptr>
  size_type capacity() const noexcept {
    return m_ht.capacity();
  }

  void shrink_to_fit() { m_ht.shrink_to_fit(); }

  /**
   * Insert the value before pos shifting all the elements on the right of pos
   * (including pos) one position to the right.
   *
   * Amortized linear time-complexity in the distance between pos and end().
   */
  std::pair<iterator, bool> insert_at_position(const_iterator pos,
                                               const value_type& value) {
    return m_ht.insert_at_position(pos, value);
  }

  /**
   * @copydoc insert_at_position(const_iterator pos, const value_type& value)
   */
  std::pair<iterator, bool> insert_at_position(const_iterator pos,
                                               value_type&& value) {
    return m_ht.insert_at_position(pos, std::move(value));
  }

  /**
   * @copydoc insert_at_position(const_iterator pos, const value_type& value)
   *
   * Same as insert_at_position(pos, value_type(std::forward<Args>(args)...),
   * mainly here for coherence.
   */
  template <class... Args>
  std::pair<iterator, bool> emplace_at_position(const_iterator pos,
                                                Args&&... args) {
    return m_ht.emplace_at_position(pos, std::forward<Args>(args)...);
  }

  /**
   * @copydoc insert_at_position(const_iterator pos, const value_type& value)
   */
  template <class... Args>
  std::pair<iterator, bool> try_emplace_at_position(const_iterator pos,
                                                    const key_type& k,
                                                    Args&&... args) {
    return m_ht.try_emplace_at_position(pos, k, std::forward<Args>(args)...);
  }

  /**
   * @copydoc insert_at_position(const_iterator pos, const value_type& value)
   */
  template <class... Args>
  std::pair<iterator, bool> try_emplace_at_position(const_iterator pos,
                                                    key_type&& k,
                                                    Args&&... args) {
    return m_ht.try_emplace_at_position(
        pos, std::move(k), std::forward<Args>(args)...);
  }

  void pop_back() { m_ht.pop_back(); }

  /**
   * Faster erase operation with an O(1) average complexity but it doesn't
   * preserve the insertion order.
   *
   * If an erasure occurs, the last element of the map will take the place of
   * the erased element.
   */
  iterator unordered_erase(iterator pos) { return m_ht.unordered_erase(pos); }

  /**
   * @copydoc unordered_erase(iterator pos)
   */
  iterator unordered_erase(const_iterator pos) {
    return m_ht.unordered_erase(pos);
  }

  /**
   * @copydoc unordered_erase(iterator pos)
   */
  size_type unordered_erase(const key_type& key) {
    return m_ht.unordered_erase(key);
  }

  /**
   * @copydoc unordered_erase(iterator pos)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  size_type unordered_erase(const key_type& key,
                            std::size_t precalculated_hash) {
    return m_ht.unordered_erase(key, precalculated_hash);
  }

  /**
   * @copydoc unordered_erase(iterator pos)
   *
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type unordered_erase(const K& key) {
    return m_ht.unordered_erase(key);
  }

  /**
   * @copydoc unordered_erase(const K& key)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
  template <
      class K,
      class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type unordered_erase(const K& key, std::size_t precalculated_hash) {
    return m_ht.unordered_erase(key, precalculated_hash);
  }

  /**
   * Serialize the map through the `serializer` parameter.
   *
   * The `serializer` parameter must be a function object that supports the
   * following call:
   *  - `template<typename U> void operator()(const U& value);` where the types
   * `std::uint64_t`, `float` and `std::pair<Key, T>` must be supported for U.
   *
   * The implementation leaves binary compatibility (endianness, IEEE 754 for
   * floats, ...) of the types it serializes in the hands of the `Serializer`
   * function object if compatibility is required.
   */
  template <class Serializer>
  void serialize(Serializer& serializer) const {  // NOLINT
    m_ht.serialize(serializer);
  }

  /**
   * Deserialize a previously serialized map through the `deserializer`
   * parameter.
   *
   * The `deserializer` parameter must be a function object that supports the
   * following calls:
   *  - `template<typename U> U operator()();` where the types `std::uint64_t`,
   * `float` and `std::pair<Key, T>` must be supported for U.
   *
   * If the deserialized hash map type is hash compatible with the serialized
   * map, the deserialization process can be sped up by setting
   * `hash_compatible` to true. To be hash compatible, the Hash and KeyEqual
   * must behave the same way than the ones used on the serialized map. The
   * `std::size_t` must also be of the same size as the one on the platform used
   * to serialize the map, the same apply for `IndexType`. If these criteria are
   * not met, the behaviour is undefined with `hash_compatible` sets to true.
   *
   * The behaviour is undefined if the type `Key` and `T` of the `ordered_map`
   * are not the same as the types used during serialization.
   *
   * The implementation leaves binary compatibility (endianness, IEEE 754 for
   * floats, size of int, ...) of the types it deserializes in the hands of the
   * `Deserializer` function object if compatibility is required.
   */
  template <class Deserializer>
  static ordered_map deserialize(Deserializer& deserializer,  // NOLINT
                                 bool hash_compatible = false) {
    ordered_map map(0);
    map.m_ht.deserialize(deserializer, hash_compatible);

    return map;
  }

  friend bool operator==(const ordered_map& lhs, const ordered_map& rhs) {
    return lhs.m_ht == rhs.m_ht;
  }
  friend bool operator!=(const ordered_map& lhs, const ordered_map& rhs) {
    return lhs.m_ht != rhs.m_ht;
  }
  friend bool operator<(const ordered_map& lhs, const ordered_map& rhs) {
    return lhs.m_ht < rhs.m_ht;
  }
  friend bool operator<=(const ordered_map& lhs, const ordered_map& rhs) {
    return lhs.m_ht <= rhs.m_ht;
  }
  friend bool operator>(const ordered_map& lhs, const ordered_map& rhs) {
    return lhs.m_ht > rhs.m_ht;
  }
  friend bool operator>=(const ordered_map& lhs, const ordered_map& rhs) {
    return lhs.m_ht >= rhs.m_ht;
  }

  friend void swap(ordered_map& lhs, ordered_map& rhs) { lhs.swap(rhs); }

 private:
  ht m_ht;
};

}  // end namespace paddle

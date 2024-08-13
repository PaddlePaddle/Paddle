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

#include <iterator>
#include <list>

#include "paddle/common/macros.h"

namespace pir {

class Operation;
///
/// \brief Value Iterator
///
template <typename OperandType>
class ValueUseIterator {
 public:
  ValueUseIterator(OperandType use = nullptr) : current_(use) {}  // NOLINT

  bool operator==(const ValueUseIterator<OperandType>& rhs) const {
    return current_ == rhs.current_;
  }
  bool operator!=(const ValueUseIterator<OperandType>& rhs) const {
    return !(*this == rhs);
  }

  Operation* owner() const { return current_.owner(); }

  OperandType& operator*() { return current_; }

  OperandType* operator->() { return &operator*(); }

  ValueUseIterator<OperandType>& operator++() {
    current_ = current_.next_use();
    return *this;
  }

  ValueUseIterator<OperandType> operator++(int) {
    ValueUseIterator<OperandType> tmp = *this;
    current_ = current_.next_use();
    return tmp;
  }

 protected:
  OperandType current_;
};

///
/// \brief The wrapper for std::list<ElementType*>::iterator
///
template <typename ElementType>
class PointerListIterator {
  typename std::list<ElementType*>::iterator iterator_;

 public:
  // use to support std::next, std::prev. std::advance
  typedef ptrdiff_t difference_type;
  typedef std::bidirectional_iterator_tag iterator_category;
  typedef ElementType value_type;
  typedef value_type* pointer;
  typedef value_type& reference;

  PointerListIterator() = default;
  PointerListIterator(
      const typename std::list<ElementType*>::iterator& iter)  // NOLINT
      : iterator_(iter) {}

  ElementType& operator*() const noexcept { return **iterator_; }

  ElementType* operator->() const noexcept { return &this->operator*(); }

  PointerListIterator& operator++() noexcept {
    ++iterator_;
    return *this;
  }
  PointerListIterator operator++(int) noexcept {
    PointerListIterator __tmp = *this;
    ++iterator_;
    return __tmp;
  }

  PointerListIterator& operator--() noexcept {
    --iterator_;
    return *this;
  }

  PointerListIterator operator--(int) noexcept {
    PointerListIterator __tmp = *this;
    iterator_--;
    return __tmp;
  }

  bool operator==(const PointerListIterator& __x) const noexcept {
    return iterator_ == __x.iterator_;
  }

  bool operator!=(const PointerListIterator& __x) const noexcept {
    return iterator_ != __x.iterator_;
  }

  void set_underlying_pointer(ElementType* ptr) { *iterator_ = ptr; }

  operator typename std::list<ElementType*>::iterator() const {
    return iterator_;
  }
  operator typename std::list<ElementType*>::const_iterator() const {
    return iterator_;
  }

  // If iterator do not point to a element, it is unsafe.
  operator ElementType*() const { return *iterator_; }
};

///
/// \brief The wrapper for std::list<ElementType*>::const_iterator
///
template <typename ElementType>
class PointerListConstIterator {
  typename std::list<ElementType*>::const_iterator iterator_;

 public:
  // use to support std::next, std::prev. std::advance
  typedef ptrdiff_t difference_type;
  typedef std::bidirectional_iterator_tag iterator_category;
  typedef ElementType value_type;
  typedef value_type* pointer;
  typedef value_type& reference;

  PointerListConstIterator() = default;
  PointerListConstIterator(
      const PointerListIterator<ElementType>& iter)  // NOLINT
      : iterator_(
            static_cast<typename std::list<ElementType*>::iterator>(iter)) {}
  PointerListConstIterator(
      const typename std::list<ElementType*>::iterator& iter)
      : iterator_(iter) {}  // NOLINT
  PointerListConstIterator(
      const typename std::list<ElementType*>::const_iterator& iter)
      : iterator_(iter) {}  // NOLINT

  ElementType& operator*() const noexcept { return **iterator_; }

  ElementType* operator->() const noexcept { return &this->operator*(); }

  PointerListConstIterator& operator++() noexcept {
    ++iterator_;
    return *this;
  }
  PointerListConstIterator operator++(int) noexcept {
    PointerListConstIterator __tmp = *this;
    ++iterator_;
    return __tmp;
  }

  PointerListConstIterator& operator--() noexcept {
    --iterator_;
    return *this;
  }

  PointerListConstIterator operator--(int) noexcept {
    PointerListConstIterator __tmp = *this;
    iterator_--;
    return __tmp;
  }

  bool operator==(const PointerListConstIterator& __x) const noexcept {
    return iterator_ == __x.iterator_;
  }

  bool operator!=(const PointerListConstIterator& __x) const noexcept {
    return iterator_ != __x.iterator_;
  }

  operator typename std::list<ElementType*>::const_iterator() const {
    return iterator_;
  }
  // If iterator do not point to a element, it is unsafe.
  operator ElementType*() const { return *iterator_; }
};

///
/// \brief The DoubleLevelContainer used to flatten two-level containers into
/// one level.
///
template <typename ContainerT>
class DoubleLevelContainer {
 public:
  class Iterator;
  Iterator begin();
  Iterator end();

 protected:
  // only support constructed by derived class : ContainerT;
  DoubleLevelContainer() = default;
  DISABLE_COPY_AND_ASSIGN(DoubleLevelContainer);
  const ContainerT& container() const {
    return *static_cast<const ContainerT*>(this);
  }
  ContainerT& container() { return *static_cast<ContainerT*>(this); }
};
template <typename ContainerT>
class DoubleLevelContainer<ContainerT>::Iterator {
 public:
  using OuterIterator = typename ContainerT::Iterator;
  using InnerIterator = typename ContainerT::Element::Iterator;
  using Element = typename ContainerT::Element::Element;

  Iterator() = default;

  Element& operator*() const noexcept { return *inner_iter_; }

  Element* operator->() const noexcept { return &this->operator*(); }

  Iterator& operator++() noexcept {
    ++inner_iter_;
    while (inner_iter_ == outer_iter_->end()) {
      ++outer_iter_;
      if (outer_iter_ == outer_end_) break;
      inner_iter_ = outer_iter_->begin();
    }
    return *this;
  }
  Iterator operator++(int) noexcept {
    Iterator __tmp = *this;
    ++*this;
    return __tmp;
  }

  Iterator& operator--() noexcept {
    if (outer_iter_ == outer_end_) {
      outer_iter_--;
      inner_iter_ = outer_iter_->end();
    }
    while (inner_iter_ == outer_iter_->begin()) {
      --outer_iter_;
      inner_iter_ = outer_iter_->end();
    }
    --inner_iter_;
    return *this;
  }

  Iterator operator--(int) noexcept {
    Iterator __tmp = *this;
    --*this;
    return __tmp;
  }

  bool operator==(const Iterator& __x) const noexcept {
    return outer_iter_ == __x.outer_iter_ &&
           (outer_iter_ == outer_end_ || inner_iter_ == __x.inner_iter_);
  }

  bool operator!=(const Iterator& __x) const noexcept {
    return !this->operator==(__x);
  }

 private:
  friend class DoubleLevelContainer<ContainerT>;

  // only used by DoubleLevelContainer<ContainerT>::begin() && end();
  Iterator(const OuterIterator& outer_iter,
           const OuterIterator& outer_end,
           const InnerIterator& inner_iter = InnerIterator())
      : outer_iter_(outer_iter),
        outer_end_(outer_end),
        inner_iter_(inner_iter) {}

  OuterIterator outer_iter_, outer_end_;
  InnerIterator inner_iter_;
};
template <typename ContainerT>
typename DoubleLevelContainer<ContainerT>::Iterator
DoubleLevelContainer<ContainerT>::begin() {
  auto outer_iter = container().begin();
  while (outer_iter != container().end()) {
    if (outer_iter->empty()) {
      ++outer_iter;
    } else {
      return Iterator(outer_iter, container().end(), outer_iter->begin());
    }
  }
  return Iterator(outer_iter, container().end());
}

template <typename ContainerT>
typename DoubleLevelContainer<ContainerT>::Iterator
DoubleLevelContainer<ContainerT>::end() {
  return Iterator(container().end(), container().end());
}
}  // namespace pir

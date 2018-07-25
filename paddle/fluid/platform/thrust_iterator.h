// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace paddle {
namespace platform {

#define DEFINE_THRUST_ITERATOR_ADAPTOR_TYPES(iterator, underlying_iterator) \
 private:                                                                   \
  friend class ::thrust::iterator_core_access;                              \
  typedef underlying_iterator UnderlyingIterator;                           \
  typedef iterator ThisIterator;                                            \
  typedef ::thrust::iterator_adaptor<ThisIterator, UnderlyingIterator>      \
      BaseIterator;                                                         \
  typedef typename BaseIterator::difference_type DifferenceType;            \
  typedef typename BaseIterator::reference Reference

template <typename Iterator>
class RepeatIterator
    : public ::thrust::iterator_adaptor<RepeatIterator<Iterator>, Iterator> {
  DEFINE_THRUST_ITERATOR_ADAPTOR_TYPES(RepeatIterator, Iterator);

 public:
  __host__ __device__ RepeatIterator(const UnderlyingIterator &it,
                                     DifferenceType rep_num)
      : BaseIterator(it), idx_(0), rep_num_(rep_num) {}

  __host__ __device__ DifferenceType GetRepeatNum() const { return rep_num_; }

 private:
  DifferenceType idx_;
  DifferenceType rep_num_;

  __host__ __device__ Reference dereference() const {
    return *(this->base() + idx_ / rep_num_);
  }

  __host__ __device__ bool equal(const ThisIterator &other) const {
    return distance_to(other) == 0;
  }

  __host__ __device__ void advance(DifferenceType n) { idx_ += n; }

  __host__ __device__ void increment() { ++idx_; }

  __host__ __device__ void decrement() { --idx_; }

  __host__ __device__ DifferenceType
  distance_to(const ThisIterator &other) const {
    return (other.base() - this->base()) * rep_num_ + other.idx_ - idx_;
  }
};

template <typename Iterator, typename... Args>
__host__ __device__ RepeatIterator<Iterator> MakeRepeatIterator(
    const Iterator &it, Args &&... args) {
  return RepeatIterator<Iterator>(it, std::forward<Args>(args)...);
}

template <typename Iterator>
class CloneIterator
    : public ::thrust::iterator_adaptor<CloneIterator<Iterator>, Iterator> {
  DEFINE_THRUST_ITERATOR_ADAPTOR_TYPES(CloneIterator, Iterator);

 public:
  __host__ __device__ CloneIterator(const Iterator &begin, DifferenceType len,
                                    DifferenceType clone_num)
      : BaseIterator(begin), idx_(0), len_(len), clone_num_(clone_num) {}

  __host__ __device__ CloneIterator(const Iterator &begin, const Iterator &end,
                                    DifferenceType clone_num)
      : CloneIterator(begin, end - begin, clone_num) {}

  __host__ __device__ DifferenceType GetCloneNum() const { return clone_num_; }

 private:
  DifferenceType idx_;
  DifferenceType len_;
  DifferenceType clone_num_;

  __host__ __device__ Reference dereference() const {
    return *(this->base() + idx_ % len_);
  }

  __host__ __device__ bool equal(const ThisIterator &other) const {
    return distance_to(other) == 0;
  }

  __host__ __device__ void advance(DifferenceType n) { idx_ += n; }

  __host__ __device__ void increment() { ++idx_; }

  __host__ __device__ void decrement() { --idx_; }

  __host__ __device__ DifferenceType
  distance_to(const ThisIterator &other) const {
    return (other.base() - this->base()) * clone_num_ + other.idx_ - idx_;
  }
};

template <typename Iterator, typename... Args>
__host__ __device__ CloneIterator<Iterator> MakeCloneIterator(
    const Iterator &it, Args &&... args) {
  return CloneIterator<Iterator>(it, std::forward<Args>(args)...);
}

template <typename Iterator, typename IndexConvertFunctor>
class ConvertIndexIterator
    : public ::thrust::iterator_adaptor<
          ConvertIndexIterator<Iterator, IndexConvertFunctor>, Iterator> {
  DEFINE_THRUST_ITERATOR_ADAPTOR_TYPES(ConvertIndexIterator, Iterator);

 public:
  __host__ __device__ ConvertIndexIterator(const UnderlyingIterator &it,
                                           const IndexConvertFunctor &functor)
      : BaseIterator(it), idx_(0), functor_(functor) {}

  __host__ __device__ const IndexConvertFunctor &GetConverter() const {
    return functor_;
  }

 private:
  DifferenceType idx_;
  const IndexConvertFunctor functor_;

  __host__ __device__ Reference dereference() const {
    return *(this->base() + functor_(idx_));
  }

  __host__ __device__ bool equal(const ThisIterator &other) const {
    return distance_to(other) == 0;
  }

  __host__ __device__ void advance(DifferenceType n) { idx_ += n; }

  __host__ __device__ void increment() { ++idx_; }

  __host__ __device__ void decrement() { --idx_; }

  __host__ __device__ DifferenceType
  distance_to(const ThisIterator &other) const {
    return functor_.Distance(this->base(), functor_(idx_), other.base(),
                             functor_(other.idx_));
  }
};

template <typename Iterator, typename IndexConvertFunctor, typename... Args>
__host__ __device__ ConvertIndexIterator<Iterator, IndexConvertFunctor>
MakeConvertIndexIterator(const Iterator &it, const IndexConvertFunctor &functor,
                         Args &&... args) {
  return ConvertIndexIterator<Iterator, IndexConvertFunctor>(
      it, functor, std::forward<Args>(args)...);
}

template <typename IntType>
struct DivideFunctor {
  __host__ __device__ explicit DivideFunctor(IntType num) : num_(num) {}

  __host__ __device__ IntType operator()(IntType n) { return n / num_; }

  __host__ __device__ IntType GetBaseNum() const { return num_; }

 private:
  const IntType num_;
};

template <typename IntType>
struct ModFunctor {
  __host__ __device__ explicit ModFunctor(IntType num) : num_(num) {}

  __host__ __device__ IntType operator()(IntType n) { return n % num_; }

  __host__ __device__ IntType GetBaseNum() const { return num_; }

 private:
  const IntType num_;
};

template <typename IntType>
using RowIndexMatrixIterator =
    ::thrust::transform_iterator<DivideFunctor<IntType>,
                                 ::thrust::counting_iterator<IntType>>;

template <typename IntType>
using ColIndexMatrixIterator =
    ::thrust::transform_iterator<ModFunctor<IntType>,
                                 ::thrust::counting_iterator<IntType>>;

template <typename IntType>
__host__ __device__ inline RowIndexMatrixIterator<IntType>
MakeRowIndexMatrixIterator(IntType start, IntType cols) {
  return ::thrust::make_transform_iterator(
      ::thrust::make_counting_iterator(start * cols),
      DivideFunctor<IntType>(cols));
}

template <typename IntType>
__host__ __device__ inline ColIndexMatrixIterator<IntType>
MakeColIndexMatrixIterator(IntType start, IntType rows) {
  return ::thrust::make_transform_iterator(
      ::thrust::make_counting_iterator(start * rows),
      ModFunctor<IntType>(rows));
}

template <typename... Iterators>
__host__ __device__ inline auto MakeZipIterator(Iterators &&... iterators)
    -> decltype(::thrust::make_zip_iterator(
        ::thrust::make_tuple(std::forward<Iterators>(iterators)...))) {
  return ::thrust::make_zip_iterator(
      ::thrust::make_tuple(std::forward<Iterators>(iterators)...));
}

template <typename Functor, typename... Iterators>
__host__ __device__ inline auto MakeZipTransformIterator(
    const Functor &functor, Iterators &&... iterators)
    -> decltype(::thrust::make_transform_iterator(
        MakeZipIterator(std::forward<Iterators>(iterators)...), functor)) {
  return ::thrust::make_transform_iterator(
      MakeZipIterator(std::forward<Iterators>(iterators)...), functor);
}

#undef DEFINE_THRUST_ITERATOR_ADAPTOR_TYPES

}  // namespace platform
}  // namespace paddle

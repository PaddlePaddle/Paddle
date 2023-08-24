// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::equation::config {

class Context {
 public:
  Context(const Context&) = delete;
  Context(Context&&) = delete;

  explicit Context(const std::vector<std::uint64_t>& in_tensors_ranks,
                   const std::vector<std::uint64_t>& out_tensors_ranks,
                   const std::shared_ptr<std::vector<Equation>>& equations)
      : in_tensors_ranks_(in_tensors_ranks),
        out_tensors_ranks_(out_tensors_ranks),
        equations_(equations) {
    Init(&in_iterator_tuples_, in_tensors_ranks);
    Init(&out_iterator_tuples_, out_tensors_ranks);
    Init(&in_stride_tuples_, in_tensors_ranks);
    Init(&out_stride_tuples_, out_tensors_ranks);
    Init(&in_dim_tuples_, in_tensors_ranks);
    Init(&out_dim_tuples_, out_tensors_ranks);

    GenerateDots();
  }

  const std::vector<std::uint64_t>& GetInTensorsRanks() const {
    return in_tensors_ranks_;
  }

  const std::vector<std::uint64_t>& GetOutTensorsRanks() const {
    return out_tensors_ranks_;
  }

  void Equal(const Iterator& lhs, const Iterator& rhs) {
    this->Equal<Iterator>(lhs, rhs);
  }

  void Equal(const Index& lhs, const Index& rhs) {
    this->Equal<Index>(lhs, rhs);
  }

  void Equal(const IteratorTuple& lhs, const IteratorTuple& rhs) {
    CHECK(lhs.size() == rhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
      this->Equal(lhs.at(i), rhs.at(i));
    }
  }

  const IteratorTuple& GetInIteratorTuple(std::size_t input_idx) const {
    return in_iterator_tuples_.at(input_idx);
  }

  const IteratorTuple& GetOutIteratorTuple(std::size_t output_idx) const {
    return out_iterator_tuples_.at(output_idx);
  }

  const Index& GetInIndex(std::size_t input_idx) const {
    return in_indexes_.at(input_idx);
  }

  const Index& GetOutIndex(std::size_t output_idx) const {
    return out_indexes_.at(output_idx);
  }

  const StrideTuple& GetInStrideTuple(std::size_t input_idx) const {
    return in_stride_tuples_.at(input_idx);
  }

  const StrideTuple& GetOutStrideTuple(std::size_t output_idx) const {
    return out_stride_tuples_.at(output_idx);
  }

  const DimTuple& GetInDimTuple(std::size_t input_idx) const {
    return in_dim_tuples_.at(input_idx);
  }

  const DimTuple& GetOutDimTuple(std::size_t output_idx) const {
    return out_dim_tuples_.at(output_idx);
  }

 private:
  template <typename ContainerT>
  void Init(std::vector<ContainerT>* vec,
            const std::vector<std::uint64_t>& tensors_ranks) {
    for (std::size_t i = 0; i < tensors_ranks.size(); ++i) {
      vec->push_back(ContainerT{});
      for (std::size_t j = 0; j < tensors_ranks.at(i); ++j) {
        vec->at(i).push_back(ContainerT::value_type{UniqueId::New()});
      }
    }
  }

  Index Dot(const IteratorTuple& iterator_tuple,
            const StrideTuple& stride_tuple) {
    CHECK(iterator_tuple.size() == stride_tuple.size());
    Index index{UniqueId::New()};
    equations_->emplace_back(
        equation::Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>(
            stride_tuple, index, iterator_tuple));
    equations_->emplace_back(
        equation::UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>(
            stride_tuple, iterator_tuple, index));
    return index;
  }

  void GenerateDots() {
    for (std::size_t i = 0; i < in_tensors_ranks_.size(); ++i) {
      Index index{UniqueId::New()};
      in_indexes_.emplace_back(index);
      Equal(index, Dot(GetInIteratorTuple(i), GetInStrideTuple(i)));
    }
    for (std::size_t i = 0; i < out_tensors_ranks_.size(); ++i) {
      Index index{UniqueId::New()};
      out_indexes_.emplace_back(index);
      Equal(index, Dot(GetOutIteratorTuple(i), GetOutStrideTuple(i)));
    }
  }

  template <typename T>
  void Equal(const T& lhs, const T& rhs) {
    equations_->emplace_back(Identity<tOut<T>, tIn<T>>(lhs, rhs));
    equations_->emplace_back(Identity<tOut<T>, tIn<T>>(rhs, lhs));
  }

  std::vector<std::uint64_t> in_tensors_ranks_;
  std::vector<std::uint64_t> out_tensors_ranks_;
  std::vector<IteratorTuple> in_iterator_tuples_;
  std::vector<IteratorTuple> out_iterator_tuples_;
  std::vector<Index> in_indexes_;
  std::vector<Index> out_indexes_;
  std::vector<StrideTuple> in_stride_tuples_;
  std::vector<StrideTuple> out_stride_tuples_;
  std::vector<DimTuple> in_dim_tuples_;
  std::vector<DimTuple> out_dim_tuples_;

  std::shared_ptr<std::vector<Equation>> equations_;
};

}  // namespace cinn::adt::equation::config

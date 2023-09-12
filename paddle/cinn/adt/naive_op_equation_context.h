// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <optional>
#include <vector>

#include "glog/logging.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::equation::config {

class NativeOpEquationContext final : public OpEquationContext {
 public:
  NativeOpEquationContext(const NativeOpEquationContext&) = delete;
  NativeOpEquationContext(NativeOpEquationContext&&) = delete;

  explicit NativeOpEquationContext(
      const std::vector<std::uint64_t>& in_tensors_ranks,
      const std::vector<std::uint64_t>& out_tensors_ranks)
      : in_tensors_ranks_(in_tensors_ranks),
        out_tensors_ranks_(out_tensors_ranks),
        equations_{},
        in_msg_box_in_indexes_(MakeArgIndexes(in_tensors_ranks.size())),
        in_msg_box_out_indexes_(MakeArgIndexes(out_tensors_ranks.size())),
        out_msg_box_in_indexes_(MakeArgIndexes(in_tensors_ranks.size())),
        out_msg_box_out_indexes_(MakeArgIndexes(out_tensors_ranks.size())) {
    Init(&in_iterator_tuples_, in_tensors_ranks);
    Init(&out_iterator_tuples_, out_tensors_ranks);
    Init(&in_stride_tuples_, in_tensors_ranks);
    Init(&out_stride_tuples_, out_tensors_ranks);
    Init(&in_dim_tuples_, in_tensors_ranks);
    Init(&out_dim_tuples_, out_tensors_ranks);
    GenerateDots();
    fake_op_placeholder_ = GenerateFakeOpPlaceholder();
  }

  ~NativeOpEquationContext() = default;

  const std::vector<std::uint64_t>& GetInTensorsRanks() const override {
    return in_tensors_ranks_;
  }

  const std::vector<std::uint64_t>& GetOutTensorsRanks() const override {
    return out_tensors_ranks_;
  }

  void Equal(const Iterator& lhs, const Iterator& rhs) override {
    this->Equal<Iterator>(lhs, rhs);
  }

  void Equal(const Index& lhs, const Index& rhs) override {
    this->Equal<Index>(lhs, rhs);
  }

  void Equal(const IteratorTuple& lhs, const IteratorTuple& rhs) override {
    CHECK(lhs->size() == rhs->size());
    for (std::size_t i = 0; i < lhs->size(); ++i) {
      this->Equal(lhs->at(i), rhs->at(i));
    }
  }

  const IteratorTuple& GetInIteratorTuple(
      std::size_t input_idx) const override {
    return in_iterator_tuples_.at(input_idx);
  }

  const IteratorTuple& GetOutIteratorTuple(
      std::size_t output_idx) const override {
    return out_iterator_tuples_.at(output_idx);
  }

  const Index& GetInIndex(std::size_t input_idx) const override {
    return in_msg_box_in_indexes_.value()->at(input_idx);
  }

  const Index& GetOutIndex(std::size_t output_idx) const override {
    return in_msg_box_out_indexes_.value()->at(output_idx);
  }

  const StrideTuple& GetInStrideTuple(std::size_t input_idx) const override {
    return in_stride_tuples_.at(input_idx);
  }

  const StrideTuple& GetOutStrideTuple(std::size_t output_idx) const override {
    return out_stride_tuples_.at(output_idx);
  }

  const DimTuple& GetInDimTuple(std::size_t input_idx) const override {
    return in_dim_tuples_.at(input_idx);
  }

  const DimTuple& GetOutDimTuple(std::size_t output_idx) const override {
    return out_dim_tuples_.at(output_idx);
  }

  const Equations& equations() const { return equations_; }

  const tInMsgBox<List<Index>>& in_msg_box_in_indexes() const {
    return in_msg_box_in_indexes_;
  }
  const tInMsgBox<List<Index>>& in_msg_box_out_indexes() const {
    return in_msg_box_out_indexes_;
  }
  const tOutMsgBox<List<Index>>& out_msg_box_in_indexes() const {
    return out_msg_box_in_indexes_;
  }
  const tOutMsgBox<List<Index>>& out_msg_box_out_indexes() const {
    return out_msg_box_out_indexes_;
  }

  const FakeOpPlaceHolder& fake_op_placeholder() const {
    return fake_op_placeholder_;
  }

  template <typename DoEachT>
  void VisitEachTensorIndex(const DoEachT& DoEach) const {
    VisitEachInputTensorIndex(DoEach);
    VisitEachOutputTensorIndex(DoEach);
  }

  template <typename DoEachT>
  void VisitEachInputTensorIndex(const DoEachT& DoEach) const {
    for (const auto& in_index : *in_msg_box_in_indexes_.value()) {
      DoEach(in_index);
    }
  }

  template <typename DoEachT>
  void VisitEachOutputTensorIndex(const DoEachT& DoEach) const {
    for (const auto& out_index : *in_msg_box_out_indexes_.value()) {
      DoEach(out_index);
    }
  }

  template <typename DoEachT>
  void VisitEachEquation(const DoEachT& DoEach) const {
    for (const auto& equation : *equations_) {
      DoEach(equation);
    }
  }

  std::optional<equation::Index> OutMsgBoxIndex4InMsgBoxIndex(
      const equation::Index& index) const {
    std::optional<Index> ret = OutMsgBoxInIndex4InMsgBoxInIndex(index);
    if (ret.has_value()) {
      return ret.value();
    }
    return OutMsgBoxOutIndex4InMsgBoxOutIndex(index);
  }

  std::optional<equation::Index> OutMsgBoxInIndex4InMsgBoxInIndex(
      const equation::Index& index) const {
    std::optional<std::size_t> pos =
        FindPos(in_msg_box_in_indexes_.value(), index);
    if (!pos.has_value()) {
      return std::nullopt;
    }
    CHECK_LT(pos.value(), out_msg_box_in_indexes().value()->size());
    return out_msg_box_in_indexes().value()->at(pos.value());
  }

  std::optional<equation::Index> OutMsgBoxOutIndex4InMsgBoxOutIndex(
      const equation::Index& index) const {
    std::optional<std::size_t> pos =
        FindPos(in_msg_box_out_indexes_.value(), index);
    if (!pos.has_value()) {
      return std::nullopt;
    }
    CHECK_LT(pos.value(), out_msg_box_out_indexes().value()->size());
    return out_msg_box_out_indexes().value()->at(pos.value());
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
    CHECK(iterator_tuple->size() == stride_tuple->size());
    Index index{UniqueId::New()};
    equations_->emplace_back(
        equation::Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>(
            stride_tuple, index, iterator_tuple));
    equations_->emplace_back(
        equation::UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>(
            stride_tuple, iterator_tuple, index));
    return index;
  }

  static List<Index> MakeArgIndexes(std::size_t num_args) {
    List<Index> ret{};
    for (std::size_t i = 0; i < num_args; ++i) {
      Index index{UniqueId::New()};
      ret->emplace_back(index);
    }
    return ret;
  }

  void GenerateDots() {
    for (std::size_t i = 0; i < in_tensors_ranks_.size(); ++i) {
      Equal(GetInIndex(i), Dot(GetInIteratorTuple(i), GetInStrideTuple(i)));
    }
    for (std::size_t i = 0; i < out_tensors_ranks_.size(); ++i) {
      Equal(GetOutIndex(i), Dot(GetOutIteratorTuple(i), GetOutStrideTuple(i)));
    }
  }

  template <typename T>
  void Equal(const T& lhs, const T& rhs) {
    equations_->emplace_back(Identity<tOut<T>, tIn<T>>(lhs, rhs));
    equations_->emplace_back(Identity<tOut<T>, tIn<T>>(rhs, lhs));
  }

  FakeOpPlaceHolder GenerateFakeOpPlaceholder() const {
    FakeOpPlaceHolder fake_op_placeholder{UniqueId::New()};

    equations_->emplace_back(InMsgBox2OutMsgBox<tOut<tOutMsgBox<OpArgIndexes>>,
                                                tIn<tInMsgBox<OpArgIndexes>>>{
        fake_op_placeholder,
        MakeOutMsgBoxOpArgIndexes(),
        MakeInMsgBoxOpArgIndexes()});

    return fake_op_placeholder;
  }

  tOutMsgBox<OpArgIndexes> MakeOutMsgBoxOpArgIndexes() const {
    return OpArgIndexes{out_msg_box_in_indexes_.value(),
                        out_msg_box_out_indexes_.value()};
  }

  tInMsgBox<OpArgIndexes> MakeInMsgBoxOpArgIndexes() const {
    return OpArgIndexes{in_msg_box_in_indexes_.value(),
                        in_msg_box_out_indexes_.value()};
  }

  static std::optional<std::size_t> FindPos(const List<Index>& vector,
                                            const Index& index) {
    for (std::size_t i = 0; i < vector->size(); ++i) {
      if (vector->at(i) == index) {
        return i;
      }
    }
    return std::nullopt;
  }

  std::vector<std::uint64_t> in_tensors_ranks_;
  std::vector<std::uint64_t> out_tensors_ranks_;
  Equations equations_;
  tInMsgBox<List<Index>> in_msg_box_in_indexes_;
  tInMsgBox<List<Index>> in_msg_box_out_indexes_;
  tOutMsgBox<List<Index>> out_msg_box_in_indexes_;
  tOutMsgBox<List<Index>> out_msg_box_out_indexes_;
  std::vector<IteratorTuple> in_iterator_tuples_;
  std::vector<IteratorTuple> out_iterator_tuples_;
  std::vector<StrideTuple> in_stride_tuples_;
  std::vector<StrideTuple> out_stride_tuples_;
  std::vector<DimTuple> in_dim_tuples_;
  std::vector<DimTuple> out_dim_tuples_;

  FakeOpPlaceHolder fake_op_placeholder_;
};

}  // namespace cinn::adt::equation::config

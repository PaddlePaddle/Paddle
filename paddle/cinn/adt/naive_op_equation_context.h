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
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/op_arg_pos.h"
#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::config {

class NaiveOpEquationContext;

class NaiveConditionalEqualHandler final : public ConditionalEqualHandler {
 public:
  NaiveConditionalEqualHandler(const NaiveConditionalEqualHandler&) = delete;
  NaiveConditionalEqualHandler(NaiveConditionalEqualHandler&&) = delete;

  NaiveConditionalEqualHandler(NaiveOpEquationContext* ctx,
                               const Equations& equations)
      : ctx_(ctx), equations_(equations) {}

  void Where(const EquationStaticLogical& logical) const override;

 private:
  NaiveOpEquationContext* ctx_;
  Equations equations_;
};

class NaiveOpEquationContext final : public OpEquationContext {
 public:
  NaiveOpEquationContext(const NaiveOpEquationContext&) = delete;
  NaiveOpEquationContext(NaiveOpEquationContext&&) = delete;

  using GetArgStaticDimT = std::function<std::optional<std::int64_t>(
      std::size_t tensor_idx, std::size_t dim_idx)>;

  explicit NaiveOpEquationContext(
      const std::vector<std::uint64_t>& in_tensors_ranks,
      const std::vector<std::uint64_t>& out_tensors_ranks,
      GetArgStaticDimT GetInDim,
      GetArgStaticDimT GetOutDim)
      : in_tensors_ranks_(in_tensors_ranks),
        out_tensors_ranks_(out_tensors_ranks),
        GetInDim_(GetInDim),
        GetOutDim_(GetOutDim),
        equations_{},
        in_msg_box_in_indexes_(MakeArgIndexes(in_tensors_ranks.size())),
        in_msg_box_out_indexes_(MakeArgIndexes(out_tensors_ranks.size())),
        out_msg_box_in_indexes_(MakeArgIndexes(in_tensors_ranks.size())),
        out_msg_box_out_indexes_(MakeArgIndexes(out_tensors_ranks.size())) {
    Init<Iterator>(&in_iterator_tuples_, in_tensors_ranks);
    Init<Iterator>(&out_iterator_tuples_, out_tensors_ranks);
    Init<Stride>(&in_stride_tuples_, in_tensors_ranks);
    Init<Stride>(&out_stride_tuples_, out_tensors_ranks);
    Init<Dim>(&in_dim_tuples_, in_tensors_ranks);
    Init<Dim>(&out_dim_tuples_, out_tensors_ranks);
    GenerateDots();
    fake_op_placeholder_ = GenerateFakeOpPlaceholder();
  }

  ~NaiveOpEquationContext() = default;

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

  std::unique_ptr<ConditionalEqualHandler> ConditionalEqual(
      const Iterator& lhs, const Iterator& rhs) override {
    Equations equations{};
    return ConditionalEqual(lhs, rhs, &equations);
  }

  std::unique_ptr<ConditionalEqualHandler> ConditionalEqual(
      const Iterator& iterator, std::size_t constant) override {
    Equations equations{};
    Iterator const_iter = MakeConstantIterator(constant, &equations);
    return ConditionalEqual(iterator, const_iter, &equations);
  }

  Iterator MakeConstantIterator(std::size_t constant,
                                Equations* equations) const {
    using ConstF = ConstantFunction<tOut<Iterator>, tIn<Index>>;
    Iterator const_iter{UniqueId::New()};
    VisitEachTensorIndex([&](const auto& in_msg_box_index) {
      (*equations)
          ->emplace_back(ConstF{const_iter, in_msg_box_index, constant});
    });
    return const_iter;
  }

  std::unique_ptr<ConditionalEqualHandler> ConditionalEqual(
      const Iterator& lhs, const Iterator& rhs, Equations* equations) {
    (*equations)
        ->emplace_back(Identity<tOut<Iterator>, tIn<Iterator>>(lhs, rhs));
    (*equations)
        ->emplace_back(Identity<tOut<Iterator>, tIn<Iterator>>(rhs, lhs));
    return std::make_unique<NaiveConditionalEqualHandler>(this, *equations);
  }

  std::unique_ptr<ConditionalEqualHandler> ConditionalEqual(
      const Index& lhs, const Index& rhs) override {
    Equations equations{};
    equations->emplace_back(Identity<tOut<Index>, tIn<Index>>(lhs, rhs));
    equations->emplace_back(Identity<tOut<Index>, tIn<Index>>(rhs, lhs));
    return std::make_unique<NaiveConditionalEqualHandler>(this, equations);
  }

  EquationStaticLogical EQ(const Dim& lhs, const Dim& rhs) const override {
    return EquationStaticLogical{
        cinn::adt::EQ<EquationStaticValue, EquationStaticValue>(
            GetDimSize(lhs), GetDimSize(rhs))};
  }

  EquationStaticLogical NE(const Dim& lhs, const Dim& rhs) const override {
    return EquationStaticLogical{
        cinn::adt::NE<EquationStaticValue, EquationStaticValue>(
            GetDimSize(lhs), GetDimSize(rhs))};
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

  void AddEquations(const Equations& equations) {
    for (const auto& equation : *equations) {
      equations_->emplace_back(equation);
    }
  }

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

  std::optional<Index> OutMsgBoxIndex4InMsgBoxIndex(const Index& index) const {
    std::optional<Index> ret = OutMsgBoxInIndex4InMsgBoxInIndex(index);
    if (ret.has_value()) {
      return ret.value();
    }
    return OutMsgBoxOutIndex4InMsgBoxOutIndex(index);
  }

  std::optional<Index> OutMsgBoxInIndex4InMsgBoxInIndex(
      const Index& index) const {
    std::optional<std::size_t> pos =
        FindPos(in_msg_box_in_indexes_.value(), index);
    if (!pos.has_value()) {
      return std::nullopt;
    }
    CHECK_LT(pos.value(), out_msg_box_in_indexes().value()->size());
    return out_msg_box_in_indexes().value()->at(pos.value());
  }

  std::optional<Index> OutMsgBoxOutIndex4InMsgBoxOutIndex(
      const Index& index) const {
    std::optional<std::size_t> pos =
        FindPos(in_msg_box_out_indexes_.value(), index);
    if (!pos.has_value()) {
      return std::nullopt;
    }
    CHECK_LT(pos.value(), out_msg_box_out_indexes().value()->size());
    return out_msg_box_out_indexes().value()->at(pos.value());
  }

  void EraseOutMsgBoxIndexes(
      const std::vector<Index>& truncated_output_tensor_indexes);

  OpArgPos GetOpArgPos(const Index& index) const {
    const auto& input_pos = FindPos(in_msg_box_in_indexes_.value(), index);
    if (input_pos.has_value()) {
      return tIn<std::size_t>{input_pos.value()};
    }
    const auto& output_pos = FindPos(in_msg_box_out_indexes_.value(), index);
    if (output_pos.has_value()) {
      return tOut<std::size_t>{output_pos.value()};
    }
    return Undefined{};
  }

  std::int64_t GetDimSize(const Dim& dim) const;

  OpArgDimPos GetArgDimPosDescriptor(const Dim& dim) const {
    const auto& input_pos = FindArgDimPos(in_dim_tuples_, dim);
    if (input_pos.has_value()) {
      return tIn<ArgDimPosDescriptor>{input_pos.value()};
    }
    const auto& output_pos = FindArgDimPos(out_dim_tuples_, dim);
    if (output_pos.has_value()) {
      return tOut<ArgDimPosDescriptor>{output_pos.value()};
    }
    return Undefined{};
  }

 private:
  template <typename value_type, typename ContainerT>
  void Init(ContainerT* vec, const std::vector<std::uint64_t>& tensors_ranks) {
    for (std::size_t i = 0; i < tensors_ranks.size(); ++i) {
      vec->push_back(typename ContainerT::value_type{});
      for (std::size_t j = 0; j < tensors_ranks.at(i); ++j) {
        vec->at(i)->emplace_back(value_type{UniqueId::New()});
      }
    }
  }

  Index Dot(const IteratorTuple& iterator_tuple,
            const StrideTuple& stride_tuple) {
    CHECK(iterator_tuple->size() == stride_tuple->size());
    Index index{UniqueId::New()};
    equations_->emplace_back(
        adt::Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>{
            stride_tuple, index, iterator_tuple});
    equations_->emplace_back(
        adt::UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>{
            stride_tuple, iterator_tuple, index});
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

    equations_->emplace_back(InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                                                tOut<tOutMsgBox<OpArgIndexes>>,
                                                tIn<tInMsgBox<OpArgIndexes>>>{
        fake_op_placeholder,
        MakeOutMsgBoxOpArgIndexes(),
        MakeInMsgBoxOpArgIndexes()});

    return fake_op_placeholder;
  }

  tOutMsgBox<OpArgIndexes> MakeOutMsgBoxOpArgIndexes() const {
    return tOutMsgBox<OpArgIndexes>{OpArgIndexes{
        out_msg_box_in_indexes_.value(), out_msg_box_out_indexes_.value()}};
  }

  tInMsgBox<OpArgIndexes> MakeInMsgBoxOpArgIndexes() const {
    return tInMsgBox<OpArgIndexes>{OpArgIndexes{
        in_msg_box_in_indexes_.value(), in_msg_box_out_indexes_.value()}};
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

  static std::optional<ArgDimPosDescriptor> FindArgDimPos(
      const std::vector<DimTuple>& dim_tuples, const Dim& dim) {
    for (std::size_t i = 0; i < dim_tuples.size(); ++i) {
      for (std::size_t j = 0; j < dim_tuples.at(i)->size(); ++j) {
        if (dim_tuples.at(i)->at(j) == dim) {
          return ArgDimPosDescriptor{i, j};
        }
      }
    }
    return std::nullopt;
  }

  std::vector<std::uint64_t> in_tensors_ranks_;
  std::vector<std::uint64_t> out_tensors_ranks_;
  GetArgStaticDimT GetInDim_;
  GetArgStaticDimT GetOutDim_;
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

std::function<std::shared_ptr<config::NaiveOpEquationContext>(const OpStmt&)>
GenerateContext4LocalOpStmt(const List<OpStmt>& op_stmts);

}  // namespace cinn::adt::config

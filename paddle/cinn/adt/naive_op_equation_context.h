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
#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/adt/op_arg_pos.h"
#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/common/enforce.h"

namespace cinn::adt::config {

class NaiveOpEquationContext final : public OpEquationContext {
 public:
  NaiveOpEquationContext(const NaiveOpEquationContext&) = delete;
  NaiveOpEquationContext(NaiveOpEquationContext&&) = delete;

  using GetArgStaticDimT = std::function<std::optional<std::int64_t>(
      std::size_t tensor_idx, std::size_t dim_idx)>;
  using GetArgSymbolicDimT = std::function<std::optional<DimExpr>(
      std::size_t tensor_idx, std::size_t dim_idx)>;

  explicit NaiveOpEquationContext(
      const std::vector<std::uint64_t>& in_tensors_ranks,
      const std::vector<std::uint64_t>& out_tensors_ranks,
      GetArgStaticDimT GetInDim,
      GetArgStaticDimT GetOutDim,
      GetArgSymbolicDimT GetSymbolicInDim,
      GetArgSymbolicDimT GetSymbolicOutDim,
      cinn::utils::AttributeMap attr_map_type)
      : in_tensors_ranks_(in_tensors_ranks),
        out_tensors_ranks_(out_tensors_ranks),
        GetInDim_(GetInDim),
        GetOutDim_(GetOutDim),
        GetSymbolicInDim_(GetSymbolicInDim),
        GetSymbolicOutDim_(GetSymbolicOutDim),
        equations_{},
        attr_map_type_(attr_map_type),
        fake_op_placeholder_{UniqueId::New()} {
    Init<Iterator>(&in_iterator_tuples_, in_tensors_ranks);
    Init<Iterator>(&out_iterator_tuples_, out_tensors_ranks);
    InitInputDimExpr(&in_dim_tuples_, in_tensors_ranks);
    InitOutputDimExpr(&out_dim_tuples_, out_tensors_ranks);
    in_indexes_ = MakeArgIndexes(in_tensors_ranks.size());
    out_indexes_ = MakeArgIndexes(out_tensors_ranks.size());
    GenerateDots();
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
    PADDLE_ENFORCE_EQ(lhs->size(),
                      rhs->size(),
                      ::common::errors::InvalidArgument(
                          "The sizes of lhs and rhs must be equal. "
                          "lhs size: %d, rhs size: %d",
                          lhs->size(),
                          rhs->size()));
    for (std::size_t i = 0; i < lhs->size(); ++i) {
      this->Equal(lhs->at(i), rhs->at(i));
    }
  }

  Iterator GetBroadcastedInputIterator(const Iterator& out_tensor_iterator,
                                       const DimExpr& dim) override {
    Iterator input_tensor_iterator{UniqueId::New()};
    using Function =
        GetBroadcastedIterator<DimExpr, tOut<Iterator>, tIn<Iterator>>;
    equations_->emplace_back(
        Function{dim, input_tensor_iterator, out_tensor_iterator});
    return input_tensor_iterator;
  }

  Iterator GetConstantIterator(const Index& in_index, int constant) override {
    Iterator output_tensor_iterator{UniqueId::New()};
    using ConstF = ConstantFunction<tOut<Iterator>, tIn<Index>>;
    equations_->emplace_back(
        ConstF{output_tensor_iterator, in_index, DimExpr{constant}});
    return output_tensor_iterator;
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
    return in_indexes_->at(input_idx);
  }

  const Index& GetOutIndex(std::size_t output_idx) const override {
    return out_indexes_->at(output_idx);
  }

  const DimTuple& GetInDimTuple(std::size_t input_idx) const override {
    return in_dim_tuples_.at(input_idx);
  }

  const DimTuple& GetOutDimTuple(std::size_t output_idx) const override {
    return out_dim_tuples_.at(output_idx);
  }

  const List<Index>& in_indexes() const { return in_indexes_; }

  const List<Index>& out_indexes() const { return out_indexes_; }

  const Equations& equations() const { return equations_; }

  void AddEquations(const Equations& equations) {
    for (const auto& equation : *equations) {
      equations_->emplace_back(equation);
    }
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
    for (const auto& in_index : *in_indexes_) {
      DoEach(in_index);
    }
  }

  template <typename DoEachT>
  void VisitEachOutputTensorIndex(const DoEachT& DoEach) const {
    for (const auto& out_index : *out_indexes_) {
      DoEach(out_index);
    }
  }

  template <typename DoEachT>
  void VisitEachEquation(const DoEachT& DoEach) const {
    for (const auto& equation : *equations_) {
      DoEach(equation);
    }
  }

  template <typename DoEachT>
  void VisitEachArgPos(const DoEachT& DoEach) const {
    for (std::size_t arg_idx = 0; arg_idx < in_tensors_ranks_.size();
         ++arg_idx) {
      for (std::size_t axis = 0; axis < in_tensors_ranks_.at(arg_idx); ++axis) {
        DoEach(/*is_out*/ false, arg_idx, axis);
      }
    }
    for (std::size_t arg_idx = 0; arg_idx < out_tensors_ranks_.size();
         ++arg_idx) {
      for (std::size_t axis = 0; axis < out_tensors_ranks_.at(arg_idx);
           ++axis) {
        DoEach(/*is_out*/ true, arg_idx, axis);
      }
    }
  }

  OpArgPos GetOpArgPos(const Index& index) const {
    const auto& input_pos = FindPos(in_indexes_, index);
    if (input_pos.has_value()) {
      return tIn<std::size_t>{input_pos.value()};
    }
    const auto& output_pos = FindPos(out_indexes_, index);
    if (output_pos.has_value()) {
      return tOut<std::size_t>{output_pos.value()};
    }
    return Undefined{};
  }

  DimExpr GetDim(bool is_out, std::size_t arg_idx, std::size_t axis) const {
    if (is_out) {
      return out_dim_tuples_.at(arg_idx)->at(axis);
    } else {
      return in_dim_tuples_.at(arg_idx)->at(axis);
    }
  }

  std::optional<std::int64_t> GetStaticDimSize(bool is_out,
                                               std::size_t arg_idx,
                                               std::size_t axis) const {
    const auto* Get = (is_out ? &GetOutDim_ : &GetInDim_);
    const auto& opt_dim = (*Get)(arg_idx, axis);
    return opt_dim;
  }

  std::optional<DimExpr> GetSymbolicDimSize(bool is_out,
                                            std::size_t arg_idx,
                                            std::size_t axis) const {
    const auto* Get = (is_out ? &GetSymbolicOutDim_ : &GetSymbolicInDim_);
    const auto& opt_dim = (*Get)(arg_idx, axis);
    return opt_dim;
  }

  void Print() const;

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

  void InitInputDimExpr(std::vector<DimTuple>* vec,
                        const std::vector<std::uint64_t>& tensors_ranks) {
    for (std::size_t i = 0; i < tensors_ranks.size(); ++i) {
      vec->push_back(DimTuple{});
      for (std::size_t j = 0; j < tensors_ranks.at(i); ++j) {
        const auto& opt_expr = GetSymbolicInDim_(i, j);
        PADDLE_ENFORCE_EQ(opt_expr.has_value(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The optional expression must have a value."));
        vec->at(i)->emplace_back(opt_expr.value());
      }
    }
  }

  void InitOutputDimExpr(std::vector<DimTuple>* vec,
                         const std::vector<std::uint64_t>& tensors_ranks) {
    for (std::size_t i = 0; i < tensors_ranks.size(); ++i) {
      vec->push_back(DimTuple{});
      for (std::size_t j = 0; j < tensors_ranks.at(i); ++j) {
        const auto& opt_expr = GetSymbolicOutDim_(i, j);
        PADDLE_ENFORCE_EQ(opt_expr.has_value(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The optional expression must have a value at "
                              "tensor index %d and dimension index %d.",
                              i,
                              j));
        vec->at(i)->emplace_back(opt_expr.value());
      }
    }
  }

  Index IndexDot(const IteratorTuple& iterator_tuple,
                 const DimTuple& dim_tuple) {
    PADDLE_ENFORCE_EQ(iterator_tuple->size(),
                      dim_tuple->size(),
                      ::common::errors::InvalidArgument(
                          "The sizes of iterator_tuple and dim_tuple must be "
                          "equal. iterator_tuple size: %d, dim_tuple size: %d",
                          iterator_tuple->size(),
                          dim_tuple->size()));
    Index index{UniqueId::New()};
    equations_->emplace_back(
        adt::IndexDot<List<DimExpr>, tOut<Index>, tIn<List<Iterator>>>{
            dim_tuple, index, iterator_tuple});
    equations_->emplace_back(
        adt::IndexUnDot<List<DimExpr>, tOut<List<Iterator>>, tIn<Index>>{
            dim_tuple, iterator_tuple, index});
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
      Equal(GetInIndex(i), IndexDot(GetInIteratorTuple(i), GetInDimTuple(i)));
    }
    for (std::size_t i = 0; i < out_tensors_ranks_.size(); ++i) {
      Equal(GetOutIndex(i),
            IndexDot(GetOutIteratorTuple(i), GetOutDimTuple(i)));
    }
  }

  template <typename T>
  void Equal(const T& lhs, const T& rhs) {
    equations_->emplace_back(Identity<tOut<T>, tIn<T>>(lhs, rhs));
    equations_->emplace_back(Identity<tOut<T>, tIn<T>>(rhs, lhs));
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

  const utils::Attribute& GetAttribute(const std::string& name) const {
    const auto& iter = attr_map_type_.find(name);
    PADDLE_ENFORCE_EQ(
        iter != attr_map_type_.end(),
        true,
        ::common::errors::InvalidArgument(
            "Can't find Attribute with this name: %s", name.c_str()));
    return iter->second;
  }

  std::vector<std::uint64_t> in_tensors_ranks_;
  std::vector<std::uint64_t> out_tensors_ranks_;
  GetArgStaticDimT GetInDim_;
  GetArgStaticDimT GetOutDim_;
  GetArgSymbolicDimT GetSymbolicInDim_;
  GetArgSymbolicDimT GetSymbolicOutDim_;
  Equations equations_;
  const cinn::utils::AttributeMap attr_map_type_;
  FakeOpPlaceHolder fake_op_placeholder_;

  std::vector<IteratorTuple> in_iterator_tuples_;
  std::vector<IteratorTuple> out_iterator_tuples_;
  std::vector<DimTuple> in_dim_tuples_;
  std::vector<DimTuple> out_dim_tuples_;
  List<Index> in_indexes_;
  List<Index> out_indexes_;
};

std::function<std::shared_ptr<config::NaiveOpEquationContext>(const OpStmt&)>
GenerateContext4LocalOpStmt(const List<OpStmt>& op_stmts);

}  // namespace cinn::adt::config

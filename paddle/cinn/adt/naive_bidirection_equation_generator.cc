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

#include "paddle/cinn/adt/naive_bidirection_equation_generator.h"

#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/common/enforce.h"
namespace cinn::adt {

namespace {

using EquationCtx4OpStmtT =
    std::function<std::shared_ptr<config::NaiveOpEquationContext>(
        const OpStmt&)>;

template <
    typename DoEachT /*: void(&)(std::size_t, OpStmt, OpEquationContext)*/>
void VisitEachOpStmtAndEquationCtx(
    const List<OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op_stmts->size(); ++i) {
    const auto& ctx = EquationCtx4OpStmt(op_stmts->at(i));
    DoEach(i, op_stmts->at(i), ctx);
  }
}

List<Index> MakeArgIndexes(std::size_t num_args) {
  List<Index> ret{};
  for (std::size_t i = 0; i < num_args; ++i) {
    Index index{UniqueId::New()};
    ret->emplace_back(index);
  }
  return ret;
}

OpArgIndexes<std::optional<Index>> MakeOutMsgOpArgIndexes(
    const List<std::optional<Index>>& opt_out_msg_in_indexes,
    const List<std::optional<Index>>& opt_out_msg_out_indexes) {
  List<Index> out_msg_in_indexes{};
  for (const auto& out_msg_in_index : *opt_out_msg_in_indexes) {
    PADDLE_ENFORCE_EQ(out_msg_in_index.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The out_msg_in_index should have value."));
    out_msg_in_indexes->emplace_back(out_msg_in_index.value());
  }
  return OpArgIndexes<std::optional<Index>>{out_msg_in_indexes,
                                            opt_out_msg_out_indexes};
}

OpArgIndexes<Index> MakeInMsgOpArgIndexes(
    const List<Index>& in_msg_in_indexes,
    const List<Index>& in_msg_out_indexes) {
  return OpArgIndexes<Index>{in_msg_in_indexes, in_msg_out_indexes};
}

template <typename DoEachT>
void VisitEachInMsgOutMsgPair(const List<Index>& in_msg_indexes,
                              const List<Index>& out_msg_indexes,
                              const DoEachT& DoEach) {
  PADDLE_ENFORCE_EQ(
      in_msg_indexes->size(),
      out_msg_indexes->size(),
      ::common::errors::InvalidArgument(
          "The size of in_msg_indexes and out_msg_indexes should be equal, but "
          "got in_msg_indexes size = %d, out_msg_indexes size = %d.",
          in_msg_indexes->size(),
          out_msg_indexes->size()));
  for (std::size_t i = 0; i < in_msg_indexes->size(); ++i) {
    DoEach(in_msg_indexes->at(i), out_msg_indexes->at(i));
  }
}

List<std::optional<Index>> GetOutMsgIndexes(
    const List<Index>& in_indexes,
    const NaiveBidirectionEquationGenerator& generator) {
  List<std::optional<Index>> ret{};
  for (const auto& index : *in_indexes) {
    ret->emplace_back(generator.OutMsgIndex4InMsgIndex(index));
  }
  return ret;
}

using InMsg2OutMsgT = InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                                   tOut<OpArgIndexes<std::optional<Index>>>,
                                   tIn<OpArgIndexes<Index>>>;

}  // namespace

void NaiveBidirectionEquationGenerator::InitInMsgIndex2OutMsgIndex() {
  const auto& InitEachOpInMsgIndex2OutMsgIndex =
      [&](const std::shared_ptr<config::NaiveOpEquationContext>& ctx,
          bool is_output) {
        List<Index> in_msg_indexes =
            is_output ? ctx->out_indexes() : ctx->in_indexes();
        std::size_t out_msg_index_size = is_output
                                             ? ctx->GetOutTensorsRanks().size()
                                             : ctx->GetInTensorsRanks().size();
        List<Index> out_msg_indexes = MakeArgIndexes(out_msg_index_size);
        VisitEachInMsgOutMsgPair(
            in_msg_indexes,
            out_msg_indexes,
            [&](const Index& in_index, const Index& out_index) {
              PADDLE_ENFORCE_EQ(
                  this->in_msg_index2out_msg_index_.emplace(in_index, out_index)
                      .second,
                  true,
                  ::common::errors::InvalidArgument(
                      "The out_msg_index2in_msg_index_ map has already "
                      "contained the out_index."));
            });
      };

  VisitEachOpStmtAndEquationCtx(
      this->op_stmts_,
      this->EquationCtx4OpStmt_,
      [&](std::size_t idx,
          const OpStmt& op_stmt,
          const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        InitEachOpInMsgIndex2OutMsgIndex(ctx, /*is_output=*/false);
        InitEachOpInMsgIndex2OutMsgIndex(ctx, /*is_output=*/true);
      });
}

void NaiveBidirectionEquationGenerator::InitEquations() {
  VisitEachOpStmtAndEquationCtx(
      this->op_stmts_,
      this->EquationCtx4OpStmt_,
      [&](std::size_t idx,
          const OpStmt& op_stmt,
          const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        List<Index> in_msg_in_indexes = ctx->in_indexes();
        List<Index> in_msg_out_indexes = ctx->out_indexes();
        List<std::optional<Index>> out_msg_in_indexes =
            GetOutMsgIndexes(in_msg_in_indexes, *this);
        List<std::optional<Index>> out_msg_out_indexes =
            GetOutMsgIndexes(in_msg_out_indexes, *this);

        Equation equation = InMsg2OutMsgT{
            ctx->fake_op_placeholder(),
            MakeOutMsgOpArgIndexes(out_msg_in_indexes, out_msg_out_indexes),
            MakeInMsgOpArgIndexes(in_msg_in_indexes, in_msg_out_indexes)};

        this->fake_op_placeholders_->emplace_back(ctx->fake_op_placeholder());
        this->equations_->emplace_back(equation);
      });
}

std::function<const OpStmt*(const FakeOpPlaceHolder&)>
NaiveBidirectionEquationGenerator::MakeGetterOpStmt4OpPlaceHolder() const {
  using FakeOpPlaceHolder2OpStmt =
      std::unordered_map<FakeOpPlaceHolder, OpStmt>;
  const auto& fake_op_placeholder2op_stmt =
      std::make_shared<FakeOpPlaceHolder2OpStmt>();

  for (std::size_t i = 0; i < fake_op_placeholders_->size(); ++i) {
    PADDLE_ENFORCE_EQ(
        fake_op_placeholder2op_stmt
            ->emplace(fake_op_placeholders_->at(i), op_stmts_->at(i))
            .second,
        true,
        ::common::errors::InvalidArgument(
            "The fake_op_placeholder2op_stmt map has already contained the "
            "fake_op_placeholder."));
  }

  return [fake_op_placeholder2op_stmt](
             const FakeOpPlaceHolder& fake_op_placeholder) {
    return &fake_op_placeholder2op_stmt->at(fake_op_placeholder);
  };
}

}  // namespace cinn::adt

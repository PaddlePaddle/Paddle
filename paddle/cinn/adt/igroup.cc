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

#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"

namespace cinn::adt {

namespace {

std::shared_ptr<IndexExprInferContext> MakeIndexExprInferContext(
    const IGroup& igroup) {
  std::unordered_map<Variable, const Value> anchor_iterator2value{};

  const auto& anchor_iterators = igroup.GetAnchorIterators();

  for (std::size_t i = 0; i < anchor_iterators->size(); ++i) {
    CHECK(anchor_iterator2value
              .emplace(anchor_iterators->at(i), anchor_iterators->at(i))
              .second);
  }

  return std::make_shared<IndexExprInferContext>(anchor_iterator2value,
                                                 igroup.constants_provider());
}

std::function<Value(const Iterator&)> MakeGetterValue4Iterator(
    const IGroup* igroup) {
  GraphView igroup_view = igroup->GetDefaultGraphView();

  const auto& ctx = MakeIndexExprInferContext(*igroup);

  std::vector<Variable> starts{};
  for (const auto& anchor_iterator : *igroup->GetAnchorIterators()) {
    starts.emplace_back(anchor_iterator);
  }
  SolveEquations(igroup_view, starts, ctx.get());
  return [ctx](const Iterator& iterator) { return ctx->GetValue(iterator); };
}

List<LoopSize> MakeAnchorLoopSize(const Tensor& tensor) {
  List<LoopSize> ret{};
  CHECK(tensor.Has<adapter::Tensor>());
  for (const auto& dim : tensor.Get<adapter::Tensor>().GetShape()) {
    ret->emplace_back(dim);
  }
  return ret;
}

}  // namespace

void IGroup::InitAnchorScheduleDims() {
  const auto& Value4Iterator = MakeGetterValue4Iterator(this);

  const auto& loop_size = MakeAnchorLoopSize(this->anchor_tensor());

  anchor_schedule_dims_ = MakeAnchorScheduleDims(
      *this, Value4Iterator, loop_size, this->GetAnchorIterators());
}

List<Iterator> IGroup::GetIndexIterators(const Index& index) const {
  List<Iterator> ret{};
  for (const auto& op_stmt : *op_stmts_) {
    const auto& ctx = EquationCtx4OpStmt_(op_stmt);
    const OpArgPos& arg_pos = ctx->GetOpArgPos(index);
    if (arg_pos.Has<tIn<std::size_t>>()) {
      return ctx->GetInIteratorTuple(arg_pos.Get<tIn<std::size_t>>().value());
    } else if (arg_pos.Has<tOut<std::size_t>>()) {
      return ctx->GetOutIteratorTuple(arg_pos.Get<tOut<std::size_t>>().value());
    } else if (arg_pos.Has<Undefined>()) {
      // do nothing
    } else {
      LOG(FATAL) << "Dead code";
    }
  }
  LOG(FATAL) << "Can not find anchor iterators";
}

}  // namespace cinn::adt

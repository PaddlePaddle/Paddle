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

#include <memory>
#include <vector>

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/m_ir.h"

namespace cinn::adt {

using AnchorIndex = eqaution::Variable;

class IGroup final {
 public:
  IGroup(const IGroup&) = delete;
  IGroup(IGroup&&) = delete;

  explicit IGroup(const List<m_expr::OpStmt>& op_stmts, const AnchorIndex& anchor_index)
      : op_stmt_nodes_(op_stmts),
        anchor_index_(anchor_index) {
     ADT_TODO();
  }

  const List<m_expr::OpStmt>& op_stmts() const { return op_stmt_nodes_; }

  const AnchorIndex& anchor_index() const { return anchor_index_; }

  const m_expr::Tensor& anchor_tensor() const { return GetTensor(anchor_index()); }

  GraphView GetDefaultGraphView() const { ADT_TODO(); }

  const m_expr::Tensor& GetTensor(const Index& index) const {
    AD_TODO();
  }

 private:
  AnchorIndex anchor_index_;
  List<m_expr::OpStmt> op_stmts_;
  std::unordered_map<m_expr::OpStmt, equation::IndexExprInferContext> op_stmt2index_expr_infer_ctx_;
};

}  // namespace cinn::adt

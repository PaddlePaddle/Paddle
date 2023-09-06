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

using AnchorTensor = eqaution::Variable;

class IGroup final {
 public:
  IGroup(const IGroup&) = delete;
  IGroup(IGroup&&) = delete;

  explicit IGroup(const std::shared_ptr<m_ir::MapIRList>& map_irs,
                  const std::shared_ptr<AnchorTensor>& anchor_tensor,
                  const std::shared_ptr<equation::Graph>& equation_graph)
      : map_irs_(map_irs),
        anchor_tensor_(anchor_tensor),
        equation_graph_(equation_graph) {}

  const std::shared_ptr<m_ir::MapIRList>& map_irs() const { return map_irs_; }

  const std::shared_ptr<AnchorTensor>& anchor_tensor() const {
    return anchor_tensor_;
  }

  GraphView GetDefaultGraphView() const { ADT_TODO(); }

  cinn::hlir::framework::NodeData* GetTensor(const Index& index) const;

 private:
  std::shared_ptr<AnchorTensor> anchor_tensor_;
  std::shared_ptr<m_ir::MapIRList> map_irs_;
  std::shared_ptr<equation::Graph> equation_graph_;
  equation::IndexExprInferContext index_expr_infer_ctx_;
};

}  // namespace cinn::adt

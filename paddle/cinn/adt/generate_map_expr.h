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

#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/m_ir.h"
#include "paddle/cinn/hlir/framework/graph.h"

namespace cinn::adt {

class IGroup;
class KGroup;

MapExpr GenerateMapExpr(
    const std::shared_ptr<cinn::hlir::framework::Graph::Group>& group);

void TryGenerateMapExprFromGraph(
    const std::shared_ptr<cinn::hlir::framework::Graph>& graph);

}  // namespace cinn::adt

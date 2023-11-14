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

#include <memory>
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/ir.h"
namespace cinn {
namespace hlir {
namespace pass {
using shape_t = utils::ShapeType;
class LowerOpPassImpl {
 public:
  std::vector<ir::LoweredFunc> operator()(hlir::framework::Graph& graph,
                                          common::Target& target) {
    std::vector<ir::LoweredFunc> ret;
    for (auto& group : graph.fusion_groups) {
      auto& dtype_dict =
          graph.GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
              "inferdtype");
      auto& shape_dict =
          graph.GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>(
              "infershape");
      auto& dyn_shape_dict = graph.GetMutableAttrs<
          absl::flat_hash_map<std::string, std::vector<std::string>>>(
          "inferdynshape");
      auto op_lowerer = framework::CreateOpLowerer(
          dtype_dict, shape_dict, target, dyn_shape_dict);
      auto lowered_funcs = op_lowerer.Lower(group, false, false, false);
      ret.insert(ret.end(), lowered_funcs.begin(), lowered_funcs.end());
    }
    return ret;
  }
  std::vector<ir::LoweredFunc> operator()(frontend::Program& program,
                                          common::Target& target) {
    auto graph = framework::Graph(program, {}, target);
    return operator()(graph, target);
  }
};
}  // namespace pass
}  // namespace hlir
}  // namespace cinn

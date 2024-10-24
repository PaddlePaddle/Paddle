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

#include "paddle/cinn/optim/buffer_assign.h"

#include "paddle/cinn/common/union_find.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/lang/lower_impl.h"

namespace cinn {
namespace optim {

namespace {

struct BufferUFNode : public cinn::common::UnionFindNode {
  explicit BufferUFNode(const std::string& x) : tensor_name(x) {}

  const char* type_info() const override { return __type_info__; }

  std::string tensor_name;
  static const char* __type_info__;
};

const char* BufferUFNode::__type_info__ = "BufferUFNode";

struct IRReplaceTensorMutator : ir::IRMutator<> {
  const std::map<std::string, ir::Tensor>& tensor_map;
  explicit IRReplaceTensorMutator(
      const std::map<std::string, ir::Tensor>& tensor_map)
      : tensor_map(tensor_map) {}
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    auto it = tensor_map.find(op->name);
    if (it != tensor_map.end()) {
      *expr = Expr(it->second);
    }
  }
};

}  // namespace

}  // namespace optim
}  // namespace cinn

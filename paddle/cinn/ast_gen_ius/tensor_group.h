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
#include <absl/container/flat_hash_map.h>

#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn {
namespace ast_gen_ius {

/* Collection used for Tensors, used in AST generation */
class TensorGroup {
 public:
  explicit TensorGroup(const std::vector<ir::Tensor>& tensors);
  ~TensorGroup();

  bool Contain(const std::string& name) const;

  void Insert(const ir::Tensor& tensor);

  ir::Tensor Get(const std::string& name);

  std::set<ir::Tensor> GetAllTensors();

  void CtrlDepend(const ir::Tensor& tensor, const ir::Tensor& to_dep);

  std::set<ir::Tensor> GetCrtlDepTensors(const std::string& tensor_name);

  std::string GetShareMemRootName(const std::string& tensor_name);

  void ShareMemoryBuffer(const ir::Tensor& tensor, const ir::Tensor& to_share);

  absl::flat_hash_map<std::string, ir::Tensor> AllocateBuffers();

  // Returns tensors in topological order and remove those args
  // Becuase the order is used for generating function body, we don't have to
  // generate args
  std::vector<ir::Tensor> GetGenFuncTopoOrder(
      const std::vector<ir::Tensor>& func_args = {});

  bool HasMarkedReduceInit(const std::string& tensor_name) const;

  // Marks a tensor needs to do reduce init
  ir::Tensor MarkReduceInit(const std::string& tensor_name);

 private:
  absl::flat_hash_map<std::string, ir::Tensor> name_to_tensor_;

  // Stores vector of tensor names, which the key tensor depends on
  std::unordered_map<std::string, std::unordered_set<std::string>> ctrl_dep_;

  // Keeps Union Find Set style, each tensor name whose buffer is shared maps to
  // the same name tensor
  std::unordered_map<std::string, std::string> share_memory_tensor_;

  std::unordered_set<std::string> tensor_name_needs_reduce_init_;
};

}  // namespace ast_gen_ius
}  // namespace cinn

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
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace ast_gen_ius {

/**
 * Collection which maintains the relation between Tensor(s) such as control
 * dependency, memory sharing ... it is used in AST generation
 */
class TensorGroup {
 public:
  /**
   * Constructor for a TensorGroup, the argument tensors should be output tensor
   * arguments of the AST body to be generated. The dependent tensors of the
   * output tensors will be collected during construction.
   */
  explicit TensorGroup(const std::vector<ir::Tensor>& tensors);

  /**
   * Constructor for a TensorGroup, the argument tensors should be output tensor
   * arguments of the AST body to be generated. The dependent tensors of the
   * output tensors will be collected during construction.
   */
  explicit TensorGroup(
      const std::unordered_map<std::string, ir::Tensor>& tensor_map);

  /**
   * Destructor.
   */
  ~TensorGroup();

  void ShowLog() const;

  /**
   * Returns true if TensorGroup collection contains a tensor with input name.
   */
  bool Contain(const std::string& name) const;

  /**
   * Insert a Tensor into TensorGroup collection.
   */
  void Insert(const ir::Tensor& tensor);

  /**
   * Returns the Tensor in TensorGroup collection with the given name.
   */
  ir::Tensor Get(const std::string& name);

  /**
   * Returns all Tensors in TensorGroup.
   */
  std::set<ir::Tensor> GetAllTensors();

  /**
   * Mark `tensor` depends on `to_dep`.
   */
  void CtrlDepend(const ir::Tensor& tensor, const ir::Tensor& to_dep);

  /**
   * Get all tensors which the tensor with given name depends on.
   */
  std::set<ir::Tensor> GetCtrlDepTensors(const std::string& tensor_name);

  /**
   * Get Union-Find set algorithm root tensor name which shares memory with the
   * tensor whose name is the input.
   */
  std::string GetShareMemRootName(const std::string& tensor_name);

  /**
   * Mark two tensors share memory, it only marks using Union-Find set
   * algorithm, doesn't do really memory sharing/allocation
   */
  void MarkShareMemBuffer(const ir::Tensor& tensor, const ir::Tensor& to_share);

  /**
   * Allocate buffers for Tensors in TensorGroup, it handles the shared memory
   * using Union-Find set algorithm.
   */
  absl::flat_hash_map<std::string, ir::Tensor> AllocateBuffers();

  /**
   * Returns tensors in topological order and remove those args
   * Because the order is used for generating function body, we don't have to
   * generate args
   */
  std::vector<ir::Tensor> GetGenFuncTopoOrder(
      const std::vector<ir::Tensor>& func_args = {});

 private:
  /** collection of output tensor names */
  std::set<std::string> output_tensor_names_;

  /** collection of all tensors in this TensorGroup */
  absl::flat_hash_map<std::string, ir::Tensor> name_to_tensor_;

  /** Stores vector of tensor names, which the key tensor depends on */
  std::unordered_map<std::string, std::unordered_set<std::string>> ctrl_dep_;

  /**
   * Keeps Union Find Set style, each tensor name whose buffer is shared, maps
   * to the same name tensor.
   */
  std::unordered_map<std::string, std::string> share_memory_tensor_;
};

}  // namespace ast_gen_ius
}  // namespace cinn

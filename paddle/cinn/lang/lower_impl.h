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

#pragma once
#include <absl/container/flat_hash_map.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/buffer_assign.h"
#include "paddle/cinn/optim/compute_inline_expand.h"
#include "paddle/cinn/optim/fold_cinn_call_arguments.h"
#include "paddle/cinn/optim/optimize.h"
#include "paddle/cinn/optim/replace_call_with_expr.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/poly/ast_gen.h"

namespace cinn {

namespace poly {
class Stage;
}  // namespace poly

namespace lang {
namespace detail {

/**
 * After the AstGen build the forloop from isl exprs, all the ISL Call nodes
 * should be mapped to the corresponding CINN expressions, there should be no
 * remaining.
 */
void CheckNoIslCallRemains(const Expr* expr);

/**
 * A Computation graph node.
 */
struct CompuGraphNode : public cinn::common::GraphNode {
  explicit CompuGraphNode(ir::Tensor tensor) : tensor(tensor) {}

  ir::Tensor tensor;

  std::string id() const override;
  const char* type_info() const override;
  static const char* __type_info__;
};

class LowerImpl {
 public:
  std::vector<ir::LoweredFunc> operator()();

  /**
   * Get the computational graph.
   */
  const cinn::common::Graph* comp_graph() const { return compu_graph_.get(); }

  /**
   * \brief generate the argument list of the final output function.
   * We put the scalar_args in front of tensor_args, e.g. get tensor_args{A,B},
   * scalar_args{m}, the final argument list is {m, A, B}, the input and output
   * tensor can be mixed in the tensor_args, the kInput and kOutput token will
   * deduce from their usage in the computation.
   */
  std::vector<ir::Argument> GenerateFunctionArgumentList(Expr fn_body);

  std::vector<ir::Argument> GenFuncArgForSplitKernel(
      Expr func_iterator, std::vector<ir::Tensor> temp_tensors);

  /**
   * \brief generate the body expression of the final output function.
   */
  std::vector<Expr> GenerateFunctionBody(const poly::Schedule* schedule);

 private:
  /**
   * \brief Collect the temporary tensors.
   * A temporary tensor is one that is in the computation graph, not inlined and
   * not in the tensor_args(similar to a temporary variable inside function).
   */
  std::vector<Tensor> CollectTemporaryTensors();

  /**
   * \brief Check both the tensor_args and scalar_args not contain duplication
   * (different argument with the same name).
   */
  void CheckArgsUnique();

  /**
   * \brief Get a map, for each tensor in the tensor_args, map from name to
   * itself.
   */
  inline absl::flat_hash_map<std::string, Tensor> GenTensorArgMap();

  /**
   * \brief Get a map, for each tensor in the computation graph, map from name
   * to itself.
   */
  inline absl::flat_hash_map<std::string, Tensor> GenAllTensorMap();

  /**
   * \brief Get all the tensors, including the input, output and temporary ones.
   */
  std::vector<Tensor> CollectAllTensors();

  /**
   * \brief Collect the extra dependencies between tensors.
   *
   * The extra dependencies include
   * 1. the control deps in Stage.
   *
   * TODO(Superjomn) remove the field `extra_depend_stages`
   */
  std::set<std::pair<std::string, std::string>> CollectExtraDependencies()
      const;

 private:
  const std::string& fn_name_;
  const std::vector<Tensor>& tensor_args_;
  const std::vector<Var>& scalar_args_;
  std::vector<Tensor> temp_tensor_args_;
  Target target_;

  //! A computation graph generated from the tensor_args and scalar_args.
  std::unique_ptr<cinn::common::Graph> compu_graph_;

  //! CUDA axis info for this function.
  std::vector<ir::CudaAxisInfo> cuda_axis_info_;

  bool support_ir_schedule_ = false;
};

/**
 * \brief Tell whether a tensor contains some GPU related information, such some
 * schedule.
 */
bool TensorContainsGPUInfo(ir::Tensor t, poly::Stage* stage);

/**
 * Mark the PolyFor as Vectorized if it is scheduled Vectorize in Stage.
 */
struct MarkVectorizeMutator : public ir::IRMutator<Expr*> {
  const std::map<std::string, ir::VectorizeInfo>& vectorizes;

  explicit MarkVectorizeMutator(const std::map<std::string /*tensor name*/,
                                               ir::VectorizeInfo>& vectorizes)
      : vectorizes(vectorizes) {}

  void operator()(Expr* expr) { ir::IRMutator<Expr*>::Visit(expr, expr); }

  // NOTE This mutator takes PolyFor as input, not For.
  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    forloop_stack.push_back(node);
    ir::IRMutator<ir::Expr*>::Visit(op, expr);
    forloop_stack.pop_back();
  }

  // each statement in ISL is bound to a Store node.
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* tensor_n = op->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor_n,
        ::common::errors::InvalidArgument("Sorry, but op->tensor is null"));
    auto it = vectorizes.find(tensor_n->name);
    if (it != vectorizes.end()) {
      PADDLE_ENFORCE_LT(
          it->second.level,
          forloop_stack.size(),
          ::common::errors::InvalidArgument(
              "Required it->second.level shall be less than "
              "forloop_stack.size()."
              "But receive it->second.level = %d, forloop_stack.size() = %d ",
              it->second.level,
              forloop_stack.size()));
      forloop_stack[it->second.level]->set_vectorize_info(it->second);
      PADDLE_ENFORCE_EQ(
          it->second.valid(),
          true,
          ::common::errors::InvalidArgument("it->second.valid() is false"));
    }
  }

  std::vector<ir::PolyFor*> forloop_stack;
};

/**
 * Mark the PolyFor as Unroll if is called Unroll in Stage.
 */
struct MarkUnrollMutator : public ir::IRMutator<Expr*> {
  std::map<std::string, std::set<int> /*level*/> unrolls;

  explicit MarkUnrollMutator(
      const std::map<std::string, std::set<int>>& unrolls)
      : unrolls(unrolls) {}

  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    stack.push_back(node);
    ir::IRMutator<>::Visit(op, expr);
    stack.pop_back();
  }

  // each statement in ISL is bound to a Store node.
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* tensor_n = op->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor_n,
        ::common::errors::InvalidArgument("Sorry, but op->tensor is null"));
    auto it = unrolls.find(tensor_n->name);
    if (it != unrolls.end()) {
      for (int level : it->second) {
        VLOG(1) << "Mark " << level << " Unrolled";
        PADDLE_ENFORCE_LT(level,
                          stack.size(),
                          ::common::errors::InvalidArgument(
                              "Required level shall be less than stack.size()."
                              "But receive level = %d, stack.size() = %d ",
                              level,
                              stack.size()));
        stack[level]->set_unrolled();
      }
    }
  }

  std::vector<ir::PolyFor*> stack;
};

/**
 * Mark the PolyFor as Parallel if is called Parallel in Stage.
 */
struct MarkParallelMutator : public ir::IRMutator<Expr*> {
  std::map<std::string, std::set<int> /*level*/> parallels;

  explicit MarkParallelMutator(
      const std::map<std::string, std::set<int>>& parallels)
      : parallels(parallels) {}

  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    stack.push_back(node);
    ir::IRMutator<>::Visit(op, expr);
    stack.pop_back();
  }

  // each statement in ISL is bound to a Store node.
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* tensor_n = op->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor_n,
        ::common::errors::InvalidArgument("Sorry, but op->tensor is null"));
    auto it = parallels.find(tensor_n->name);
    if (it != parallels.end()) {
      for (int level : it->second) {
        VLOG(1) << "Mark " << level << " Parallelled";
        PADDLE_ENFORCE_LT(level,
                          stack.size(),
                          ::common::errors::InvalidArgument(
                              "Required level shall be less than stack.size()."
                              "But receive level = %d, stack.size() = %d ",
                              level,
                              stack.size()));
        stack[level]->set_parallel();
      }
    }
  }

  std::vector<ir::PolyFor*> stack;
};

}  // namespace detail
}  // namespace lang
}  // namespace cinn

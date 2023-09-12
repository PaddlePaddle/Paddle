// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule_error.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_mutator.h"
#include "paddle/cinn/utils/random_engine.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

// Self-defined operator to support std::set<Expr>
struct CompExpr {
  bool operator()(const Expr& left, const Expr& right) const {
    return utils::GetStreamCnt(left) < utils::GetStreamCnt(right);
  }
};

// Self-defined operator to support std::set<Var>
struct CompVar {
  bool operator()(const Var& left, const Var& right) const {
    return left->name < right->name;
  }
};

struct MappingVarToExprMutator : public ir::IRMutator<> {
  explicit MappingVarToExprMutator(
      const std::map<Var, Expr, CompVar>& replacing_map)
      : replacing_map_(replacing_map) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (replacing_map_.count(op->as_var_ref())) {
      *op = replacing_map_.at(op->as_var_ref());
    }
  }

 private:
  const std::map<Var, Expr, CompVar>& replacing_map_;
};

struct FindLoopsVisitor {
  explicit FindLoopsVisitor(const Expr& block) : block_(block) {}

  std::vector<Expr> operator()(const Expr* expr) {
    CHECK(block_.As<ir::ScheduleBlockRealize>());
    visit_end = false;
    Visit(expr);
    return result;
  }

 private:
  void Visit(const Expr* expr) {
    if (visit_end || !expr->defined()) return;
    if (expr->As<ir::For>()) {
      father_loops.emplace_back(*expr);
      Visit(&(expr->As<ir::For>()->body));
      father_loops.pop_back();
    } else if (expr->As<ir::ScheduleBlockRealize>()) {
      if (!expr->As<ir::ScheduleBlockRealize>()->iter_values.empty() &&
          (*expr == block_)) {
        result = father_loops;
        visit_end = true;
        return;
      } else {
        Visit(&(expr->As<ir::ScheduleBlockRealize>()->schedule_block));
      }
    } else if (expr->As<ir::ScheduleBlock>()) {
      Visit(&(expr->As<ir::ScheduleBlock>()->body));
    } else if (expr->As<ir::Block>()) {
      for (auto& n : expr->As<ir::Block>()->stmts) Visit(&n);
    } else if (expr->As<ir::IfThenElse>()) {
      Visit(&(expr->As<ir::IfThenElse>()->true_case));
      Visit(&(expr->As<ir::IfThenElse>()->false_case));
    }
  }

  std::vector<Expr> father_loops{};
  std::vector<Expr> result{};
  bool visit_end{false};
  const Expr& block_;
};

/**
 * \brief Given a ScheduleBlockRealize node, return the Store tensor in its
 * body.
 * @param block The given ScheduleBlockRealize node
 * @return The Store tensor in block
 */
Tensor GetTensor(const Expr& block);

struct FindBlocksVisitor {
  explicit FindBlocksVisitor(const std::string& block_name = "")
      : block_name_(block_name) {}

  std::vector<Expr> operator()(const Expr* expr) {
    Visit(expr);
    return result;
  }

 private:
  void Visit(const Expr* expr) {
    if (!expr->defined()) return;
    if (!block_name_.empty() && !result.empty()) return;
    if (expr->As<ir::For>()) {
      Visit(&(expr->As<ir::For>()->body));
    } else if (expr->As<ir::ScheduleBlockRealize>()) {
      if (!expr->As<ir::ScheduleBlockRealize>()->iter_values.empty()) {
        auto* schedule_block = expr->As<ir::ScheduleBlockRealize>()
                                   ->schedule_block.As<ir::ScheduleBlock>();
        if (block_name_.empty() || schedule_block->name == block_name_) {
          result.emplace_back(*expr);
        }
      } else {
        Visit(&(expr->As<ir::ScheduleBlockRealize>()->schedule_block));
      }
    } else if (expr->As<ir::ScheduleBlock>()) {
      Visit(&(expr->As<ir::ScheduleBlock>()->body));
    } else if (expr->As<ir::Block>()) {
      for (auto& n : expr->As<ir::Block>()->stmts) Visit(&n);
    } else if (expr->As<ir::IfThenElse>()) {
      Visit(&(expr->As<ir::IfThenElse>()->true_case));
      Visit(&(expr->As<ir::IfThenElse>()->false_case));
    }
  }
  std::string block_name_;
  std::vector<Expr> result{};
};

struct CacheBlockInfo {
  /*! \brief The tensor to be read. */
  Tensor read_tensor;
  /*! \brief The tensor to be written. */
  Tensor write_tensor;
  /*! \brief The tensor allocation to be inserted into the block signature. */
  Tensor alloc;
  /*! \brief The AST node whose body is where the cache stage should be
   * inserted. */
  Expr loc_block;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  int loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Expr cache_block;
};

// a struct to present the min value and the extent of a iterable range,
// where it is represented as a semi-closed interval, i.e [min, min + extent)
struct IterRange {
  IterRange(Expr begin, Expr length) : min(begin), extent(length) {}

  Expr min;
  Expr extent;
};

/**
 * \brief Given a ScheduleBlockRealize node, return the index-th Load tensor in
 * its body.
 * @param block The given ScheduleBlockRealize node
 * @param index The index of Load tensor
 * @return The index-th Load tensor in block
 */
Tensor GetReadTensor(const Expr& block, int index);

/**
 * \brief Given a For node, return its extent as int.
 * @param loop The given For node
 * @return The extent of For node
 */
int GetLoopExtent(const Expr& loop);

/**
 * \brief Given a vector of Exors, return whether they contain a var with
 * specific name.
 * @param exprs The given vector of Exprs
 * @param var_name The name of specific var
 * @return Whether there is a Var with the same name as var_name
 */
bool ContainVar(const std::vector<Expr>& exprs, const std::string& var_name);

/**
 * \brief Given a _LoweredFunc_, set its cuda_axis_info based on its func_body.
 * @param lowered_func A pointer to the given _LoweredFunc_
 */
void SetCudaAxisInfo(Expr* lowered_func);

/*!
 * \brief Check if a Expr node contains a ScheduleBlockRealize node.
 * \param container The container Expr node.
 * \param expr The node we want to find.
 * \return If the container contains the expr.
 */
bool Contains(const Expr& container, const Expr& expr);

/**
 * \brief Given a For loop, return the next For loop in its body.
 * @param for_loop The given For loop.
 * @return The next For loop.
 */
Expr GetNextForLoop(const Expr& for_loop);

/**
 * \brief Given two For loops, return all ir::IfThenElse nodes between them.
 * @param top The given top For loop.
 * @param bottom The given bottom For loop.
 * @return All ir::IfThenElse nodes between them.
 */
std::vector<Expr> GetIfThenElseInRange(const Expr& top, const Expr& bottom);

/**
 * Replace Vars in replaced to Exprs in candidates in source. Vars -> Exprs is
 * one-to-one correspondence.
 * @param source The Expr we will implement the change.
 * @param replaced The Vars to be replaced.
 * @param candidates The Exprs to replace Vars in replaced.
 */
void ReplaceExpr(Expr* source,
                 const std::vector<Var>& replaced,
                 const std::vector<Expr>& candidates);

/**
 * Validate the factors param of Split. We will check if factors are validate
 * and change -1 to positive integer.
 * @param factors The original factors.
 * @param total_extent The extent of the loop to be splitted.
 * @return return The valiated factors.
 */
std::vector<int> ValidateFactors(const std::vector<int>& factors,
                                 int total_extent,
                                 const ModuleExpr& module_expr);

void CHECKRfactorValidation(const Expr& rf_loop, int rf_axis);

/**
 * Return loops that contain the expr.
 * @param expr The expr.
 * @param root The root of the whole AST.
 * @return return Loops in AST that contain the expr.
 */
std::vector<Expr> GetLoopsOfExpr(const Expr& expr, const Expr& root);

/**
 * Given an index Expr and all vars' range, return the accessed range in this
 * indice.
 * @param index The Expr of a specified indice.
 * @param iter_vars The vars in expr.
 * @param iter_range Each var's range.
 * @return return an IterRange represents the accessed range of this indice, If
 * it is not constant, return corresponding tensor's shape.
 */
IterRange GetAccessedRange(const Expr& index,
                           const std::vector<Var>& iter_vars,
                           const std::vector<IterRange>& iter_ranges);

/**
 * Given a ScheduleBlockRealize, an AST root, a tensor and its tensor_indices,
 * return the accessed buffer region of the tensor in block.
 * @param block The ScheduleBlockRealize.
 * @param tensor_indices The tensor's indices.
 * @param tensor The tensor.
 * @param root The root of whole AST.
 * @return return The accessed buffer region of the tensor in block.
 */

std::vector<IterRange> CalculateTensorRegions(
    const Expr& block,
    const std::vector<Expr>& tensor_indices,
    const Tensor& tensor,
    const Expr& root);

/**
 * Return n-th access tensor in block
 * @param block The ScheduleBlockRealize.
 * @param index The index indicating which tensor we want to get.
 * @param is_write We want to get write tensor or read tensor.
 * @return return The n-th access tensor in block. Should be ir::Store(is_write)
 * or ir::Load(!is_write).
 */
Expr GetNthAccessExpr(const Expr& block, int index, bool is_write);

/**
 * Make a tensor's cache tensor.
 * @param tensor The original tensor.
 * @param memory_type The memory type of the cache tensor.
 * @return return The tensor's cache tensor.
 */
Tensor MakeCacheTensor(const Tensor& tensor, const std::string& memory_type);

/**
 * Make a the cache tensor's block.
 * @param buffer_region The accessed region of cache tensor.
 * @param info The information of cache block.
 * @param memory_type The memory type of cache tensor.
 * @param device_api The device api of this Expr.
 * @return return ScheduleBlockRealize of the cache tensor.
 */
Expr MakeCacheBlock(const std::vector<IterRange>& buffer_ranges,
                    CacheBlockInfo* info,
                    const std::string& memory_type,
                    DeviceAPI device_api);

/**
 * Fidn cache tensor block's insertion point in the whole AST(root).
 * @param root The whole AST.
 * @param info The information of cache block.
 * @param is_write Are we inserting a write cache tensor or a read cache tensor.
 */
void FindInsertionPoint(const Expr& root, CacheBlockInfo* info, bool is_write);

/**
 * \brief Given a vector of For loops, return a set of them.
 * @param loops The given vector of For loops.
 * @return A set containing all the For loops in loops.
 */
const std::set<Expr, CompExpr> CollectLoopsToSet(
    const std::vector<Expr>& loops);

/**
 * \brief Given a set of For loops, return the boundary among them.
 * @param loop_set The given set of For loops.
 * @return A pair of the boundary among For loops.(The top For and bottom For)
 */
std::pair<Expr, Expr> GetBoundaryOfReorderRange(
    const std::set<Expr, CompExpr>& loop_set);

/**
 * \brief Given two For loops, return all loops between them.
 * @param top The top For loop.
 * @param bottom The bottom For loop.
 * @return A vector containing all For loops between the boundary, stored in
 * ascending order.
 */
std::vector<Expr> GetLoopsInRange(const Expr& top, const Expr& bottom);

/**
 * \brief Given params, construct a new loop.
 */
Expr ConstructNewLoopChain(const std::vector<Expr>& chain,
                           const std::vector<Expr>& ordered_loops,
                           const std::set<Expr, CompExpr>& loop_set,
                           std::vector<Expr>& if_nodes);  // NOLINT

/*!
 * \brief Find producers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its producers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's producers(ScheduleBlockRealize nodes) in root.
 */
std::vector<Expr> GetProducers(const Expr& block, const Expr& root);

/*!
 * \brief Find consumers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its consumers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's consumers(ScheduleBlockRealize nodes) in root.
 */
std::vector<Expr> GetConsumers(const Expr& block, const Expr& root);

/*!
 * \brief Check if the params of ComputeAt is validate.
 * \param block The block node we want to move in ComputeAt.
 * \param loop The for node we want to put the block under in ComputeAt.
 * \param root The root ScheduleBlockRealize node of block and loop.
 */
void CheckComputeAtValidation(const Expr& block,
                              const Expr& loop,
                              const Expr& root);

/*!
 * \brief Insert a new ScheduleBlockRealize in a loop's body(under its
 * IfThenElse Node, if any) \param for_loop The for loop whose body we want to
 * modify \param insertion The ScheduleBlockRealize we want to insert \param
 * index The position index of the for_loop body `stmts` to be inserted:
 *        - `index = -1` means inserted into the tail
 *        - otherwise, it should be a index between [0, stmts size)
 */
void InsertBlock(Expr& for_loop,  // NOLINT
                 const Expr& insertion,
                 int index = 0);  // NOLINT

/*!
 * \brief Make a union of two range. The detailed function is :
 * new_range.min = min(range1.min, range2.min)
 * new_range.extent = max(range1.min + range1.extent, range2.min +
 * range2.extent) - new_range.min Notice that the pair<Expr, Expr> indicates a
 * range's min and extent. \param range1 The first range \param range2 The
 * second range \return The union of these two ranges
 */
IterRange RangeUnion(const IterRange& range1, const IterRange& range2);

/*!
 * \brief Calculate the required buffer region given a block and its required
 * blocks. For example, if block is : B[i0, j0] = A[i0, j0] loop is : for (i, 0,
 * 64) { for (j, 0, 64) { C[i, j] = B[i, j]
 *   }
 * }
 * And required_blocks is :
 * C[i, j] = B[i, j]
 * Then we get the required B's region:
 * B[i, j], where:
 * i : [i, i]
 * j : [0, 64]
 * \param block The ScheduleBlockRealize node begin required
 * \param loop The loop where we will insert the block under it
 * @param root The root of the whole AST.
 * \param required_blocks vector of ScheduleBlockRealize nodes that require the
 * block \param is_store_provided Whether Store nodes of the block provide the
 * tensor, true means it is in compute_at case, otherwise false means in
 * reverse_compuate_at case \return Each index's range of block's tensor.
 * Indicating the buffer region being required.
 */
std::vector<IterRange> CalculateRequiredRegions(
    const Expr& block,
    const Expr& loop,
    const Expr& root,
    const std::vector<Expr>& required_blocks,
    bool is_store_provided = true);

Expr CheckComputeInlineValidationAndGetStore(const Expr& schedule_block,
                                             const Expr& root);

/*!
 * \brief Check if the reverse compute inline validation passes for a given
 * schedule block and root expression, and retrieve the store expression if so.
 * Reverse compute inline validation ensures that the outputs of a loop nest are
 * properly computed in reverse order. \param schedule_block The schedule block
 * to check. \param root The root expression of the loop nest. \return A tuple
 * containing the load that will be inlined, the store that will be inlined and
 * the target store.
 */
std::tuple<Expr, Expr, Expr> CheckReverseComputeInlineValidationAndGetExprs(
    const Expr& schedule_block, const Expr& root);

/*!
 * \brief Get the prime factors of a number.
 * For example, 12 = 2^2 * 3^1, then the return value is {2: 2, 3: 1}.
 * \param n The number to be factorized.
 * \return A map of prime factors and their corresponding exponents.
 */
std::unordered_map<int, int> PrimeFactorize(int n);

/*!
 * \brief Given a number returns the form of the product of its n factors
 * For example:
 *  n = 2, dividend = 12, return one of {2, 6}, {6, 2}, {3, 4}, {4, 3}
 * \param seed The random number generator to use.
 * \param n The number to be factorized.
 * \param dividend The dividend of the number.
 */
std::vector<int> SampleTile(utils::LinearRandomEngine::StateType* rand_seed,
                            int n,
                            int dividend);
}  // namespace ir
}  // namespace cinn

// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include <optional>
#include <unordered_map>
#include <vector>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace common {

class IterMapToExprNormalizer : public ir::IRMutator<> {
 public:
  explicit IterMapToExprNormalizer(const SymbolicExprAnalyzer& analyzer)
      : analyzer_(analyzer) {}

  void Convert(Expr* expr) { Visit(expr, expr); }

 private:
  void Visit(const Expr* expr, Expr* op) override;

  ir::IndexExpr ConvertIterSum(ir::IterSum* expr);

  ir::IndexExpr ConvertIterSplit(ir::IterSplit* expr);

 private:
  common::SymbolicExprAnalyzer analyzer_;
};

class IterMapRewriter : public ir::IRMutator<> {
 public:
  explicit IterMapRewriter(const std::vector<ir::Var>& input_iters,
                           const SymbolicExprAnalyzer& analyzer)
      : analyzer_(analyzer) {
    for (const auto& iter : input_iters) {
      // Remove unit extent `for` ?
      // if (IsOne(iter->upper_bound)) {
      //   var_map_[iter->name] = ir::IterSum::Make({}, iter->lower_bound);
      // }
      if (IsZero(iter->lower_bound)) {
        auto tmp =
            ir::IterMark::Make(ir::IndexExpr(iter.ptr()), iter->upper_bound);
        auto mark = tmp.As<ir::IterMark>();
        var_map_[iter->name] = ir::IterSplit::Make(tmp);
        input_marks_.push_back(*mark);
      } else {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "iter should start from 0, but got %d", iter->lower_bound));
      }
    }
  }

  void Visit(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Rewrite(Expr* expr) {
    IRMutator::Visit(expr, expr);
    *expr = ToIterSum(*expr);
  }

  void Visit(const ir::_Var_* op, Expr* expr) override;

  void Visit(const ir::Add* op, Expr* expr) override;

  void Visit(const ir::Sub* op, Expr* expr) override;

  void Visit(const ir::Mul* op, Expr* expr) override;

  void Visit(const ir::Div* op, Expr* expr) override;

  void Visit(const ir::Mod* op, Expr* expr) override;

 private:
  static Expr ToIterSum(const Expr& expr);

  static void AddToLhs(ir::IterSum* lhs, const ir::IterSplit& rhs, int sign);

  static void AddToLhs(ir::IterSum* lhs, const ir::IterSum& rhs, int sign);

  static void MulToLhs(ir::IterSum* lhs, const ir::IndexExpr& rhs);

  Expr PreprocessDividend(const Expr& dividend);

  Expr SplitDivConst(Expr lhs, ir::IndexExpr base, ir::IndexExpr rhs);

  Expr SplitModConst(Expr lhs, ir::IndexExpr base, ir::IndexExpr rhs);

  /*!
   * \brief This function will find the iter which has the expected scale.
   * For example:
   * expr:
   *  IterSum(IterSplit(IterMark(i), scale = 32),
   *          IterSplit(IterMark(j), scale = 8),
   *          base = 0)                    // i * 32 + j * 8
   * ret: `1` when expected_scale = 8
   * ret: `0` when expected_scale = 32
   *
   * \param expr the input IterSum to search.
   * \param skip_flag the flag to indicate whether a Iter should be skipped.
   * \param match_source Whether to only match the same source.
   * \param rbegin the last position in reverse searching. -1 means the last.
   * \param first_possible_unit_extent_pos the first possible position of the
   * unit extent.
   *  \return the index of the Iter with expected scale. return -1
   * if not found.
   */
  int32_t FindSplitWithExactScale(const ir::IterSum& expr,
                                  const std::vector<bool>& skip_flag,
                                  const ir::IndexExpr& expected_scale,
                                  const Expr& match_source,
                                  int32_t rbegin = -1,
                                  int32_t first_possible_unit_extent_pos = 0);

  /*!
   * \brief Find the first possible position where IterSplit->extent = 1.
   * \param expr the input IterSum to search.
   * \return the index of the first IterSplit with extent = 1,
   * return IterSum.args.size if not found.
   */
  int32_t FindFirstPossibleUnitExtentIndex(const ir::IterSum& expr);

  /*!
   * \brief This function will find the base Iter which has the smallest scale.
   *
   * For example:
   * expr:
   *  IterSum(IterSplit(IterMark(i), scale = 32),
   *          IterSplit(IterMark(j), scale = 8),
   *          base = 0)                    // i * 32 + j * 8
   *
   * ret: `1` when match_source = nullptr
   * ret: `0` when match_source = IterMark(i)
   *
   * \param expr the input IterSum to search.
   * \param skip_flag the flag to indicate whether a Iter should be skipped.
   * \param match_source Whether to only match the same source.
   * \param rbegin the last position in reverse searching. -1 means the last.
   * \return the index of the base Iter. return -1 if not found.
   */
  int32_t FindBaseSplit(const ir::IterSum& expr,
                        const std::vector<bool>& skip_flag,
                        const Expr& match_source,
                        int32_t rbegin = -1);

  /*!
   * \brief TryFuse will create new IterMark and returns an aggregated IterSum
   * that only has one IterSplit with the new IterMark.
   *
   * For example:
   * expr:
   *  IterSum(IterSplit(IterMark(i), scale = 32),
   *          IterSplit(IterMark(j), scale = 8),
   *          base = 0)                    // i * 32 + j * 8
   * ret:
   *  IterSum(IterSplit(IterMark(i * 4 + j), scale = 8),
   *          base = 0)                    // Treat `i * 4 + j` as a IterMark
   *
   * \param expr the input IterSum.
   * \return the IterSum after fused.
   */
  std::optional<Expr> TryFuse(const Expr& expr);

  /*!
   * \brief TryFuseSameSource will simplify the IterSum by fusing IterSplits
   * with same source. same source means the IterSplits have same IterMark.
   *
   * For example:
   * expr:
   *  IterSum(IterSplit(IterMark(f), lower = 4, ext = 8 scale = 4),
   *          IterSplit(IterMark(f), lower = 1, ext = 4 scale = 1),
   *          base = 0)                        // f /4 * 4 + f % 4
   * ret:
   *  IterSum(IterSplit(IterMark(f), scale = 1),
   *          base = 0)                        // f
   *
   * \param expr the input IterSum
   * \return the IterSum after fused.
   */
  std::optional<Expr> TryFuseSameSource(const Expr& expr);

  std::unordered_map<std::string, Expr> var_map_;
  std::vector<ir::IterMark> input_marks_;
  std::unordered_map<Expr, Expr> sum_fuse_map_;
  common::SymbolicExprAnalyzer analyzer_;
};

class SimplifyBlockBinding : public ir::IRMutator<> {
 public:
  explicit SimplifyBlockBinding(const std::vector<ir::Var>& loop_var,
                                const SymbolicExprAnalyzer& analyzer)
      : loop_var_(loop_var), analyzer_(analyzer) {}

  static void SimplifyBindings(ir::Expr expr,
                               const std::vector<ir::Expr>& loop_srefs,
                               const SymbolicExprAnalyzer& analyzer) {
    std::vector<ir::Var> loop_var;
    for (const ir::Expr& sref : loop_srefs) {
      const ir::For* loop = sref.As<ir::For>();
      loop_var.emplace_back(loop->loop_var);
    }
    SimplifyBlockBinding(loop_var, analyzer)(&expr);
  }
  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::For* op, Expr* expr) override;

  void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override;

  std::vector<ir::Var> loop_var_;
  common::SymbolicExprAnalyzer analyzer_;
};

void IterMapSimplify(std::vector<Expr>& indices,  // NOLINT
                     const std::vector<cinn::ir::Var>& input_iters,
                     const SymbolicExprAnalyzer& analyzer);
}  // namespace common
}  // namespace cinn

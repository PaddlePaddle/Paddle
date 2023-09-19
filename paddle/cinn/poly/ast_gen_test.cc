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

#include "paddle/cinn/poly/ast_gen.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/poly/schedule.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace poly {

using namespace cinn::ir;  // NOLINT

TEST(TransIdentityExtentToContextId, basic) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl::set set(ctx, "{ s[i,j=0,k] : 0<=i<12 and 12<k<32 }");
  auto new_set = TransIdentityExtentToContextId(set);
  LOG(INFO) << new_set;

  ASSERT_EQ(utils::GetStreamCnt(new_set),
            "[_const_0] -> { s[i, j, k] : _const_0 <= 1 and 0 <= i <= 11 and 0 "
            "<= j <= _const_0 and 13 <= k <= 31 }");
}

TEST(TransIdentityExtentToContextId, basic1) {
  isl_ctx* ctx = isl_ctx_alloc();
  isl::set set(ctx, "[n] -> { s[i,j=0,k] : 0<=i<n and 12<k<32 }");
  LOG(INFO) << "set: " << set;
  auto new_set = TransIdentityExtentToContextId(set);
  LOG(INFO) << new_set;
}

TEST(AstGen_Build, not_delete_length1_loop) {
  std::vector<Expr> origin_shape = {Expr(10), Expr(10), Expr(10), Expr(10)};
  for (int num_len1 = 0; num_len1 <= origin_shape.size(); ++num_len1) {
    std::vector<int> index_length1(origin_shape.size(), 0);
    for (int i = 1; i <= num_len1; ++i) {
      index_length1[index_length1.size() - i] = 1;
    }
    do {
      // Create shape that has 'num_len1' loops with length 1
      // And this loop iterates for every combination of possible length 1
      std::vector<Expr> len1_shape = origin_shape;
      for (int i = 0; i < origin_shape.size(); ++i) {
        if (index_length1[i] == 1) {
          len1_shape[i] = Expr(1);
        }
      }
      LOG(INFO) << "index_length1 hint = " << index_length1[0]
                << index_length1[1] << index_length1[2] << index_length1[3];
      lang::Placeholder<float> A("A", len1_shape);
      Tensor B = lang::Compute(
          len1_shape,
          [&](const std::vector<Expr>& indice) {
            return lang::Relu(A(indice), 0);
          },
          "relu_test");

      StageMap stage_map = CreateStages({B});
      std::vector<cinn::poly::Stage*> stages;
      stages.push_back(stage_map[B]);

      std::unique_ptr<Schedule> schedule = poly::CreateSchedule(
          stages,
          poly::ScheduleKind::Poly,
          std::vector<std::pair<std::string, std::string>>());

      for (auto& group : schedule->groups) {
        isl::set context(Context::isl_ctx(), "{:}");
        poly::AstGen gen(context, stages, group);
        isl::ast_node ast = gen.Build();
        ir::Expr e;
        poly::IslAstNodeToCinnExpr(ast, gen.domain().as_set(), &e);
        LOG(INFO) << "Domain = " << gen.domain().as_set();
        LOG(INFO) << "Expr for not delete length1 loop";
        LOG(INFO) << "\n" << e;

        std::stringstream ss;
        ss << e;
        std::string expr_str = ss.str();
        std::string target_str = R"ROC(poly_for (i, 0, (i <= 9), 1)
{
  poly_for (j, 0, (j <= 9), 1)
  {
    poly_for (k, 0, (k <= 9), 1)
    {
      poly_for (a, 0, (a <= 9), 1)
      {
        relu_test(i, j, k, a)
      }
    }
  }
})ROC";
        int pos = -1;
        std::vector<char> iterator_names = {'i', 'j', 'k', 'a'};
        for (int i = 0; i < origin_shape.size(); ++i) {
          pos = target_str.find("9", pos + 1);
          if (index_length1[i] == 1) {
            target_str[pos] = '0';
            target_str[target_str.rfind(iterator_names[i])] = '0';
          }
        }

        LOG(INFO) << "Target Expr string:";
        LOG(INFO) << "\n" << target_str;
        ASSERT_EQ(expr_str, target_str);
      }
    } while (std::next_permutation(index_length1.begin(), index_length1.end()));
  }
}

}  // namespace poly
}  // namespace cinn

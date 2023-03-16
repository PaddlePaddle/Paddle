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

#include "Pass/AnalysisManager.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"

namespace {

struct Analysis1 {
  explicit Analysis1(mlir::Operation*) {}
};

struct Analysis2 {
  explicit Analysis2(mlir::Operation*) {}
};

TEST(AnalysisManagerTest, Preservation) {
  mlir::MLIRContext context;

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  infra::AnalysisManagerHolder amh(*module, nullptr);
  infra::AnalysisManager am = amh;

  am.GetAnalysis<Analysis1>();
  am.GetAnalysis<Analysis2>();

  infra::detail::PreservedAnalyses pa;
  pa.Preserve<Analysis1>();
  am.Invalidate(pa);

  EXPECT_TRUE(am.GetCachedAnalysis<Analysis1>().has_value());
  EXPECT_FALSE(am.GetCachedAnalysis<Analysis2>().has_value());
}

// Test analyses with custom invalidation logic.
struct AnalysisWithCustomInvalidatation {
  explicit AnalysisWithCustomInvalidatation(mlir::Operation*) {}
  bool IsInvalidated(const infra::AnalysisManager::PreservedAnalyses& pa) {
    return !pa.IsPreserved<Analysis1>();
  }
};
TEST(AnalysisManagerTest, CustomInvalidation) {
  mlir::MLIRContext context;

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  infra::AnalysisManagerHolder amh(*module, nullptr);
  infra::AnalysisManager am = amh;

  infra::detail::PreservedAnalyses pa;

  am.GetAnalysis<AnalysisWithCustomInvalidatation>();
  am.Invalidate(pa);
  EXPECT_FALSE(
      am.GetCachedAnalysis<AnalysisWithCustomInvalidatation>().has_value());

  am.GetAnalysis<AnalysisWithCustomInvalidatation>();
  pa.Preserve<Analysis1>();
  am.Invalidate(pa);
  EXPECT_TRUE(
      am.GetCachedAnalysis<AnalysisWithCustomInvalidatation>().has_value());
}

/// Test analyses with dependency
struct AnalysisWithDeps {
  explicit AnalysisWithDeps(mlir::Operation*,
                            infra::AnalysisManager& am) {  // NOLINT
    am.GetAnalysis<Analysis1>();
  }

  bool IsInvalidated(const infra::AnalysisManager::PreservedAnalyses& pa) {
    return !pa.IsPreserved<AnalysisWithDeps>() || !pa.IsPreserved<Analysis1>();
  }
};
TEST(AnalysisManagerTest, DependentAnalysis) {
  mlir::MLIRContext context;

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  infra::AnalysisManagerHolder amh(*module, nullptr);
  infra::AnalysisManager am = amh;

  am.GetAnalysis<AnalysisWithDeps>();
  EXPECT_TRUE(am.GetCachedAnalysis<AnalysisWithDeps>().has_value());
  EXPECT_TRUE(am.GetCachedAnalysis<Analysis1>().has_value());

  infra::detail::PreservedAnalyses pa;
  pa.Preserve<AnalysisWithDeps>();
  am.Invalidate(pa);

  EXPECT_FALSE(am.GetCachedAnalysis<AnalysisWithDeps>().has_value());
  EXPECT_FALSE(am.GetCachedAnalysis<Analysis1>().has_value());
}

// Test with 2 ctors.
struct AnalysisWith2Ctors {
  explicit AnalysisWith2Ctors(mlir::Operation*) { ctor1 = true; }
  explicit AnalysisWith2Ctors(mlir::Operation*,
                              infra::AnalysisManager& am) {  // NOLINT
    ctor2 = true;
  }
  bool ctor1 = false;
  bool ctor2 = false;
};
TEST(AnalysisManagerTest, AnalysisWith2Ctors) {
  mlir::MLIRContext context;

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  infra::AnalysisManagerHolder amh(*module, nullptr);
  infra::AnalysisManager am = amh;

  auto& an = am.GetAnalysis<AnalysisWith2Ctors>();
  EXPECT_TRUE(an.ctor2);
  EXPECT_FALSE(an.ctor1);
}

}  // namespace

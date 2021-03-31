// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/layer_norm_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_test_util.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {

class LayerNormFuseTest {
 public:
  LayerNormFuseTest()
      : m_prog{test::BuildProgramDesc(
            {"x", "x_mean_out", "x_sub_mean_out", "x_sub_mean_sqr_out",
             "std_dev_out", "std_dev_eps_out", "std_dev_eps_sqrt_out",
             "division_out", "scale_out", "shift_out"},
            {"sqr_pow", "eps", "gamma", "beta"})},
        m_place{},
        m_exe{m_place} {
    const BlockDesc& block_desc = m_prog.Block(0);
    auto* x_var_desc = block_desc.FindVar("x");
    x_var_desc->SetDataType(proto::VarType::FP32);
    x_var_desc->SetShape({3, 32, 48});

    auto* eps_var_desc = block_desc.FindVar("eps");
    eps_var_desc->SetDataType(proto::VarType::FP32);
    eps_var_desc->SetShape({1});

    auto* gamma_var_desc = block_desc.FindVar("gamma");
    gamma_var_desc->SetDataType(proto::VarType::FP32);
    gamma_var_desc->SetShape({48});

    auto* beta_var_desc = block_desc.FindVar("beta");
    beta_var_desc->SetDataType(proto::VarType::FP32);
    beta_var_desc->SetShape({48});

    auto* x_mean = test::CreateOp(&m_prog, "reduce_mean", {{"X", "x"}},
                                  {{"Out", "x_mean_out"}}, false);
    x_mean->SetAttr("dim", std::vector<int>{-1});
    x_mean->SetAttr("keep_dim", true);
    x_mean->SetAttr("reduce_all", false);

    test::CreateOp(&m_prog, "elementwise_sub",
                   {{"X", "x"}, {"Y", "x_mean_out"}},
                   {{"Out", "x_sub_mean_out"}}, false);
    test::CreateOp(&m_prog, "elementwise_pow",
                   {{"X", "x_sub_mean_out"}, {"Y", "sqr_pow"}},
                   {{"Out", "x_sub_mean_sqr_out"}}, false);
    auto* std_dev =
        test::CreateOp(&m_prog, "reduce_mean", {{"X", "x_sub_mean_sqr_out"}},
                       {{"Out", "std_dev_out"}}, false);
    std_dev->SetAttr("dim", std::vector<int>{-1});
    std_dev->SetAttr("keep_dim", true);
    std_dev->SetAttr("reduce_all", false);

    test::CreateOp(&m_prog, "elementwise_add",
                   {{"X", "std_dev_out"}, {"Y", "eps"}},
                   {{"Out", "std_dev_eps_out"}}, false);
    test::CreateOp(&m_prog, "sqrt", {{"X", "std_dev_eps_out"}},
                   {{"Out", "std_dev_eps_sqrt_out"}}, false);
    test::CreateOp(&m_prog, "elementwise_div",
                   {{"X", "x_sub_mean_out"}, {"Y", "std_dev_eps_sqrt_out"}},
                   {{"Out", "division_out"}}, false);
    test::CreateOp(&m_prog, "elementwise_mul",
                   {{"X", "division_out"}, {"Y", "gamma"}},
                   {{"Out", "scale_out"}}, false);
    test::CreateOp(&m_prog, "elementwise_add",
                   {{"X", "scale_out"}, {"Y", "beta"}}, {{"Out", "shift_out"}},
                   false);
  }

  template <typename Func>
  LayerNormFuseTest(const Func& func, int removed_nodes = 0,
                    int added_nodes = 0)
      : LayerNormFuseTest() {
    m_removed_nodes = removed_nodes;
    m_added_nodes = added_nodes;
    func(m_prog.Block(0));
  }

  void setupGraph() {
    auto initFun = [this](const Scope& scope,
                          const paddle::platform::CPUPlace& place) {
      this->initEpsTensorValue(scope, place);
    };
    setupGraphWithInitFunc(initFun);
  }

  template <typename Func>
  void setupGraphWithInitFunc(const Func& func) {
    m_graph.reset(new Graph(m_prog));
    // Init scope, as it is used in pass
    m_exe.CreateVariables(m_prog, 0, true, &m_scope);
    func(m_scope, m_place);
    m_graph->SetNotOwned(kParamScopeAttr, &m_scope);
  }

  void run(bool fusion = false) const {
    EXPECT_TRUE(test::RunPassAndAssert(m_graph.get(), "layer_norm_fuse_pass",
                                       "x", "shift_out", m_removed_nodes,
                                       m_added_nodes));
    EXPECT_TRUE(CheckSubgraphOpsCount(*m_graph, fusion));
  }

  const ProgramDesc& getProgramDesc() const { return m_prog; }
  const Graph* getGraph() const { return m_graph.get(); }

 private:
  void initEpsTensorValue(const Scope& scope,
                          const paddle::platform::CPUPlace& place) {
    float eps_value = 1e-5;
    test::InitLoDTensorHolder<float>(scope, place, "eps", {1}, &eps_value);
  }

  bool CheckSubgraphOpsCount(const Graph& graph, bool fusion) const {
    if (fusion)
      return test::AssertOpsCount(graph, {{"reduce_mean", 0},
                                          {"elementwise_sub", 0},
                                          {"elementwise_pow", 0},
                                          {"elementwise_add", 0},
                                          {"sqrt", 0},
                                          {"elementwise_div", 0},
                                          {"elementwise_mul", 0},
                                          {"layer_norm", 1}});
    else
      return test::AssertOpsCount(graph, {{"reduce_mean", 2},
                                          {"elementwise_sub", 1},
                                          {"elementwise_pow", 1},
                                          {"elementwise_add", 2},
                                          {"sqrt", 1},
                                          {"elementwise_div", 1},
                                          {"elementwise_mul", 1},
                                          {"layer_norm", 0}});
  }

  int m_removed_nodes{19};
  int m_added_nodes{3};
  ProgramDesc m_prog;
  paddle::platform::CPUPlace m_place;
  NaiveExecutor m_exe;
  Scope m_scope;
  std::unique_ptr<Graph> m_graph{nullptr};
};

}  // namespace

// ------------------------------ Test cases -----------------------------------

TEST(FuseLayerNormPass, TestFuse) {
  LayerNormFuseTest lnorm_test;
  lnorm_test.setupGraph();
  lnorm_test.run(true);

  // additional attribute checks
  for (const auto* node : lnorm_test.getGraph()->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "layer_norm") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("is_test"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("is_test")));
      ASSERT_TRUE(op->HasAttr("begin_norm_axis"));
      ASSERT_TRUE(op->HasAttr("epsilon"));
    }
  }
}

TEST(FuseLayerNormPass, TestInvalidEpsNumel) {
  const auto editEpsFun = [](const BlockDesc& block_desc) {
    auto* eps_var_desc = block_desc.FindVar("eps");
    eps_var_desc->SetDataType(proto::VarType::FP32);
    eps_var_desc->SetShape({2});
  };
  const auto initEpsTensor = [](const Scope& scope,
                                const paddle::platform::CPUPlace& place) {
    auto eps_values = std::vector<float>{1e-5f, 1e-5f};
    test::InitLoDTensorHolder<float>(scope, place, "eps", {2},
                                     eps_values.data());
  };

  LayerNormFuseTest lnorm_test(editEpsFun);
  lnorm_test.setupGraphWithInitFunc(initEpsTensor);
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, TestInvalidEpsDataType) {
  const auto editEpsFun = [](const BlockDesc& block_desc) {
    auto* eps_var_desc = block_desc.FindVar("eps");
    eps_var_desc->SetDataType(proto::VarType::FP64);
    eps_var_desc->SetShape({1});
  };
  const auto initEpsTensor = [](const Scope& scope,
                                const paddle::platform::CPUPlace& place) {
    double eps_value = 1e-5;
    test::InitLoDTensorHolder<double>(scope, place, "eps", {1}, &eps_value);
  };

  LayerNormFuseTest lnorm_test(editEpsFun);
  lnorm_test.setupGraphWithInitFunc(initEpsTensor);
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, TestInvalidGammaRank) {
  const auto editGammaFun = [](const BlockDesc& block_desc) {
    auto* gamma_var_desc = block_desc.FindVar("gamma");
    gamma_var_desc->SetDataType(proto::VarType::FP32);
    gamma_var_desc->SetShape({48, 32});
  };

  LayerNormFuseTest lnorm_test(editGammaFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, TestInvalidBetaRank) {
  const auto editBetaFun = [](const BlockDesc& block_desc) {
    auto* beta_var_desc = block_desc.FindVar("beta");
    beta_var_desc->SetDataType(proto::VarType::FP32);
    beta_var_desc->SetShape({48, 32});
  };

  LayerNormFuseTest lnorm_test(editBetaFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, TestUnequalGammaBetaShapes) {
  const auto editGammaBetaFun = [](const BlockDesc& block_desc) {
    auto* beta_var_desc = block_desc.FindVar("beta");
    beta_var_desc->SetDataType(proto::VarType::FP32);
    beta_var_desc->SetShape({32});
  };

  LayerNormFuseTest lnorm_test(editGammaBetaFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, TestGammaBetaUnequalInputChannelShape) {
  const auto editGammaBetaFun = [](const BlockDesc& block_desc) {
    auto* beta_var_desc = block_desc.FindVar("beta");
    beta_var_desc->SetDataType(proto::VarType::FP32);
    beta_var_desc->SetShape({32});

    auto* gamma_var_desc = block_desc.FindVar("gamma");
    gamma_var_desc->SetDataType(proto::VarType::FP32);
    gamma_var_desc->SetShape({32});
  };

  LayerNormFuseTest lnorm_test(editGammaBetaFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadInMeanDimAttrRank) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* x_mean_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "x_mean_out");
    ASSERT_NE(x_mean_desc, nullptr);
    x_mean_desc->SetAttr("dim", std::vector<int>{1, 1});
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadInMeanDimAttr) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* x_mean_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "x_mean_out");
    ASSERT_NE(x_mean_desc, nullptr);
    x_mean_desc->SetAttr("dim", std::vector<int>{1});
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadInMeanKeepDimAttr) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* x_mean_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "x_mean_out");
    ASSERT_NE(x_mean_desc, nullptr);
    x_mean_desc->SetAttr("keep_dim", false);
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadInMeanReduceAllAttr) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* x_mean_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "x_mean_out");
    ASSERT_NE(x_mean_desc, nullptr);
    x_mean_desc->SetAttr("reduce_all", true);
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadStdDevMeanDimAttrRank) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* std_dev_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "std_dev_out");
    ASSERT_NE(std_dev_desc, nullptr);
    std_dev_desc->SetAttr("dim", std::vector<int>{1, 1});
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadStdDevMeanDimAttr) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* std_dev_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "std_dev_out");
    ASSERT_NE(std_dev_desc, nullptr);
    std_dev_desc->SetAttr("dim", std::vector<int>{1});
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadStdDevMeanKeepDimAttr) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* std_dev_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "std_dev_out");
    ASSERT_NE(std_dev_desc, nullptr);
    std_dev_desc->SetAttr("keep_dim", false);
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, NoFusionBadStdDevMeanReduceAllAttr) {
  const auto editFun = [](const BlockDesc& block_desc) {
    auto* std_dev_desc =
        test::GetOp(block_desc, "reduce_mean", "Out", "std_dev_out");
    ASSERT_NE(std_dev_desc, nullptr);
    std_dev_desc->SetAttr("reduce_all", true);
  };

  LayerNormFuseTest lnorm_test(editFun);
  lnorm_test.setupGraph();
  lnorm_test.run(false);
}

TEST(FuseLayerNormPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("layer_norm_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(layer_norm_fuse_pass);

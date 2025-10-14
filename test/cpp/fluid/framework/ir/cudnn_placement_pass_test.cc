// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/cudnn_placement_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle::framework::ir {

class PlacementPassTest {
 private:
  void RegisterOpKernel() {
    static bool is_registered = false;
    if (!is_registered) {
      auto& all_kernels = OperatorWithKernel::AllOpKernels();

      phi::GPUPlace place = phi::GPUPlace(0);
      OpKernelType plain_kernel_type = OpKernelType(proto::VarType::FP32,
                                                    place,
                                                    DataLayout::kAnyLayout,
                                                    LibraryType::kPlain);
      OpKernelType cudnn_kernel_type = OpKernelType(proto::VarType::FP32,
                                                    place,
                                                    DataLayout::kAnyLayout,
                                                    LibraryType::kCUDNN);

      auto fake_kernel_func = [](const ExecutionContext&) -> void {
        static int num_calls = 0;
        num_calls++;
      };

      all_kernels["conv2d"][cudnn_kernel_type] = fake_kernel_func;
      all_kernels["pool2d"][cudnn_kernel_type] = fake_kernel_func;
      all_kernels["depthwise_conv2d"][plain_kernel_type] = fake_kernel_func;
      all_kernels["relu"][plain_kernel_type] = fake_kernel_func;

      is_registered = true;
    }
  }

 public:
  void MainTest(std::initializer_list<std::string> cudnn_enabled_op_types,
                unsigned expected_use_cudnn_true_count) {
    // operator                                 use_cudnn
    // --------------------------------------------------
    // (a,b)->concat->c                         -
    // (c,weights,bias)->conv2d->f              false
    // f->relu->g                               -
    // g->pool2d->h                             false
    // (h,weights2,bias2)->depthwise_conv2d->k  false
    // k->relu->l                               -
    Layers layers;
    VarDesc* a = layers.data("a");
    VarDesc* b = layers.data("b");
    VarDesc* c = layers.concat(std::vector<VarDesc*>({a, b}));
    VarDesc* weights_0 = layers.data("weights_0");
    VarDesc* bias_0 = layers.data("bias_0");
    VarDesc* f = layers.conv2d(c, weights_0, bias_0, false);
    VarDesc* g = layers.relu(f);
    VarDesc* h = layers.pool2d(g, false);
    VarDesc* weights_1 = layers.data("weights_1");
    VarDesc* bias_1 = layers.data("bias_1");
    VarDesc* k = layers.depthwise_conv2d(h, weights_1, bias_1, false);
    layers.relu(k);

    RegisterOpKernel();

    std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
    auto pass = PassRegistry::Instance().Get("cudnn_placement_pass");
    pass->Set("cudnn_enabled_op_types",
              new std::unordered_set<std::string>(cudnn_enabled_op_types));

    graph.reset(pass->Apply(graph.release()));

    unsigned use_cudnn_true_count = 0;
    for (auto* node : graph->Nodes()) {
      if (node->IsOp() && node->Op()) {
        auto* op = node->Op();
        if (op->HasAttr("use_cudnn") &&
            PADDLE_GET_CONST(bool, op->GetAttr("use_cudnn"))) {
          ++use_cudnn_true_count;
        }
      }
    }

    EXPECT_EQ(use_cudnn_true_count, expected_use_cudnn_true_count);
  }

  void PlacementNameTest() {
    auto pass = PassRegistry::Instance().Get("cudnn_placement_pass");
    EXPECT_EQ(static_cast<PlacementPassBase*>(pass.get())->GetPlacementName(),
              "cuDNN");
  }
};

TEST(CUDNNPlacementPass, enable_conv2d) {
  // 1 conv2d
  PlacementPassTest().MainTest({"conv2d"}, 1);
}

TEST(CUDNNPlacementPass, enable_relu_pool) {
  // 1 conv2d + 1 pool2d
  PlacementPassTest().MainTest({"conv2d", "pool2d"}, 2);
}

TEST(CUDNNPlacementPass, enable_all) {
  // 1 conv2d + 1 pool2d
  // depthwise_conv2d does not have CUDNN kernel.
  PlacementPassTest().MainTest({}, 2);
}

TEST(CUDNNPlacementPass, placement_name) {
  PlacementPassTest().PlacementNameTest();
}

}  // namespace paddle::framework::ir

USE_PASS(cudnn_placement_pass);

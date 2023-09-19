/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/delete_dropout_op_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(DeleteDropoutOpsPass, dropout) {
  for (std::string dropout_implementation :
       {"downgrade_in_infer", "upscale_in_train"}) {
    for (auto inplace : {false, true}) {
      if (dropout_implementation == "downgrade_in_infer" && inplace == true) {
        continue;
      }

      LOG(INFO) << "dropout_implementation: " << dropout_implementation
                << ", inplace: " << inplace;
      Layers layers;
      // (x, y) -> mul -> tmp_0
      // (tmp_0) -> dropout -> (tmp_1)
      // (tmp_1, z) -> elementwise_add -> (tmp_2)
      // or
      // (tmp_1, z) -> elementwise_add -> (tmp_0)
      auto* x = layers.data("x");
      auto* y = layers.data("y");
      auto* z = layers.data("z");
      auto* mul_out = layers.mul(x, y);
      auto* dropout_out = layers.dropout(mul_out, 0.5f, dropout_implementation);
      if (inplace) {
        layers.elementwise_add(dropout_out, z, mul_out);
      } else {
        layers.elementwise_add(dropout_out, z);
      }

      std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
      auto pass = PassRegistry::Instance().Get("delete_dropout_op_x_pass");
      int num_dropout_nodes_before = GetNumOpNodes(graph, "dropout");
      int num_scale_nodes_before = GetNumOpNodes(graph, "scale");
      VLOG(3) << DebugString(graph);

      graph.reset(pass->Apply(graph.release()));
      int num_dropout_nodes_after = GetNumOpNodes(graph, "dropout");
      int num_scale_nodes_after = GetNumOpNodes(graph, "scale");

      VLOG(3) << DebugString(graph);

      PADDLE_ENFORCE_EQ(
          num_dropout_nodes_after,
          0,
          platform::errors::InvalidArgument("num_dropout_nodes_after = %d.",
                                            num_dropout_nodes_after));

      if (dropout_implementation == "downgrade_in_infer") {
        PADDLE_ENFORCE_EQ(
            num_dropout_nodes_before,
            num_scale_nodes_after - num_scale_nodes_before,
            platform::errors::InvalidArgument(
                "num_dropout_nodes_before = %d, num_scale_nodes_after = %d, "
                "num_scale_nodes_before = %d.",
                num_dropout_nodes_before,
                num_scale_nodes_after,
                num_scale_nodes_before));
      } else {
        PADDLE_ENFORCE_EQ(
            num_scale_nodes_after - num_scale_nodes_before,
            0,
            platform::errors::InvalidArgument(
                "num_scale_nodes_after = %d, num_scale_nodes_before = %d.",
                num_scale_nodes_after,
                num_scale_nodes_before));
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(delete_dropout_op_x_pass);

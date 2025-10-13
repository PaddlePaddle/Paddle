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

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

#define APPLY_PASS                                                        \
  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program())); \
  auto pass = PassRegistry::Instance().Get("fast_where_xpu_fuse_pass");   \
  pass->Apply(graph.get());

#define VERIFY_GRAPH(x, y)                                                    \
  auto num_op_nodes = GetNumOpNodes(graph);                                   \
  PADDLE_ENFORCE_EQ(                                                          \
      num_op_nodes,                                                           \
      1,                                                                      \
      common::errors::PreconditionNotMet(                                     \
          "The graph contains only one op node, but %d op nodes found.",      \
          num_op_nodes));                                                     \
  auto fast_where_xpu_op_nodes = GetOpNodes(graph, "fast_where_xpu");         \
  PADDLE_ENFORCE_EQ(fast_where_xpu_op_nodes.size(),                           \
                    1,                                                        \
                    common::errors::PreconditionNotMet(                       \
                        "The graph contains only a fast_where_xpu op node, "  \
                        "but %d op nodes found.",                             \
                        fast_where_xpu_op_nodes.size()));                     \
  const auto& x_name = fast_where_xpu_op_nodes[0]->Op()->Input("x")[0];       \
  PADDLE_ENFORCE_EQ(x_name,                                                   \
                    #x,                                                       \
                    common::errors::PreconditionNotMet(                       \
                        "The input 'x' of fast_where_xpu op should be '%s', " \
                        "but receive '%s'.",                                  \
                        #x,                                                   \
                        x_name));                                             \
  const auto& y_name = fast_where_xpu_op_nodes[0]->Op()->Input("y")[0];       \
  PADDLE_ENFORCE_EQ(y_name,                                                   \
                    #y,                                                       \
                    common::errors::PreconditionNotMet(                       \
                        "The input 'y' of fast_where_xpu op should be '%s', " \
                        "but receive '%s'.",                                  \
                        #y,                                                   \
                        y_name));

TEST(FastWhereXPUFusePass, one_case0) {
  Layers layers;
  auto* condition =
      layers.data("condition", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  auto* cast_out = layers.cast(condition, 0, 5);
  cast_out->SetShape({20, 1});
  auto* scale_out = layers.scale(cast_out, -1.0f, 1.0f, true);
  scale_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(x, scale_out);
  mul0_out->SetShape({20, 7});
  auto* mul1_out = layers.elementwise_mul(y, cast_out);
  mul1_out->SetShape({20, 7});
  auto* add_out = layers.elementwise_add(mul0_out, mul1_out);
  add_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(y, x)
}

TEST(FastWhereXPUFusePass, one_case1) {
  Layers layers;
  auto* condition =
      layers.data("condition", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  auto* cast_out = layers.cast(condition, 0, 5);
  cast_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(x, cast_out);
  mul0_out->SetShape({20, 7});
  auto* scale_out = layers.scale(cast_out, -1.0f, 1.0f, true);
  scale_out->SetShape({20, 1});
  auto* mul1_out = layers.elementwise_mul(y, scale_out);
  mul1_out->SetShape({20, 7});
  auto* add_out = layers.elementwise_add(mul0_out, mul1_out);
  add_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(x, y)
}

TEST(FastWhereXPUFusePass, one_case2) {
  Layers layers;
  auto* condition =
      layers.data("condition", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  auto* cast_out = layers.cast(condition, 0, 5);
  cast_out->SetShape({20, 1});
  auto* scale_out = layers.scale(cast_out, -1.0f, 1.0f, true);
  scale_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(scale_out, x);
  mul0_out->SetShape({20, 7});
  auto* mul1_out = layers.elementwise_mul(cast_out, y);
  mul1_out->SetShape({20, 7});
  auto* add_out = layers.elementwise_add(mul0_out, mul1_out);
  add_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(y, x)
}

TEST(FastWhereXPUFusePass, one_case3) {
  Layers layers;
  auto* condition =
      layers.data("condition", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  auto* cast_out = layers.cast(condition, 0, 5);
  cast_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(cast_out, x);
  mul0_out->SetShape({20, 7});
  auto* scale_out = layers.scale(cast_out, -1.0f, 1.0f, true);
  scale_out->SetShape({20, 1});
  auto* mul1_out = layers.elementwise_mul(scale_out, y);
  mul1_out->SetShape({20, 7});
  auto* add_out = layers.elementwise_add(mul0_out, mul1_out);
  add_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(x, y)
}

TEST(FastWhereXPUFusePass, one_case4) {
  Layers layers;
  auto* condition =
      layers.data("condition", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  auto* cast_out = layers.cast(condition, 0, 5);
  cast_out->SetShape({20, 1});
  auto* scale_out = layers.scale(cast_out, -1.0f, 1.0f, true);
  scale_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(scale_out, x);
  mul0_out->SetShape({20, 7});
  auto* mul1_out = layers.elementwise_mul(y, cast_out);
  mul1_out->SetShape({20, 7});
  auto* add_out = layers.elementwise_add(mul0_out, mul1_out);
  add_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(y, x)
}

TEST(FastWhereXPUFusePass, one_case5) {
  Layers layers;
  auto* condition =
      layers.data("condition", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  auto* cast_out = layers.cast(condition, 0, 5);
  cast_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(cast_out, x);
  mul0_out->SetShape({20, 7});
  auto* scale_out = layers.scale(cast_out, -1.0f, 1.0f, true);
  scale_out->SetShape({20, 1});
  auto* mul1_out = layers.elementwise_mul(y, scale_out);
  mul1_out->SetShape({20, 7});
  auto* add_out = layers.elementwise_add(mul0_out, mul1_out);
  add_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(x, y)
}

#undef VERIFY_GRAPH
#define VERIFY_GRAPH(logical_op, x, y)                                        \
  auto num_op_nodes = GetNumOpNodes(graph);                                   \
  PADDLE_ENFORCE_EQ(                                                          \
      num_op_nodes,                                                           \
      2,                                                                      \
      common::errors::PreconditionNotMet(                                     \
          "The graph contains only two op nodes, but %d op nodes found.",     \
          num_op_nodes));                                                     \
  auto logical_op_nodes = GetOpNodes(graph, #logical_op);                     \
  PADDLE_ENFORCE_EQ(                                                          \
      logical_op_nodes.size(),                                                \
      1,                                                                      \
      common::errors::PreconditionNotMet(                                     \
          "The graph contains only a '%s' op node, but %d op nodes found.",   \
          #logical_op,                                                        \
          logical_op_nodes.size()));                                          \
  auto fast_where_xpu_op_nodes = GetOpNodes(graph, "fast_where_xpu");         \
  PADDLE_ENFORCE_EQ(fast_where_xpu_op_nodes.size(),                           \
                    1,                                                        \
                    common::errors::PreconditionNotMet(                       \
                        "The graph contains only a fast_where_xpu op node, "  \
                        "but %d op nodes found.",                             \
                        fast_where_xpu_op_nodes.size()));                     \
  const auto& x_name = fast_where_xpu_op_nodes[0]->Op()->Input("x")[0];       \
  PADDLE_ENFORCE_EQ(x_name,                                                   \
                    #x,                                                       \
                    common::errors::PreconditionNotMet(                       \
                        "The input 'x' of fast_where_xpu op should be '%s', " \
                        "but receive '%s'.",                                  \
                        #x,                                                   \
                        x_name));                                             \
  const auto& y_name = fast_where_xpu_op_nodes[0]->Op()->Input("y")[0];       \
  PADDLE_ENFORCE_EQ(y_name,                                                   \
                    #y,                                                       \
                    common::errors::PreconditionNotMet(                       \
                        "The input 'y' of fast_where_xpu op should be '%s', " \
                        "but receive '%s'.",                                  \
                        #y,                                                   \
                        y_name));

TEST(FastWhereXPUFusePass, cascade_case0) {
  Layers layers;
  auto* condition0 =
      layers.data("condition0", {20, 1}, false, proto::VarType::BOOL);
  auto* condition1 =
      layers.data("condition1", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  // fast_where_xpu0
  auto* cast0_out = layers.cast(condition0, 0, 5);
  cast0_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(cast0_out, x);
  mul0_out->SetShape({20, 7});
  auto* scale0_out = layers.scale(cast0_out, -1.0f, 1.0f, true);
  scale0_out->SetShape({20, 1});
  auto* mul1_out = layers.elementwise_mul(scale0_out, y);
  mul1_out->SetShape({20, 7});
  auto* add0_out = layers.elementwise_add(mul0_out, mul1_out);
  add0_out->SetShape({20, 7});
  // fast_where_xpu1
  auto* cast1_out = layers.cast(condition1, 0, 5);
  cast1_out->SetShape({20, 1});
  auto* mul2_out = layers.elementwise_mul(cast1_out, x);
  mul2_out->SetShape({20, 7});
  auto* scale1_out = layers.scale(cast1_out, -1.0f, 1.0f, true);
  scale1_out->SetShape({20, 1});
  auto* mul3_out = layers.elementwise_mul(scale1_out, add0_out);
  mul3_out->SetShape({20, 7});
  auto* add1_out = layers.elementwise_add(mul2_out, mul3_out);
  add1_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(logical_or, x, y)
}

TEST(FastWhereXPUFusePass, cascade_case1) {
  Layers layers;
  auto* condition0 =
      layers.data("condition0", {20, 1}, false, proto::VarType::BOOL);
  auto* condition1 =
      layers.data("condition1", {20, 1}, false, proto::VarType::BOOL);
  auto* x = layers.data("x", {20, 7});
  auto* y = layers.data("y", {20, 7});
  // fast_where_xpu0
  auto* cast0_out = layers.cast(condition0, 0, 5);
  cast0_out->SetShape({20, 1});
  auto* mul0_out = layers.elementwise_mul(cast0_out, x);
  mul0_out->SetShape({20, 7});
  auto* scale0_out = layers.scale(cast0_out, -1.0f, 1.0f, true);
  scale0_out->SetShape({20, 1});
  auto* mul1_out = layers.elementwise_mul(scale0_out, y);
  mul1_out->SetShape({20, 7});
  auto* add0_out = layers.elementwise_add(mul0_out, mul1_out);
  add0_out->SetShape({20, 7});
  // fast_where_xpu1
  auto* cast1_out = layers.cast(condition1, 0, 5);
  cast1_out->SetShape({20, 1});
  auto* mul2_out = layers.elementwise_mul(cast1_out, add0_out);
  mul2_out->SetShape({20, 7});
  auto* scale1_out = layers.scale(cast1_out, -1.0f, 1.0f, true);
  scale1_out->SetShape({20, 1});
  auto* mul3_out = layers.elementwise_mul(scale1_out, y);
  mul3_out->SetShape({20, 7});
  auto* add1_out = layers.elementwise_add(mul2_out, mul3_out);
  add1_out->SetShape({20, 7});

  APPLY_PASS
  VERIFY_GRAPH(logical_and, x, y)
}

#undef APPLY_PASS
#undef VERIFY_GRAPH

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fast_where_xpu_fuse_pass);

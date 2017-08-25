/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/elemwise_op.h"
#include "gtest/gtest.h"

// Begin declare eigen unary operator
#define DIV_2(x) (x / static_cast<T>(2.0))
DEFINE_EIGEN_UNARY_FUNCTOR(Div2Functor, DIV_2);
REGISTER_UNARY_OP_AND_EIGEN_CPU_KERNEL(div2, Div2Functor, float,
                                       R"DOC(Div2 operator

Out = x/2
)DOC",
                                       div_grad, paddle::framework::NOP);

// End declare eigen unary operator

TEST(ElemwiseOp, div2) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  auto div2 =
      OpRegistry::CreateOp("div2", {{"X", {"X"}}}, {{"Out", {"Out"}}}, {});
  Scope scope;
  auto* x_var = scope.NewVar("X");
  auto* x = x_var->GetMutable<Tensor>();
  x->Resize({2, 2});
  float* d = x->mutable_data<float>(CPUPlace());

  for (size_t i = 0; i < 4; ++i) {
    d[i] = static_cast<float>(i + 1);
  }

  auto* o_var = scope.NewVar("Out");
  auto* o = o_var->GetMutable<Tensor>();

  div2->InferShape(scope);

  CPUDeviceContext ctx;
  div2->Run(scope, ctx);
  auto* o_buf = o->data<float>();
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_NEAR(o_buf[i], static_cast<float>(i + 1) / 2, 1e-5);
  }
}
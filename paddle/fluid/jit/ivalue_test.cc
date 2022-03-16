// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/jit/ivalue.h"

#include "gtest/gtest.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace jit {
using Tensor = paddle::experimental::Tensor;
using DenseTensor = phi::DenseTensor;

#define eps 1e-8

// Test memeber function
TEST(IValue, Basic) {
  int int_v = 10;
  IValue iv_i(int_v);
  EXPECT_EQ(iv_i.AsInt(), int_v);

  double double_v = 0.1;
  IValue iv_d(double_v);
  EXPECT_LT(iv_d.AsDouble() - double_v, eps);
  EXPECT_LT(double_v - iv_d.AsDouble(), eps);

  bool bool_v = false;
  IValue iv_b(bool_v);
  EXPECT_EQ(iv_b.AsBool(), bool_v);

  auto temp = std::make_shared<DenseTensor>();
  temp->Resize(phi::make_ddim({2, 4}));
  phi::CPUContext cpu_ctx;
  cpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::CPUPlace())
                           .get());
  cpu_ctx.Init();
  cpu_ctx.Alloc<float>(temp.get());
  Tensor t(temp);

  IValue iv_t(t);
  auto inner_t = iv_t.AsTensor();
  EXPECT_EQ(inner_t.numel(), 8);
}

// TEST(IValue, Serializer){
// }

}  // namespace jit
}  // namespace paddle

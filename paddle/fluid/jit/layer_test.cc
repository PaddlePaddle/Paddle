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

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include "gtest/gtest.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(matmul_v2);
USE_OP_ITSELF(relu);
USE_OP_ITSELF(reduce_mean);
USE_OP_ITSELF(feed);
USE_OP_ITSELF(fetch);
USE_OP_ITSELF(scale);

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);

namespace paddle {
namespace jit {
using Tensor = paddle::experimental::Tensor;

std::vector<IValue> PrepareInputs() {
  auto temp = std::make_shared<phi::DenseTensor>();
  temp->Resize(phi::make_ddim({2, 4}));
  phi::CPUContext cpu_ctx;
  cpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::CPUPlace())
                           .get());
  cpu_ctx.Init();
  cpu_ctx.Alloc<float>(temp.get());
  phi::funcs::set_constant(cpu_ctx, temp.get(), 2.);
  Tensor t(temp);
  // TODO(dev): associate the input name
  t.set_name("x");
  IValue iv_t(t);
  return {iv_t};
}

void Print(const IValue &ival) {
  auto t = ival.AsTensor();
  auto dt = std::dynamic_pointer_cast<phi::DenseTensor>(t.impl());
  auto *data = dt->data<float>();
  VLOG(3) << "------------------------";
  for (int i = 0; i < dt->numel(); ++i) {
    VLOG(3) << "print data: " << data[i];
  }
}

TEST(layer, Construct) {
  std::string path = "./";
  auto layer = jit::Load(path);
  auto inputs = PrepareInputs();
  auto outs = layer.forward(inputs);
  for (auto &out : outs) {
    Print(out);
  }
  auto func = layer.GetFunction("infer");
  auto outputs = (*func)(inputs);
  for (auto &out : outputs) {
    Print(out);
  }
}

}  // namespace jit
}  // namespace paddle

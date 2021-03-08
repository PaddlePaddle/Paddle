/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <string>
#include <thread>  // NOLINT
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

using Tensor = paddle::framework::Tensor;

USE_OP(reduce_any);
USE_OP_DEVICE_KERNEL(reduce_any, NPU);

template <typename T>
void TensorFromVector(const std::vector<int>& src,
                      const p::DeviceContext& ctx, Tensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  p::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  paddle::memory::Copy(
      BOOST_GET_CONST(p::NPUPlace, dst_place), dst_ptr, src_place,
      src_ptr, size,
      reinterpret_cast<const p::NPUDeviceContext&>(ctx).stream());
}

template <typename T>
void TensorToVector(const Tensor& src, const p::DeviceContext& ctx,
                    std::vector<int>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  p::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  paddle::memory::Copy(
      dst_place, dst_ptr, BOOST_GET_CONST(p::NPUPlace, src.place()),
      src_ptr, size,
      reinterpret_cast<const p::NPUDeviceContext&>(ctx).stream());
}

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();
  std::vector<int> init_x = {1, 0, 0, 0};
  TensorFromVector<bool>(init_x, ctx, tensor_x);
  tensor_x->Resize(paddle::framework::make_ddim({2, 2}));
 
  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  f::AttributeMap attrs; //= {{"dim", {0}}};
  auto op = f::OpRegistry::CreateOp("reduce_any", {{"X", {"X"}}},
                                    {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);

  std::vector<int> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  std::vector<int> expected_vec = {1, 0, 0, 0};
  EXPECT_EQ(out_vec.size(), expected_vec.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], expected_vec[i]);
  }
}

TEST(reduce_any, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<bool>(&scope, ctx);
}

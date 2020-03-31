// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstdlib>
#include <memory>
#include <random>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

USE_OP(softmax);
USE_OP_DEVICE_KERNEL(softmax, MKLDNN);

namespace paddle {
namespace operators {

template <typename T>
bool TestMain(const platform::Place &place, const framework::DDim &dims) {
  framework::Scope scope;
  auto *x = scope.Var("x")->GetMutable<framework::LoDTensor>();
  auto *y = scope.Var("y")->GetMutable<framework::LoDTensor>();

  x->Resize(dims);
  y->Resize(dims);

  size_t numel = static_cast<size_t>(framework::product(dims));

  auto x_ptr = x->mutable_data<T>(place);
  auto y_ptr = y->mutable_data<T>(place);

  std::uniform_real_distribution<T> dist(static_cast<T>(10.0),
                                         static_cast<T>(20.0));
  std::mt19937 engine;

  for (size_t i = 0; i < numel; ++i) {
    x_ptr[i] = dist(engine);
    y_ptr[i] = static_cast<T>(0);
  }

  auto &pool = platform::DeviceContextPool::Instance();

  // Out of place (reference) computation
  auto op_ref = framework::OpRegistry::CreateOp(
      "softmax", {{"X", {"x"}}}, {{"Out", {"y"}}}, {{"use_mkldnn", {true}}});
  op_ref->Run(scope, place);
  pool.Get(place)->Wait();

  // Get reference (out of place) result
  auto &ref_tensor = scope.FindVar("y")->Get<framework::LoDTensor>();

  // In-place (to be tested) computation
  auto op = framework::OpRegistry::CreateOp(
      "softmax", {{"X", {"x"}}}, {{"Out", {"x"}}}, {{"use_mkldnn", {true}}});
  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  // Get in-place result
  auto &out_tensor = scope.FindVar("x")->Get<framework::LoDTensor>();
  PADDLE_ENFORCE_EQ(
      &out_tensor, x,
      platform::errors::InvalidArgument(
          "Input and output vars should share tensor for In-place test"));

  // compare results
  auto *ref_ptr = ref_tensor.data<T>();
  auto *out_ptr = out_tensor.data<T>();
  bool is_equal = std::equal(out_ptr, out_ptr + numel, ref_ptr);
  return is_equal;
}

TEST(test_softmax_inplace, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  ASSERT_TRUE(TestMain<float>(p, dims));
}

}  // namespace operators
}  // namespace paddle

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
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle {
namespace operators {

struct InputVars {
  std::string name;
  phi::DenseTensor *tensor;
};

void Test_Pool2d_Transpose_NHWC(const std::string &transpose_type) {
  phi::DDim dims({1, 4, 8, 512});           // NHWC shape
  phi::DDim expected_dims({1, 7, 512, 3});  // NHWC expected shape
  phi::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x", scope.Var("x")->GetMutable<phi::DenseTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(static_cast<float>(10.0),
                                             static_cast<float>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(common::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }

  scope.Var("y")->GetMutable<phi::DenseTensor>();
  auto *z = scope.Var("z")->GetMutable<phi::DenseTensor>();

  auto &pool = phi::DeviceContextPool::Instance();

  // Make pool2d followed by transpose

  auto ksize = std::vector<int>(2, 2);
  auto op_pool =
      framework::OpRegistry::CreateOp("pool2d",
                                      {{"X", {"x"}}},
                                      {{"Out", {"y"}}},
                                      {{"pooling_type", {std::string("max")}},
                                       {"ksize", {ksize}},
                                       {"data_format", {std::string("NHWC")}},
                                       {"use_mkldnn", {true}}});

  auto axis = std::vector<int>(4, 0);
  axis[1] = 2;
  axis[2] = 3;
  axis[3] = 1;
  auto op_transpose = framework::OpRegistry::CreateOp(
      transpose_type,
      {{"X", {"y"}}},
      {{"Out", {"z"}}},
      {{"axis", {axis}}, {"use_mkldnn", {true}}});

  op_pool->Run(scope, p);
  op_transpose->Run(scope, p);
  pool.Get(p)->Wait();

  // Verify shape of output
  PADDLE_ENFORCE_EQ(z->dims(),
                    expected_dims,
                    common::errors::InvalidArgument(
                        "Computed shape does not match expected shape"));
}

TEST(test_pool2d_transpose_nhwc, cpu_place) {
  Test_Pool2d_Transpose_NHWC({"transpose"});
  Test_Pool2d_Transpose_NHWC({"fused_transpose"});
}

TEST(test_pool2d_relu_relu_nhwc, cpu_place) {
  phi::DDim dims({1, 4, 8, 512});           // NHWC shape
  phi::DDim expected_dims({1, 512, 3, 7});  // NCHW expected shape
  phi::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x", scope.Var("x")->GetMutable<phi::DenseTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(static_cast<float>(10.0),
                                             static_cast<float>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(common::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }

  scope.Var("y")->GetMutable<phi::DenseTensor>();
  scope.Var("u")->GetMutable<phi::DenseTensor>();
  auto *z = scope.Var("z")->GetMutable<phi::DenseTensor>();

  auto &pool = phi::DeviceContextPool::Instance();

  // Make pool2d(oneDNN) followed by relu(CPU paddle) followed by
  // relu(oneDNN). Second relu should make a shape rotation to NCHW

  auto ksize = std::vector<int>(2, 2);
  auto op_pool =
      framework::OpRegistry::CreateOp("pool2d",
                                      {{"X", {"x"}}},
                                      {{"Out", {"y"}}},
                                      {{"pooling_type", {std::string("max")}},
                                       {"ksize", {ksize}},
                                       {"data_format", {std::string("NHWC")}},
                                       {"use_mkldnn", {true}}});

  auto axis = std::vector<int>(4, 0);
  axis[1] = 2;
  axis[2] = 3;
  axis[3] = 1;
  auto op_relu1 = framework::OpRegistry::CreateOp(
      "relu",
      {{"X", {"y"}}},
      {{"Out", {"u"}}},
      {{"axis", {axis}}, {"use_mkldnn", {false}}});

  auto op_relu2 = framework::OpRegistry::CreateOp(
      "relu", {{"X", {"u"}}}, {{"Out", {"z"}}}, {{"use_mkldnn", {true}}});

  op_pool->Run(scope, p);
  op_relu1->Run(scope, p);
  op_relu2->Run(scope, p);

  pool.Get(p)->Wait();

  // Verify shape of output
  PADDLE_ENFORCE_EQ(z->dims(),
                    expected_dims,
                    common::errors::InvalidArgument(
                        "Computed shape does not match expected shape"));
}

TEST(test_pool2d_shape_nhwc, cpu_place) {
  phi::DDim dims({1, 4, 8, 512});                    // NHWC shape
  std::vector<int32_t> expected_dims{1, 3, 7, 512};  // NHWC expected shape
  phi::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x", scope.Var("x")->GetMutable<phi::DenseTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(static_cast<float>(10.0),
                                             static_cast<float>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(common::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }

  scope.Var("y")->GetMutable<phi::DenseTensor>();
  auto *z = scope.Var("z")->GetMutable<phi::DenseTensor>();

  auto &pool = phi::DeviceContextPool::Instance();

  // Make pool2d followed by shape. shape for NHWC should return
  // as output tensor not-rotated shape of Pool (

  auto ksize = std::vector<int>(2, 2);
  auto op_pool =
      framework::OpRegistry::CreateOp("pool2d",
                                      {{"X", {"x"}}},
                                      {{"Out", {"y"}}},
                                      {{"pooling_type", {std::string("max")}},
                                       {"ksize", {ksize}},
                                       {"data_format", {std::string("NHWC")}},
                                       {"use_mkldnn", {true}}});

  auto op_shape = framework::OpRegistry::CreateOp(
      "shape", {{"Input", {"y"}}}, {{"Out", {"z"}}}, {{"use_mkldnn", {true}}});

  op_pool->Run(scope, p);
  op_shape->Run(scope, p);

  pool.Get(p)->Wait();

  // repack tensor data into vector for easy comparison
  auto *zdata = z->data<int32_t>();
  std::vector<int32_t> vzdata(zdata, zdata + z->numel());

  // Verify shape of output
  PADDLE_ENFORCE_EQ(vzdata,
                    expected_dims,
                    common::errors::InvalidArgument(
                        "Computed shape does not match expected shape"));
}

}  // namespace operators
}  // namespace paddle

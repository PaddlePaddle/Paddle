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
#include "paddle/phi/core/kernel_registry.h"

USE_OP_ITSELF(pool2d);
PD_DECLARE_KERNEL(pool2d, OneDNN, ALL_LAYOUT);
USE_OP_ITSELF(relu);
PD_DECLARE_KERNEL(relu, OneDNN, ALL_LAYOUT);
USE_OP_ITSELF(transpose);
USE_OP_DEVICE_KERNEL(transpose, MKLDNN);
USE_OP_ITSELF(shape);
USE_OP_DEVICE_KERNEL(shape, MKLDNN);
USE_OP_ITSELF(crop);
USE_OP_DEVICE_KERNEL(crop, CPU);

PD_DECLARE_KERNEL(pool2d, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(shape, CPU, ALL_LAYOUT);

namespace paddle {
namespace operators {

struct InputVars {
  std::string name;
  framework::LoDTensor *tensor;
};

TEST(test_pool2d_transpose_nhwc, cpu_place) {
  framework::DDim dims({1, 4, 8, 512});           // NHWC shape
  framework::DDim expected_dims({1, 7, 512, 3});  // NHWC expected shape
  platform::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x",
                          scope.Var("x")->GetMutable<framework::LoDTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(static_cast<float>(10.0),
                                             static_cast<float>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(phi::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }

  scope.Var("y")->GetMutable<framework::LoDTensor>();
  auto *z = scope.Var("z")->GetMutable<framework::LoDTensor>();

  auto &pool = platform::DeviceContextPool::Instance();

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
      "transpose",
      {{"X", {"y"}}},
      {{"Out", {"z"}}},
      {{"axis", {axis}}, {"use_mkldnn", {true}}});

  op_pool->Run(scope, p);
  op_transpose->Run(scope, p);
  pool.Get(p)->Wait();

  // Verify shape of output
  PADDLE_ENFORCE_EQ(z->dims(),
                    expected_dims,
                    platform::errors::InvalidArgument(
                        "Computed shape does not match expected shape"));
}

TEST(test_pool2d_relu_relu_nhwc, cpu_place) {
  framework::DDim dims({1, 4, 8, 512});           // NHWC shape
  framework::DDim expected_dims({1, 512, 3, 7});  // NCHW expected shape
  platform::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x",
                          scope.Var("x")->GetMutable<framework::LoDTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(static_cast<float>(10.0),
                                             static_cast<float>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(phi::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }

  scope.Var("y")->GetMutable<framework::LoDTensor>();
  scope.Var("u")->GetMutable<framework::LoDTensor>();
  auto *z = scope.Var("z")->GetMutable<framework::LoDTensor>();

  auto &pool = platform::DeviceContextPool::Instance();

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
                    platform::errors::InvalidArgument(
                        "Computed shape does not match expected shape"));
}

TEST(test_pool2d_shape_nhwc, cpu_place) {
  framework::DDim dims({1, 4, 8, 512});              // NHWC shape
  std::vector<int32_t> expected_dims{1, 3, 7, 512};  // NHWC expected shape
  platform::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x",
                          scope.Var("x")->GetMutable<framework::LoDTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(static_cast<float>(10.0),
                                             static_cast<float>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(phi::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }

  scope.Var("y")->GetMutable<framework::LoDTensor>();
  auto *z = scope.Var("z")->GetMutable<framework::LoDTensor>();

  auto &pool = platform::DeviceContextPool::Instance();

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
                    platform::errors::InvalidArgument(
                        "Computed shape does not match expected shape"));
}

TEST(test_pool2d_crop_nhwc, cpu_place) {
  framework::DDim dims({1, 4, 8, 512});           // NHWC shape
  framework::DDim expected_dims({1, 3, 7, 512});  // NCHW expected shape
  platform::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x",
                          scope.Var("x")->GetMutable<framework::LoDTensor>()};
  InputVars second_crop_input_name = {
      "v", scope.Var("v")->GetMutable<framework::LoDTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(10.0f, 20.0f);
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(phi::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }
  // Second input (Y) to crop is having no buffer
  // but as it is MKLDNN then its shape order should be NCHW
  auto expected_dims_nchw = phi::vectorize<int64_t>(expected_dims);
  std::rotate(expected_dims_nchw.begin() + 1,
              expected_dims_nchw.end() - 1,
              expected_dims_nchw.end());
  second_crop_input_name.tensor->Resize(phi::make_ddim(expected_dims_nchw));
  const auto second_crop_input_md =
      dnnl::memory::desc(expected_dims_nchw,
                         dnnl::memory::data_type::f32,
                         dnnl::memory::format_tag::nhwc);
  second_crop_input_name.tensor->set_mem_desc(second_crop_input_md);

  scope.Var("y")->GetMutable<framework::LoDTensor>();
  auto *z = scope.Var("z")->GetMutable<framework::LoDTensor>();

  auto &pool = platform::DeviceContextPool::Instance();

  // Make pool2d followed by crop. crop may have Y input as
  // non buffered so the path to be executed is handling oneDNN kernel
  // that is followed by CPU kernel with non-buffered Input

  auto ksize = std::vector<int>(2, 2);
  auto op_pool =
      framework::OpRegistry::CreateOp("pool2d",
                                      {{"X", {"x"}}},
                                      {{"Out", {"y"}}},
                                      {{"pooling_type", {std::string("max")}},
                                       {"ksize", {ksize}},
                                       {"data_format", {std::string("NHWC")}},
                                       {"use_mkldnn", {true}}});

  std::vector<int> offsets{0, 0, 0, 0};
  auto op_crop = framework::OpRegistry::CreateOp("crop",
                                                 {{"X", {"y"}}, {"Y", {"v"}}},
                                                 {{"Out", {"z"}}},
                                                 {{"offsets", {offsets}}});

  op_pool->Run(scope, p);
  op_crop->Run(scope, p);

  pool.Get(p)->Wait();

  // Verify shape of output
  PADDLE_ENFORCE_EQ(z->dims(),
                    expected_dims,
                    platform::errors::InvalidArgument(
                        "Output shape does not match expected output shape"));
}

}  // namespace operators
}  // namespace paddle

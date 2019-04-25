/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/benchmark/op_test.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace benchmark {

using framework::LoDTensor;
using framework::OperatorBase;
using framework::Scope;
using framework::DataLayout;
using platform::CPUPlace;
using mkldnn::memory;

template <typename T>
void Multiply(const T* x, const T* y, T* z, int height, int width,
              int simd_width) {
  int offset = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int i = 0; i < simd_width; ++i) {
        z[i + offset] = y[i] * x[i + offset];
      }
      offset += simd_width;
    }
  }
}

template <typename T>
static void ComputeReferenceOutput(LoDTensor* x, LoDTensor* y, T* z,
                                   std::vector<int64_t> dims, int simd_width) {
  auto* x_data = x->data<T>();
  auto* y_data = y->data<T>();

  const int n = dims[0];
  const int C = dims[1] / simd_width;
  const int h = dims[2];
  const int w = dims[3];

  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < C; ci++) {
      auto ptr_x =
          x_data + ni * C * h * w * simd_width + ci * h * w * simd_width;
      auto ptr_y = y_data + ni * C * simd_width + ci * simd_width;
      auto ptr_z = z + ni * C * h * w * simd_width + ci * h * w * simd_width;
      Multiply<T>(ptr_x, ptr_y, ptr_z, h, w, simd_width);
    }
  }
}

template <typename T>
void MainTest(memory::format x_format, int simd_width) {
  OpTest tester;
  OpTesterConfig config;
  config.op_type = "elementwise_mul";
  config.attrs["use_mkldnn"] = "1";
  config.attrs["axis"] = "0";
  config.inputs.resize(2);
  config.inputs[0].name = "X";
  config.inputs[0].dims = {2, 32, 4, 4};
  config.inputs[0].dtype = "fp32";
  config.inputs[1].name = "Y";
  config.inputs[1].dims = {2, 32};
  config.inputs[1].dtype = "fp32";
  tester.Init(config);

  OperatorBase* op = tester.Op();
  Scope* scope = tester.Scope();

  auto* x_tensor = scope->FindVar(op->Input("X"))->GetMutable<LoDTensor>();
  auto* y_tensor = scope->FindVar(op->Input("Y"))->GetMutable<LoDTensor>();
  auto* z_tensor = scope->FindVar(op->Output("Out"))->GetMutable<LoDTensor>();

  x_tensor->set_format(x_format);
  y_tensor->set_format(memory::format::nc);
  x_tensor->set_layout(DataLayout::kMKLDNN);
  y_tensor->set_layout(DataLayout::kMKLDNN);

  tester.RunImpl(false);

  std::unique_ptr<T[]> z_refer = std::unique_ptr<T[]>(new T[z_tensor->numel()]);
  ComputeReferenceOutput(x_tensor, y_tensor, z_refer.get(),
                         config.inputs[0].dims, simd_width);

  auto* z_data = z_tensor->data<T>();
  for (int64_t i = 0; i < z_tensor->numel(); i++) {
    EXPECT_EQ(z_data[i], z_refer[i]);
  }

  scope->DropKids();
}

TEST(elementwise_mul_test, NCHW16CTimesNC) {
  if (!platform::MayIUse(platform::avx512f)) {
    VLOG(0) << "The test has been skipped since AVX512 is not supported";
    return;
  }
  memory::format x_format = memory::format::nChw16c;
  constexpr int simd_width = 16;
  MainTest<float>(x_format, simd_width);
}

TEST(elementwise_mul_test, NCHW8CTimesNC) {
  if (!platform::MayIUse(platform::avx2)) {
    VLOG(0) << "The test has been skipped since AVX2 is not supported";
    return;
  }
  memory::format x_format = memory::format::nChw8c;
  constexpr int simd_width = 8;
  MainTest<float>(x_format, simd_width);
}
}  // namespace benchmark
}  // namespace operators
}  // namespace paddle

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
#include <map>
#include <random>
#include <string>

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

class CacheTester {
 public:
  CacheTester() {
    // Clear oneDNN cache
    auto &pool = phi::DeviceContextPool::Instance();
    phi::CPUPlace place;
    onednn_dev_ctx_ = dynamic_cast<phi::OneDNNContext *>(pool.Get(place));
    onednn_dev_ctx_->ResetBlobMap(nullptr);
  }

  bool Analyze(uint16_t num_entries) {
    //  Number of created objects in cache should be as expected (num_entries)
    return onednn_dev_ctx_->GetCachedObjectsNumber() == num_entries;
  }

 private:
  phi::OneDNNContext *onednn_dev_ctx_;
};

template <typename T>
void RunOperator(const phi::Place &place,
                 const std::string &op_type,
                 const phi::DDim &dims,
                 const std::string &first_input) {
  framework::Scope scope;

  std::map<const std::string, int> num_inputs = {{"softmax", 1},
                                                 {"relu", 1},
                                                 {"conv2d", 2},
                                                 {"elementwise_add", 2},
                                                 {"elementwise_mul", 2}};

  std::string first_input_var_name = (op_type == "conv2d") ? "Input" : "X";
  std::string second_input_var_name = (op_type == "conv2d") ? "Filter" : "Y";
  std::string output_var_name = (op_type == "conv2d") ? "Output" : "Out";
  std::string output_name = "output";

  std::vector<InputVars> input_names = {
      {first_input, scope.Var(first_input)->GetMutable<phi::DenseTensor>()},
      {"x1",
       num_inputs[op_type] > 1 ? scope.Var("x1")->GetMutable<phi::DenseTensor>()
                               : nullptr},
      {"x2",
       num_inputs[op_type] > 2 ? scope.Var("x2")->GetMutable<phi::DenseTensor>()
                               : nullptr},
      {"x3",
       num_inputs[op_type] > 3 ? scope.Var("x3")->GetMutable<phi::DenseTensor>()
                               : nullptr},
      {"x4",
       num_inputs[op_type] > 4 ? scope.Var("x4")->GetMutable<phi::DenseTensor>()
                               : nullptr}};
  auto *y = scope.Var(output_name)->GetMutable<phi::DenseTensor>();

  // Initialize input data
  std::uniform_real_distribution<T> dist(static_cast<T>(10.0),
                                         static_cast<T>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(common::product(dims));
  for (int i = 0; i < num_inputs[op_type]; ++i) {
    input_names[i].tensor->Resize(dims);
    auto data_ptr = input_names[i].tensor->mutable_data<T>(place);
    for (size_t i = 0; i < numel; ++i) {
      data_ptr[i] = dist(engine);
    }
  }

  // Initialize output
  y->Resize(dims);
  auto y_ptr = y->mutable_data<T>(place);
  for (size_t i = 0; i < numel; ++i) {
    y_ptr[i] = static_cast<T>(0);
  }

  auto &pool = phi::DeviceContextPool::Instance();

  auto op = num_inputs[op_type] > 1
                ? framework::OpRegistry::CreateOp(
                      op_type,
                      {{first_input_var_name, {first_input}},
                       {second_input_var_name, {"x1"}}},
                      {{output_var_name, {output_name}}},
                      {{"use_mkldnn", {true}}})
                : framework::OpRegistry::CreateOp(
                      op_type,
                      {{first_input_var_name, {first_input}}},
                      {{output_var_name, {output_name}}},
                      {{"use_mkldnn", {true}}});

  op->Run(scope, place);
  pool.Get(place)->Wait();
}

TEST(test_conv2d_reuse_cache, cpu_place) {
  phi::DDim dims({1, 16, 32, 64});
  phi::CPUPlace p;
  CacheTester ct;
  RunOperator<float>(p, "conv2d", dims, "input_signal");
  RunOperator<float>(p, "conv2d", dims, "input_signal");
  PADDLE_ENFORCE_EQ(ct.Analyze(9),
                    true,
                    common::errors::InvalidArgument(
                        "Invalid number of cached oneDNN objects"));
}

TEST(test_conv2d_noreuse_cache, cpu_place) {
  phi::DDim dims({1, 16, 32, 64});
  phi::CPUPlace p;
  CacheTester ct;
  RunOperator<float>(p, "conv2d", dims, "input_signal");
  RunOperator<float>(p, "conv2d", dims, "input_signal2");
  PADDLE_ENFORCE_EQ(ct.Analyze(18),
                    true,
                    common::errors::InvalidArgument(
                        "Invalid number of cached oneDNN objects"));
}

}  // namespace operators
}  // namespace paddle

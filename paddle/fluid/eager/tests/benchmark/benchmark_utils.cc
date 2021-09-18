// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Eager
#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/tests/test_utils.h"

// Fluid
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"

namespace egr {

void benchmark_eager_accuracy_check(const pt::Tensor& tensor) {
  // 2. Run Forward for certain number of times
  pt::Tensor input_tensor = tensor;
  float scale = 2.0;
  float bias = 3.0;

  size_t max_num_runs = 10;
  for (size_t i = 0; i < max_num_runs; i++) {
    input_tensor =
        egr::scale(input_tensor, scale, bias, true /*bias_after_scale*/,
                   true /*trace_backward*/);
  }

  // 3. Run Backward
  std::vector<pt::Tensor> target_tensors = {input_tensor};
  RunBackward(target_tensors, {});

  // Examine Forward Grad (w.r.t max_num_runs = 10)
  PADDLE_ENFORCE(
      CompareTensorWithValue<float>(input_tensor, 8189.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 8189.0));
  // Examine Backward Grad (w.r.t max_num_runs = 10)
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 1024.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1024.0));
}

void benchmark_eager(const pt::Tensor& tensor) {
  // 2. Run Forward for certain number of times
  pt::Tensor input_tensor = tensor;
  float scale = 2.0;
  float bias = 3.0;

  size_t max_num_runs = 5000;
  for (size_t i = 0; i < max_num_runs; i++) {
    input_tensor =
        egr::scale(input_tensor, scale, bias, true /*bias_after_scale*/,
                   true /*trace_backward*/);
  }

  // 3. Run Backward
  std::vector<pt::Tensor> target_tensors = {input_tensor};
  RunBackward(target_tensors, {});
}

}  // namespace egr

namespace paddle {
namespace imperative {

// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_accuracy_check(
    std::shared_ptr<imperative::VarBase>& X,    // NOLINT
    std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
    const paddle::platform::Place& place) {
  imperative::Tracer tracer;

  framework::AttributeMap attrs;

  attrs["use_mkldnn"] = false;
  attrs["scale"] = 2;
  attrs["bias"] = 3;
  attrs["bias_after_scale"] = true;

  // NameVarBaseMap = std::map<std::string,
  // std::vector<std::shared_ptr<VarBase>>>
  imperative::NameVarBaseMap outs = {{"Out", {Out}}};
  imperative::NameVarBaseMap ins = {{"X", {X}}};

  size_t max_num_runs = 10;
  for (size_t i = 0; i < max_num_runs; i++) {
    tracer.TraceOp("scale", ins, outs, attrs, place, true);
    if (i != max_num_runs - 1) {
      ins = {{"X", outs["Out"]}};
      outs = {{"Out",
               {std::shared_ptr<imperative::VarBase>(
                   new imperative::VarBase(true, "Out"))}}};
    }
  }

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init(outs["Out"], grad_tensors, false /*retain_graph*/);
  engine->Execute();

  // Fwd Check: Expects 8189 with max_num_runs = 10
  auto* tensor =
      outs["Out"][0]->MutableVar()->GetMutable<framework::LoDTensor>();
  float* t_ptr = tensor->mutable_data<float>(place);
  std::vector<float> host_data(tensor->numel());
  if (place == paddle::platform::CUDAPlace()) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), t_ptr,
                         sizeof(float) * tensor->numel(), stream);
    t_ptr = host_data.data();
  }
  PADDLE_ENFORCE(
      t_ptr[0] == 8189.0,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 8189.0));

  // Grad Check: Expects 1024.0 with max_num_runs = 10
  auto* grad_tensor = X->MutableGradVar()->GetMutable<framework::LoDTensor>();
  float* g_ptr = grad_tensor->mutable_data<float>(place);
  std::vector<float> g_host_data(grad_tensor->numel());
  if (place == paddle::platform::CUDAPlace()) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), g_host_data.data(),
                         paddle::platform::CUDAPlace(), g_ptr,
                         sizeof(float) * grad_tensor->numel(), stream);
    g_ptr = g_host_data.data();
  }
  PADDLE_ENFORCE(
      g_ptr[0] == 1024.0,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1024.0));
}

// TODO(jiabin): Change this and remove nolint
void benchmark_fluid(std::shared_ptr<imperative::VarBase>& X,    // NOLINT
                     std::shared_ptr<imperative::VarBase>& Out,  // NOLINT
                     const paddle::platform::Place& place) {
  imperative::Tracer tracer;

  framework::AttributeMap attrs;

  attrs["use_mkldnn"] = false;
  attrs["scale"] = 2;
  attrs["bias"] = 3;
  attrs["bias_after_scale"] = true;

  // NameVarBaseMap = std::map<std::string,
  // std::vector<std::shared_ptr<VarBase>>>
  imperative::NameVarBaseMap outs = {{"Out", {Out}}};
  imperative::NameVarBaseMap ins = {{"X", {X}}};

  size_t max_num_runs = 5000;
  for (size_t i = 0; i < max_num_runs; i++) {
    tracer.TraceOp("scale", ins, outs, attrs, place, true);
    if (i != max_num_runs - 1) {
      ins = {{"X", outs["Out"]}};
      outs = {{"Out",
               {std::shared_ptr<imperative::VarBase>(
                   new imperative::VarBase(true, "Out"))}}};
    }
  }

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init(outs["Out"], grad_tensors, false /*retain_graph*/);
  engine->Execute();
}

}  // namespace imperative
}  // namespace paddle

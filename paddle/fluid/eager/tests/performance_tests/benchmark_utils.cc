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

#include "paddle/fluid/eager/tests/performance_tests/benchmark_utils.h"

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Eager
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/eager/utils.h"

// Eager Generated
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"

// Fluid
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

static size_t max_num_benchmark_runs = 4000;

namespace egr {

/* --------------------- */
/* ---- Eager Scale ---- */
/* --------------------- */
void benchmark_eager_scale(const paddle::experimental::Tensor& tensor,
                           bool accuracy_check) {
  paddle::experimental::Tensor input_tensor = tensor;
  float scale = 2.0;
  float bias = 3.0;

  size_t max_num_runs = accuracy_check ? 10 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    input_tensor = egr::scale(input_tensor,
                              scale,
                              bias,
                              true /*bias_after_scale*/,
                              true /*trace_backward*/);
  }

  std::vector<paddle::experimental::Tensor> target_tensors = {input_tensor};
  Backward(target_tensors, {});

  if (accuracy_check) {
    // Examine Forward Grad (w.r.t max_num_runs = 10)
    eager_test::CompareTensorWithValue<float>(input_tensor, 8189.0);
    // Examine Backward Grad (w.r.t max_num_runs = 10)
    eager_test::CompareGradTensorWithValue<float>(tensor, 1024.0);
  }
}

void benchmark_eager_matmul(const paddle::experimental::Tensor& X,
                            const paddle::experimental::Tensor& Y,
                            bool accuracy_check) {
  paddle::experimental::Tensor input_tensor0 = X;

  size_t max_num_runs = accuracy_check ? 2 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    input_tensor0 = matmul_ad_func(input_tensor0, Y, false, false);
  }

  std::vector<paddle::experimental::Tensor> target_tensors = {input_tensor0};
  Backward(target_tensors, {});

  if (accuracy_check) {
    // Examine Forward Grad (w.r.t max_num_runs = 2)
    eager_test::CompareTensorWithValue<float>(input_tensor0, 16);
    // Examine Backward Grad (w.r.t max_num_runs = 2)
    eager_test::CompareGradTensorWithValue<float>(X, 16);
    eager_test::CompareGradTensorWithValue<float>(Y, 16);
  }
}

/* ----------------------------------- */
/* ---- Eager Intermediate Matmul ---- */
/* ----------------------------------- */
void benchmark_eager_intermediate_matmul(const paddle::experimental::Tensor& X,
                                         const paddle::experimental::Tensor& Y,
                                         bool accuracy_check) {
  paddle::experimental::Tensor input_tensor0 = X;

  size_t max_num_runs = accuracy_check ? 2 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    input_tensor0 = matmul_v2_dygraph_function(
        input_tensor0, Y, {{"trans_x", false}, {"trans_y", false}});
  }

  std::vector<paddle::experimental::Tensor> target_tensors = {input_tensor0};
  Backward(target_tensors, {});

  if (accuracy_check) {
    // Examine Forward Grad (w.r.t max_num_runs = 2)
    eager_test::CompareTensorWithValue<float>(input_tensor0, 16);
    // Examine Backward Grad (w.r.t max_num_runs = 2)
    eager_test::CompareGradTensorWithValue<float>(X, 16);
    eager_test::CompareGradTensorWithValue<float>(Y, 16);
  }
}

/* -------------------------------- */
/* ---- Eager Intermediate MLP ---- */
/* -------------------------------- */
void benchmark_eager_intermediate_mlp(
    const paddle::experimental::Tensor& X,
    const std::vector<paddle::experimental::Tensor>& Ws,
    const std::vector<paddle::experimental::Tensor>& Bs,
    bool accuracy_check) {
  paddle::experimental::Tensor input0 = X;

  for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
    paddle::experimental::Tensor Out = matmul_v2_dygraph_function(
        input0, Ws[i], {{"trans_x", false}, {"trans_y", false}});

    input0 = elementwise_add_dygraph_function(Out, Bs[i], {});
  }

  paddle::experimental::Tensor Out =
      reduce_sum_dygraph_function(input0, {{"reduce_all", true}});

  std::vector<paddle::experimental::Tensor> target_tensors = {Out};
  Backward(target_tensors, {});

  if (accuracy_check) {
    std::unordered_map<std::string, float> result =
        compute_mlp_expected_results();

    // Examine Forward Grad (w.r.t max_num_runs = 2)
    eager_test::CompareTensorWithValue<float>(Out, result["Out"]);

    // Examine Backward Grad (w.r.t max_num_runs = 2)
    eager_test::CompareGradTensorWithValue<float>(X, result["GradX"]);
    eager_test::CompareGradTensorWithValue<float>(Ws[0], result["GradW"]);
  }
}

}  // namespace egr

namespace paddle {
namespace imperative {

static void FluidCheckTensorValue(const std::shared_ptr<imperative::VarBase>& X,
                                  const paddle::platform::Place& place,
                                  float value) {
  auto* tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
  float* t_ptr = tensor->mutable_data<float>(place);
  std::vector<float> host_data(tensor->numel());

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place == paddle::platform::CUDAPlace()) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<phi::GPUContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(),
                         host_data.data(),
                         paddle::platform::CUDAPlace(),
                         t_ptr,
                         sizeof(float) * tensor->numel(),
                         stream);
    t_ptr = host_data.data();
  }
#endif

  VLOG(6) << "Tensor Value: " << t_ptr[0] << ", Expected Value: " << value;
  PADDLE_ENFORCE(
      t_ptr[0] == value,
      paddle::platform::errors::Fatal(
          "Detected numerical Error, Expected %f but got %f", value, t_ptr[0]));
}

static void FluidCheckGradTensorValue(
    const std::shared_ptr<imperative::VarBase>& X,
    const paddle::platform::Place& place,
    float value) {
  auto* grad_tensor = X->MutableGradVar()->GetMutable<framework::LoDTensor>();
  float* g_ptr = grad_tensor->mutable_data<float>(place);
  std::vector<float> g_host_data(grad_tensor->numel());

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place == paddle::platform::CUDAPlace()) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<phi::GPUContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(),
                         g_host_data.data(),
                         paddle::platform::CUDAPlace(),
                         g_ptr,
                         sizeof(float) * grad_tensor->numel(),
                         stream);
    g_ptr = g_host_data.data();
  }
#endif

  VLOG(6) << "Tensor Value: " << g_ptr[0] << ", Expected Value: " << value;
  PADDLE_ENFORCE(
      g_ptr[0] == value,
      paddle::platform::errors::Fatal(
          "Detected numerical Error, Expected %f but got %f", value, g_ptr[0]));
}

/* --------------------- */
/* ---- Fluid Scale ---- */
/* --------------------- */
// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_scale(const std::shared_ptr<imperative::VarBase>& X,
                           const paddle::platform::Place& place,
                           bool accuracy_check) {
  imperative::Tracer tracer;
  framework::AttributeMap attrs;

  attrs["use_mkldnn"] = false;
  attrs["scale"] = 2;
  attrs["bias"] = 3;
  attrs["bias_after_scale"] = true;

  std::shared_ptr<imperative::VarBase> tmp_out = X;

  size_t max_num_runs = accuracy_check ? 10 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    imperative::NameVarBaseMap ins = {{"X", {tmp_out}}};
    imperative::NameVarBaseMap outs = {
        {"Out",
         {std::shared_ptr<imperative::VarBase>(
             new imperative::VarBase(true, "Out"))}}};

    tracer.TraceOp<VarBase>("scale", ins, outs, attrs, place, true);

    tmp_out = outs["Out"][0];
  }

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init({tmp_out}, grad_tensors, false /*retain_graph*/);
  engine->Execute();

  if (accuracy_check) {
    FluidCheckTensorValue(tmp_out, place, 8189.0);
    FluidCheckGradTensorValue(X, place, 1024.0);
  }
}

/* ---------------------- */
/* ---- Fluid Matmul ---- */
/* ---------------------- */
void benchmark_fluid_matmul(const std::shared_ptr<imperative::VarBase>& X,
                            const std::shared_ptr<imperative::VarBase>& Y,
                            const paddle::platform::Place& place,
                            bool accuracy_check) {
  imperative::Tracer tracer;

  std::shared_ptr<imperative::VarBase> tmp_out = X;

  size_t max_num_runs = accuracy_check ? 2 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    framework::AttributeMap attrs;
    imperative::NameVarBaseMap ins = {{"X", {tmp_out}}, {"Y", {Y}}};
    imperative::NameVarBaseMap outs = {
        {"Out",
         {std::shared_ptr<imperative::VarBase>(
             new imperative::VarBase(true, "Out"))}}};

    tracer.TraceOp<VarBase>("matmul_v2", ins, outs, attrs, place, true);

    tmp_out = outs["Out"][0];
  }

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init({tmp_out}, grad_tensors, false /*retain_graph*/);
  engine->Execute();

  if (accuracy_check) {
    FluidCheckTensorValue(tmp_out, place, 16);
    FluidCheckGradTensorValue(X, place, 16);
    FluidCheckGradTensorValue(Y, place, 16);
  }
}

/* ------------------- */
/* ---- Fluid MLP ---- */
/* ------------------- */
void benchmark_fluid_mlp(
    const std::shared_ptr<imperative::VarBase>& X,
    const std::vector<std::shared_ptr<imperative::VarBase>>& Ws,
    const std::vector<std::shared_ptr<imperative::VarBase>>& Bs,
    const paddle::platform::Place& place,
    bool accuracy_check) {
  imperative::Tracer tracer;

  imperative::NameVarBaseMap ins;
  imperative::NameVarBaseMap outs;
  framework::AttributeMap attrs;
  std::shared_ptr<imperative::VarBase> input0 = X;
  for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
    // Matmul0
    ins = {{"X", {input0}}, {"Y", {Ws[0]}}};
    outs = {{"Out",
             {std::shared_ptr<imperative::VarBase>(
                 new imperative::VarBase(true, "Out"))}}};

    tracer.TraceOp<VarBase>("matmul_v2", ins, outs, attrs, place, true);

    // EW-Add0
    ins = {{"X", outs["Out"]}, {"Y", {Bs[i]}}};
    outs = {{"Out",
             {std::shared_ptr<imperative::VarBase>(
                 new imperative::VarBase(true, "Out"))}}};

    tracer.TraceOp<VarBase>("elementwise_add", ins, outs, attrs, place, true);
    input0 = outs["Out"][0];
  }

  // ReduceSum
  ins = {{"X", {input0}}};
  outs = {{"Out",
           {std::shared_ptr<imperative::VarBase>(
               new imperative::VarBase(true, "Out"))}}};
  attrs = {{"reduce_all", true}};

  tracer.TraceOp<VarBase>("reduce_sum", ins, outs, attrs, place, true);

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init(outs["Out"], grad_tensors, false /*retain_graph*/);
  engine->Execute();

  if (accuracy_check) {
    std::unordered_map<std::string, float> result =
        egr::compute_mlp_expected_results();

    FluidCheckTensorValue(outs["Out"][0], place, result["Out"]);
    FluidCheckGradTensorValue(X, place, result["GradX"]);
    FluidCheckGradTensorValue(Ws[0], place, result["GradW"]);
  }
}

}  // namespace imperative
}  // namespace paddle

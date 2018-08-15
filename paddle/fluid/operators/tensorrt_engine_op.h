/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_CUDA

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {

DECLARE_int32(tensorrt_engine_batch_size);

namespace operators {

using inference::Singleton;
using inference::tensorrt::TRT_EngineManager;

class TensorRTEngineOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input0 = ctx.Inputs("Xs").front();
    framework::OpKernelType kt = framework::OpKernelType(
        framework::ToDataType(ctx.scope()
                                  .FindVar(input0)
                                  ->GetMutable<framework::LoDTensor>()
                                  ->type()),
        platform::CPUPlace());
    return kt;
  }
};

template <typename DeviceContext, typename T>
class TensorRTEngineKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto engine_name = context.Attr<std::string>("engine_uniq_key");
    if (!Singleton<TRT_EngineManager>::Global().HasEngine(engine_name)) {
      Prepare(context);
    }
    auto* engine = Singleton<TRT_EngineManager>::Global().Get(engine_name);
    auto input_names = context.op().Inputs("Xs");
    PADDLE_ENFORCE(!input_names.empty(), "should pass more than one inputs");
    PADDLE_ENFORCE_LE(FLAGS_tensorrt_engine_batch_size,
                      context.Attr<int>("max_batch"));

    std::vector<std::string> output_maps =
        context.Attr<std::vector<std::string>>("output_name_mapping");

    auto params = context.Attr<std::vector<std::string>>("parameters");
    std::unordered_set<std::string> parameters;
    for (const auto& param : params) {
      parameters.insert(param);
    }
    // Convert input tensor from fluid to engine.
    for (const auto& x : context.Inputs("Xs")) {
      if (parameters.count(x)) continue;
      // convert input and copy to TRT engine's buffer
      auto& t = inference::analysis::GetFromScope<framework::LoDTensor>(
          context.scope(), x);
      if (platform::is_cpu_place(t.place())) {
        engine->SetInputFromCPU(x, static_cast<const void*>(t.data<void>()),
                                t.memory_size());
      } else {
        engine->SetInputFromGPU(x, static_cast<const void*>(t.data<void>()),
                                t.memory_size());
      }
    }
    // Execute the engine.
    PADDLE_ENFORCE_GT(FLAGS_tensorrt_engine_batch_size, 0);
    engine->Execute(FLAGS_tensorrt_engine_batch_size);

    // Convert output tensor from engine to fluid
    int output_index = 0;
    for (const auto& y : context.Outputs("Ys")) {
      // convert output and copy to fluid.
      nvinfer1::ITensor* trt_t = engine->GetITensor(output_maps[output_index]);
      auto dims = trt_t->getDimensions();
      // Use the output ITensor's dims to reshape the Fluid Tensor.
      std::vector<int> ddim(dims.d, dims.d + dims.nbDims);

      auto* fluid_v = context.scope().FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(fluid_v, "no output variable called %s", y);
      auto* fluid_t = fluid_v->GetMutable<framework::LoDTensor>();

      fluid_t->Resize(framework::make_ddim(ddim));

      // TODO(Superjomn) find some way to determine which device to output the
      // tensor.
      // if (platform::is_cpu_place(fluid_t->place())) {
      // TODO(Superjomn) change this float to dtype size.
      auto size = inference::analysis::AccuDims(dims.d, dims.nbDims) *
                  FLAGS_tensorrt_engine_batch_size;
      engine->GetOutputInCPU(output_maps[output_index],
                             fluid_t->mutable_data<float>(platform::CPUPlace()),
                             size * sizeof(float));
      //} else {
      // engine->GetOutputInGPU(
      // y, fluid_t->mutable_data<float>(platform::CUDAPlace()),
      // size * sizeof(float));
      //}
      output_index += 1;
    }

    cudaStreamSynchronize(*engine->stream());
  }

 protected:
  // Build the engine.
  void Prepare(const framework::ExecutionContext& context) const;
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA

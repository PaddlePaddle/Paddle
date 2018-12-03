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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {

DECLARE_int32(tensorrt_engine_batch_size);

namespace operators {

using FluidDT = framework::proto::VarType_Type;
using TRT_DT = nvinfer1::DataType;

namespace {  // NOLINT

TRT_DT FluidDataType2TRT(FluidDT type) {
  switch (type) {
    case FluidDT::VarType_Type_FP32:
      return TRT_DT::kFLOAT;
    case FluidDT::VarType_Type_INT32:
      return TRT_DT::kINT32;
    default:
      return TRT_DT::kINT32;
  }
  PADDLE_THROW("unkown type");
  return TRT_DT::kINT32;
}

nvinfer1::Dims Vec2TRT_Dims(const std::vector<int64_t>& shape) {
  PADDLE_ENFORCE_GT(shape.size(), 1UL,
                    "TensorRT' tensor input requires at least 2 dimensions");
  PADDLE_ENFORCE_LE(shape.size(), 4UL,
                    "TensorRT' tensor input requires at most 4 dimensions");
  PADDLE_ENFORCE(shape.size() == 4UL || shape.size() == 2UL);
  if (shape.size() == 4UL)
    return nvinfer1::DimsCHW(shape[1], shape[2], shape[3]);
  return nvinfer1::DimsCHW(shape[1], 1, 1);
}

}  // namespace // NOLINT

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
        ctx.GetPlace());
    return kt;
  }
};

template <typename DeviceContext, typename T>
class TensorRTEngineKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto engine_name = context.Attr<std::string>("engine_uniq_key");
    int max_batch_size = context.Attr<int>("max_batch_size");
    if (!Singleton<TRT_EngineManager>::Global().HasEngine(engine_name)) {
      Prepare(context);
    }
    auto* engine = Singleton<TRT_EngineManager>::Global().Get(engine_name);
    auto input_names = context.op().Inputs("Xs");
    PADDLE_ENFORCE(!input_names.empty(), "should pass more than one inputs");
    PADDLE_ENFORCE_LE(FLAGS_tensorrt_engine_batch_size, max_batch_size);

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
    VLOG(4) << "TensorRT Engine Op Outputs:";
    for (const auto& y : context.Outputs("Ys")) {
      VLOG(4) << y;
      // convert output and copy to fluid.
      nvinfer1::ITensor* trt_t = engine->GetITensor(output_maps[output_index]);
      auto dims = trt_t->getDimensions();
      // Use the output ITensor's dims to reshape the Fluid Tensor.
      // The ITensor doesn't contain the batch size dim.
      std::vector<int> ddim;
      ddim.push_back(FLAGS_tensorrt_engine_batch_size);
      for (int i = 0; i < dims.nbDims; i++) {
        ddim.push_back(dims.d[i]);
      }

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
      engine->GetOutputInGPU(
          output_maps[output_index],
          fluid_t->mutable_data<float>(platform::CUDAPlace(
              boost::get<platform::CUDAPlace>(context.GetPlace()).device)),
          size * sizeof(float));

      output_index += 1;
    }

    cudaStreamSynchronize(*engine->stream());
  }

 protected:
  void Prepare(const framework::ExecutionContext& context) const {
    VLOG(4) << "Prepare engine";
    // Get the ProgramDesc and pass to convert.
    framework::proto::BlockDesc block_desc;
    block_desc.ParseFromString(context.Attr<std::string>("subgraph"));
    int max_batch_size = context.Attr<int>("max_batch_size");
    int workspace_size = context.Attr<int>("workspace_size");

    auto params = context.Attr<std::vector<std::string>>("parameters");
    std::unordered_set<std::string> parameters;
    for (const auto& param : params) {
      parameters.insert(param);
    }

    std::vector<std::string> output_maps =
        context.Attr<std::vector<std::string>>("output_name_mapping");

    // TODO(Superjomn) replace this with a different stream
    auto* engine = Singleton<TRT_EngineManager>::Global().Create(
        max_batch_size, workspace_size, nullptr /*engine hold its own stream*/,
        context.Attr<std::string>("engine_uniq_key"),
        boost::get<platform::CUDAPlace>(context.GetPlace()).device);

    engine->InitNetwork();

    framework::BlockDesc block(nullptr /*programdesc*/, &block_desc);
    VLOG(4) << "parsed var size " << block.AllVars().size();
    // Add inputs
    VLOG(4) << "declare inputs";
    for (auto& input : context.Inputs("Xs")) {
      if (parameters.count(input)) continue;
      VLOG(4) << "declare input " << input;
      auto* var = block.FindVar(input);
      // TensorRT engine need to create parameters. The parameter's description
      // should be set in
      PADDLE_ENFORCE(var, "no variable called %s", input);
      PADDLE_ENFORCE_EQ(var->GetType(), FluidDT::VarType_Type_LOD_TENSOR,
                        "TensorRT engine only takes LoDTensor as input");
      auto shape = var->GetShape();
      // For the special batch_size placeholder -1, drop it and pass the real
      // shape of data.
      // TODO(Superjomn) fix this with batch broadcast, or it can't handle
      // variational batch size.
      if (shape[0] == -1) {
        shape[0] = FLAGS_tensorrt_engine_batch_size;
      }
      engine->DeclareInput(
          input, FluidDataType2TRT(
                     var->Proto()->type().lod_tensor().tensor().data_type()),
          Vec2TRT_Dims(shape));
    }

    inference::Singleton<inference::tensorrt::OpConverter>::Global()
        .ConvertBlock(block_desc, parameters, context.scope(), engine);

    // Add outputs
    for (auto& output : output_maps) {
      if (!engine->HasDeclared(output)) {
        engine->DeclareOutput(output);
      }
    }

    engine->FreezeNetwork();
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA

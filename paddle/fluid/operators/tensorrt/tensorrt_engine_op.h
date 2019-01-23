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

nvinfer1::Dims Vec2TRT_Dims(const std::vector<int64_t> &shape) {
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
using inference::tensorrt::TensorRTEngine;

class TensorRTEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  mutable std::unique_ptr<TensorRTEngine> trt_engine_;
  int max_batch_size_;
  int workspace_size_;

 public:
  TensorRTEngineOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    max_batch_size_ = Attr<int>("max_batch_size");
    workspace_size_ = Attr<int>("workspace_size");

    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }
  }

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    RunTrt(scope, dev_place);
  }

  void RunTrt(const framework::Scope &scope,
              const platform::Place &dev_place) const {
    int runtime_batch = 1;
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx).stream();
    if (trt_engine_.get() == nullptr) {
      trt_engine_.reset(new TensorRTEngine(
          max_batch_size_, workspace_size_, stream,
          boost::get<platform::CUDAPlace>(dev_place).device));
      Prepare(scope, dev_place, trt_engine_.get());
    }

    auto *engine = trt_engine_.get();
    PADDLE_ENFORCE(!input_names_.empty(), "should pass more than one inputs");

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    // Convert input tensor from fluid to engine.
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      // convert input and copy to TRT engine's buffer
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      auto t_shape = framework::vectorize(t.dims());
      runtime_batch = t_shape[0];
      if (platform::is_cpu_place(t.place())) {
        engine->SetInputFromCPU(x, static_cast<const void *>(t.data<void>()),
                                t.memory_size());
      } else {
        engine->SetInputFromGPU(x, static_cast<const void *>(t.data<void>()),
                                t.memory_size());
      }
    }

    cudaStreamSynchronize(stream);
    PADDLE_ENFORCE_LE(runtime_batch, max_batch_size_);
    // Execute the engine.
    engine->Execute(runtime_batch);

    // Convert output tensor from engine to fluid
    int output_index = 0;
    VLOG(4) << "TensorRT Engine Op Outputs:";
    for (const auto &y : Outputs("Ys")) {
      VLOG(4) << y;
      // convert output and copy to fluid.
      nvinfer1::ITensor *trt_t = engine->GetITensor(output_maps[output_index]);
      auto dims = trt_t->getDimensions();
      // Use the output ITensor's dims to reshape the Fluid Tensor.
      // The ITensor doesn't contain the batch size dim.
      std::vector<int> ddim;
      ddim.push_back(runtime_batch);
      for (int i = 0; i < dims.nbDims; i++) {
        ddim.push_back(dims.d[i]);
      }

      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(fluid_v, "no output variable called %s", y);
      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();

      fluid_t->Resize(framework::make_ddim(ddim));

      // TODO(Superjomn) change this float to dtype size.
      auto size =
          inference::analysis::AccuDims(dims.d, dims.nbDims) * runtime_batch;
      engine->GetOutputInGPU(
          output_maps[output_index],
          fluid_t->mutable_data<float>(platform::CUDAPlace(
              boost::get<platform::CUDAPlace>(dev_place).device)),
          size * sizeof(float));
      output_index += 1;
    }

    cudaStreamSynchronize(stream);
  }

  void Prepare(const framework::Scope &scope, const platform::Place &dev_place,
               TensorRTEngine *engine) const {
    VLOG(4) << "Prepare engine";
    framework::proto::BlockDesc block_desc;
    block_desc.ParseFromString(Attr<std::string>("subgraph"));

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    engine->InitNetwork();

    framework::BlockDesc block(nullptr /*programdesc*/, &block_desc);
    VLOG(4) << "parsed var size " << block.AllVars().size();
    // Add inputs
    VLOG(4) << "declare inputs";
    for (auto &input : Inputs("Xs")) {
      if (param_names_.count(input)) continue;
      VLOG(4) << "declare input " << input;

      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, input);
      auto t_shape = framework::vectorize(t.dims());

      auto *var = block.FindVar(input);
      // TensorRT engine need to create parameters. The parameter's description
      // should be set in
      PADDLE_ENFORCE(var, "no variable called %s", input);
      PADDLE_ENFORCE_EQ(var->GetType(), FluidDT::VarType_Type_LOD_TENSOR,
                        "TensorRT engine only takes LoDTensor as input");

      engine->DeclareInput(
          input, FluidDataType2TRT(
                     var->Proto()->type().lod_tensor().tensor().data_type()),
          Vec2TRT_Dims(t_shape));
    }
    inference::Singleton<inference::tensorrt::OpConverter>::Global()
        .ConvertBlock(block_desc, param_names_, scope, engine);

    // Add outputs
    for (auto &output : output_maps) {
      engine->DeclareOutput(output);
    }
    engine->FreezeNetwork();
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA

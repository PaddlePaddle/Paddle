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

#ifdef PADDLE_WITH_CUDA

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/operators/tensorrt_engine_op.h"

namespace paddle {

DEFINE_int32(tensorrt_engine_batch_size, 1, "the batch_size of TensorRT");

namespace operators {

using inference::Singleton;
using inference::tensorrt::TRT_EngineManager;

using FluidDT = framework::proto::VarType_Type;
using TRT_DT = nvinfer1::DataType;

namespace {

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
  PADDLE_ENFORCE_EQ(shape.size(), 4UL);
  return nvinfer1::DimsCHW(shape[1], shape[2], shape[3]);
}

}  // namespace

template <typename DeviceContext, typename T>
void TensorRTEngineKernel<DeviceContext, T>::Prepare(
    const framework::ExecutionContext &context) const {
  VLOG(4) << "Prepare engine";
  // Get the ProgramDesc and pass to convert.
  framework::proto::BlockDesc block_desc;
  block_desc.ParseFromString(context.Attr<std::string>("subgraph"));
  int max_batch = context.Attr<int>("max_batch");
  auto max_workspace = context.Attr<int>("max_workspace");
  auto params = context.Attr<std::vector<std::string>>("parameters");
  std::unordered_set<std::string> parameters;
  for (const auto &param : params) {
    parameters.insert(param);
  }

  std::vector<std::string> output_maps =
      context.Attr<std::vector<std::string>>("output_name_mapping");

  // TODO(Superjomn) replace this with a different stream
  auto *engine = Singleton<TRT_EngineManager>::Global().Create(
      max_batch, max_workspace, nullptr /*engine hold its own stream*/,
      context.Attr<std::string>("engine_uniq_key"));
  engine->InitNetwork();

  framework::BlockDesc block(nullptr /*programdesc*/, &block_desc);
  VLOG(4) << "parsed var size " << block.AllVars().size();
  // Add inputs
  VLOG(4) << "declare inputs";
  for (auto &input : context.Inputs("Xs")) {
    if (parameters.count(input)) continue;
    VLOG(4) << "declare input " << input;
    auto *var = block.FindVar(input);
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

  inference::Singleton<inference::tensorrt::OpConverter>::Global().ConvertBlock(
      block_desc, parameters, context.scope(), engine);

  // Add outputs
  for (auto &output : output_maps) {
    engine->DeclareOutput(output);
  }

  engine->FreezeNetwork();
}

class TensorRTEngineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Xs", "A list of inputs.").AsDuplicable();
    AddOutput("Ys", "A list of outputs").AsDuplicable();
    AddAttr<std::string>("subgraph", "the subgraph.");
    AddAttr<std::string>("engine_uniq_key", "unique key for the TRT engine.");
    AddAttr<int>("max_batch", "the maximum batch size.");
    AddAttr<int>("max_workspace", "the maximum batch size.");
    AddComment("TensorRT engine operator.");
  }
};

class TensorRTEngineInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(tensorrt_engine, ops::TensorRTEngineOp,
                  ops::TensorRTEngineOpMaker, ops::TensorRTEngineOpMaker);

REGISTER_OP_CPU_KERNEL(
    tensorrt_engine,
    ops::TensorRTEngineKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TensorRTEngineKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TensorRTEngineKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TensorRTEngineKernel<paddle::platform::CPUDeviceContext, int64_t>);

#endif  // PADDLE_WITH_CUDA

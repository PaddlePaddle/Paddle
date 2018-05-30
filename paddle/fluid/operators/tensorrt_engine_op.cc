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

#include "paddle/fluid/operators/tensorrt_engine_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void paddle::operators::TensorRTEngineKernel<DeviceContext, T>::Prepare(
    const framework::ExecutionContext &context) const {
  // Get the ProgramDesc and pass to convert.
  const auto &block = context.Attr<framework::proto::BlockDesc>("subgraph");
  max_batch_ = context.Attr<int>("max_batch");
  auto max_workspace = context.Attr<int>("max_workspace");
  engine_.reset(new inference::tensorrt::TensorRTEngine(
      max_batch_, max_workspace, nullptr));
  inference::Singleton<inference::tensorrt::OpConverter>::Global().ConvertBlock(
      block, engine_.get());
  engine_->FreezeNetwork();
}

class TensorRTEngineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Xs", "A list of inputs.").AsDuplicable();
    AddOutput("Ys", "A list of outputs").AsDuplicable();
    AddAttr<std::string>("subgraph", "the subgraph");
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

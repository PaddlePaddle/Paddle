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

#include "paddle/fluid/operators/dlnne/dlnne_engine_op.h"

namespace paddle {
namespace inference {

void CopyTensorDeviceToCpu(void* dst_ptr, void* src_ptr, int total_bytes) {
  cudaDeviceSynchronize();
  cudaMemcpy(dst_ptr, src_ptr, total_bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}
void CopyTensorCpuToDevice(void* dst_ptr, void* src_ptr, int total_bytes) {
  cudaDeviceSynchronize();
  cudaMemcpy(dst_ptr, src_ptr, total_bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

}  // namespace inference

namespace operators {

class DlnneEngineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Xs", "A list of inputs.").AsDuplicable();
    AddOutput("Ys", "A list of outputs").AsDuplicable();
    AddAttr<std::string>("subgraph", "the subgraph.");
    AddAttr<std::string>(
        "engine_key",
        "The engine_key here is used to distinguish different DLNNE Engines");
    AddAttr<framework::BlockDesc*>("sub_block", "the trt block");
    AddComment("Dlnne engine operator.");
  }
};

class DlnneEngineInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(dlnne_engine, ops::DlnneEngineOp, ops::DlnneEngineOpMaker);

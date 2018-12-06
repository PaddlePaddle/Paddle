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

#include "paddle/fluid/operators/tensorrt/tensorrt_engine_op.h"

namespace paddle {

DEFINE_int32(tensorrt_engine_batch_size, 1, "the batch_size of TensorRT");

namespace operators {

class TensorRTEngineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Xs", "A list of inputs.").AsDuplicable();
    AddOutput("Ys", "A list of outputs").AsDuplicable();
    AddAttr<std::string>("subgraph", "the subgraph.");
    AddAttr<std::string>("engine_uniq_key", "unique key for the TRT engine.");
    AddAttr<int>("max_batch_size", "the maximum batch size.");
    AddAttr<int>("workspace_size", "the workspace size.");
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

#endif  // PADDLE_WITH_CUDA

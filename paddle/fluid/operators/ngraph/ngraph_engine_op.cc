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

#include <string>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/ngraph/ngraph_engine_op.h"

namespace paddle {
namespace operators {

class NgraphEngineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Xs", "A list of inputs.").AsDispensable();
    AddOutput("Ys", "A list of outputs").AsDispensable();
    AddAttr<std::string>("graph", "the graph.");
    AddAttr<std::string>("engine_key", "the engine hash key.");
    AddAttr<std::vector<int>>("interval", "op interval supported by ngraph");
    AddComment("ngraph engine operator.");
  }
};

class NgraphEngineInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(ngraph_engine, ops::NgraphEngineOp, ops::NgraphEngineOpMaker);
REGISTER_OP_CPU_KERNEL(
    ngraph_engine,
    ops::NgraphEngineKernel<paddle::platform::CPUDeviceContext, float>);

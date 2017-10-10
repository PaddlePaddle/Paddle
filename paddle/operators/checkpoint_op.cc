/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/op_registry.h"

#include <fstream>

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::LoDTensor;

class CheckpointOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Step"),
                   "Input(Step) of Checkpoint should not be null.");
    std::string absolutePath = ctx->Attrs().Get<std::string>("absolutePath");
    PADDLE_ENFORCE(absolutePath != "",
                   "Input(absolutePath) of Checkpoint should not be null.");
  }
};

class CheckpointOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CheckpointOpMaker(framework::OpProto* proto,
                    framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Step", "(Tensor), indicate the training process.");
    AddAttr<std::string>("absolutePath", "the absolutePath for save model.");
    AddAttr<int>(
        "everyNStep",
        "(int, default 0) interval for saving checkpoint in every n step."
        "0 means only save checkpoint once")
        .SetDefault(0);
    AddComment(R"DOC(
Save the workload environment to the absolute path.

All the tensors can carry the LoD (Level of Details) information,
or not.
)DOC");
  }
};

template <typename T>
class CheckpointKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::string absolutePath = ctx.template Attr<std::string>("absolutePath");
    // TODO(dzh) : checkpoint kernel need executor support :
    // 1. operator should support asynchronously running
    // 2. checkpoint op need at least two thread.
    //    Because checkpoint will happen every, so need a thread wait
    //    the timer/steps to reach the condition.
    auto* Step = ctx.Input<Tensor>("Step");
    const int* curr_step = Step->data<int>();
    int every_n_step = ctx.template Attr<int>("everyNStep");
    auto& scope = ctx.scope();

    if (every_n_step == 0 && !run_once) {
      run_once = true;
    }

    if (run_once || *curr_step % every_n_step != 0) return;

    std::vector<std::string> ins = scope.GetAllNames();
    std::vector<framework::Variable*> inputs;
    for (auto& name : ins) {
      inputs.emplace_back(scope.FindVar(name));
    }

    std::ofstream fout(absolutePath, std::fstream::app);
    PADDLE_ENFORCE(!fout.is_open(), "open file for model failed.");
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::string bytes = inputs[i]->Get<LoDTensor>().SerializeToString();
      fout << bytes << '\n';
    }
    fout.close();
  }

  // flag indicate this op may be skipped.
  mutable bool run_once = false;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(checkpoint, ops::CheckpointOp,
                             ops::CheckpointOpMaker);
REGISTER_OP_CPU_KERNEL(checkpoint, ops::CheckpointKernel<float>);

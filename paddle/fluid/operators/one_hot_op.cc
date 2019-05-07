//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/one_hot_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace operators {

class OneHotOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of OneHotOp should not be null.");
    // PADDLE_ENFORCE(ctx->HasInput("depth"),
    //               "Input(depth) of depth should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of OneHotOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "Rank of Input(X) should be at least 2.");
    if (ctx->IsRuntime() || x_dims[x_dims.size() - 1] > 0) {
      PADDLE_ENFORCE_GE(x_dims[x_dims.size() - 1], 1U,
                        "Last dimension of Input(X) should be 1.");
    }

    framework::DDim out_dims(x_dims);
    int depth = -1;
    if (!ctx->HasInput("depth")) {
      depth = ctx->Attrs().Get<int>("depth_attr");
    }

    auto remain_cpu_name_list =
        ctx->Attrs().Get<std::vector<std::string>>("remain_cpu_name_list");

    for (size_t i = 0; i < remain_cpu_name_list.size(); ++i) {
      LOG(ERROR) << "remain cpu name" << remain_cpu_name_list[i];
    }

    out_dims[out_dims.size() - 1] = depth;
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", /* --> */ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class OneHotOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, LoDTensor<int>) Input variable with rank at least 2. "
             "The last dimension of X should be 1. Each value of X is an index "
             "to indicate the position.");
    AddInput("depth", "(Tensor, Tensor<int>), Length of one-hot vector")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor, Tensor<float>) Output tensor with same rank as X. "
              "The tensor consists of one-hot representations of values in X.");

    AddAttr<int>("depth_attr",
                 "A positive integer to specify the length of one-hot vector.")
        .SetDefault(-1);
    AddAttr<std::vector<std::string>>("remain_cpu_name_list",
                                      "var name list remain cpu")
        .SetDefault(std::vector<std::string>());
    AddAttr<int>("dtype",
                 "An integer to specify the data type of one-hot "
                 "vector. The default value is FP32.")
        .SetDefault(paddle::framework::proto::VarType::FP32);
    AddComment(R"DOC(
One Hot Operator. This operator creates the one-hot representations for input
index values. The following example will help to explain the function of this
operator:

X is a LoDTensor:
  X.lod = [[0, 1, 4]]
  X.shape = [4, 1]
  X.data = [[1], [1], [3], [0]]

set depth = 4

Out is a LoDTensor:
  Out.lod = [[0, 1, 4]]
  Out.shape = [4, 4]
  Out.data = [[0., 1., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 0., 1.],
              [1., 0., 0., 0.]]
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(one_hot, ops::OneHotOp, ops::OneHotOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    one_hot, ops::OneHotKernel<paddle::platform::CPUDeviceContext, int>,
    ops::OneHotKernel<paddle::platform::CPUDeviceContext, int64_t>);

// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/uniform_random_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/generator.h"

namespace paddle {
namespace operators {

class RandintOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"),
        true,
        platform::errors::InvalidArgument("Output(Out) of RandintOp is null."));
    PADDLE_ENFORCE_LT(
        ctx->Attrs().Get<int>("low"),
        ctx->Attrs().Get<int>("high"),
        platform::errors::InvalidArgument("randint's low must less then high, "
                                          "but received: low = %d, high = %d.",
                                          ctx->Attrs().Get<int>("low"),
                                          ctx->Attrs().Get<int>("high")));

    if (ctx->HasInputs("ShapeTensorList")) {
      // top prority shape
      auto inputs_name = ctx->Inputs("ShapeTensorList");
      PADDLE_ENFORCE_GT(
          inputs_name.size(),
          0,
          platform::errors::InvalidArgument(
              "Input(ShapeTensorList)'size of Op(randint) can't be zero."
              "Please check the Attr(shape)'s size of"
              "Op(fluid.layers.randint).)"));
      auto out_dims = std::vector<int>(inputs_name.size(), -1);
      ctx->SetOutputDim("Out", phi::make_ddim(out_dims));

      return;
    }

    auto& shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    if (ctx->HasInput("ShapeTensor") && shape.empty()) {
      auto shape_dims = ctx->GetInputDim("ShapeTensor");
      PADDLE_ENFORCE_EQ(shape_dims.size(),
                        1,
                        platform::errors::InvalidArgument(
                            "ShapeError: Input(ShapeTensor)' dimension size of "
                            "Op(randint) must be 1."
                            "But received ShapeTensor's dimensions = %d.",
                            shape_dims.size()));
      int num_ele = 1;
      for (int i = 0; i < shape_dims.size(); ++i) {
        num_ele *= shape_dims[i];
      }
      auto vec_dims = std::vector<int64_t>(num_ele, -1);
      auto out_dims = phi::make_ddim(vec_dims);
      ctx->SetOutputDim("Out", out_dims);
      return;
    }

    std::vector<int64_t> tensor_shape;
    tensor_shape.reserve(shape.size());
    for (auto dim : shape) {
      tensor_shape.push_back(static_cast<int64_t>(dim));
    }
    ctx->SetOutputDim("Out", phi::make_ddim(tensor_shape));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class RandintOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("ShapeTensor",
             "(Tensor<int64_t> or Tensor<int32_t>, optional) . If provided, "
             "randint"
             "according to "
             "this given shape. It means that it has a higher priority than "
             "Attr(shape) but a lower priority than Input(ShapeTensor).")
        .AsDispensable();
    AddInput("ShapeTensorList",
             "(vector<Tensor<int64_t>> or vector<Tensor<int32_t>>, optional). "
             "If provided, randint use this. The shape of the tensor "
             "must be [1], it has the highest priority comparing with "
             "Input(ShapeTensor) and attr(shape).")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out", "The output tensor of randint op");
    AddComment(R"DOC(
This operator initializes a tensor with random integers sampled from a
uniform distribution. The random result is in set [low, high).
)DOC");
    AddAttr<std::vector<int64_t>>("shape", "The shape of the output tensor.")
        .SetDefault({});
    AddAttr<int>("low",
                 "The lower bound on the range of random values to generate.");
    AddAttr<int>("high",
                 "The upper bound on the range of random values to generate.");
    AddAttr<int>("dtype", "Output tensor data type. [Default INT64].")
        .SetDefault(framework::proto::VarType::INT64);
    AddAttr<int>("seed",
                 "Random seed used for generating samples. "
                 "0 means use a seed generated by the system."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time. [default 0].")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    randint,
    ops::RandintOp,
    ops::RandintOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

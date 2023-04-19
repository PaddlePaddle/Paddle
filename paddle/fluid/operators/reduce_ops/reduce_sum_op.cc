// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

// NOTE: Input(Out) is unnecessary in reduce_sum_grad, and Input(X) needs no
// buffer

template <typename T>
class ReduceSumOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("reduce_sum_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    int in_dtype = ctx.Attr<int>("out_dtype");
    if (in_dtype >= 0) {
      return phi::KernelKey(
          static_cast<framework::proto::VarType::Type>(in_dtype),
          ctx.GetPlace());
    }
    return phi::KernelKey(framework::OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Out")),
                          ctx.GetPlace());
  }
};

class ReduceSumCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
 public:
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;
  void Apply() override {
    // get inputs
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");

    // get attr
    std::vector<int> axis = this->Attr<std::vector<int>>("dim");
    bool keep_dim = this->Attr<bool>("keep_dim");
    bool reduce_all = this->Attr<bool>("reduce_all");
    // get output
    paddle::Tensor x_grad_t = this->GetSingleInputGrad("X");

    // get output ptr
    paddle::Tensor* x_grad = this->GetOutputPtr(&x_grad_t);

    // get output orginal name
    std::string x_grad_name = this->GetOutputName(x_grad_t);
    VLOG(6) << "Runing sum_grad composite func";
    // call composite backward func
    prim::sum_grad<prim::DescTensor>(
        x, out_grad, axis, keep_dim, reduce_all, x_grad);
    // recover output name
    this->RecoverOutputName(x_grad_t, x_grad_name);
  }
};

template <typename T>
class ReduceSumDoubleOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    op->SetAttrMap(this->Attrs());
    op->SetType("reduce_sum");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ReduceSumGradNoNeedBufferVarInferer, "X");
class ReduceSumVarTypeInference : public paddle::framework::VarTypeInference {
 public:
  void operator()(paddle::framework::InferVarTypeContext* ctx) const override {
    auto data_type = static_cast<paddle::framework::proto::VarType::Type>(
        PADDLE_GET_CONST(int, ctx->GetAttr("out_dtype")));
    if (data_type >= 0) {
      ctx->SetOutputDataType("Out", data_type);
    } else {
      auto x_type = ctx->GetInputDataType("X");
      if (x_type == framework::proto::VarType::BOOL ||
          x_type == framework::proto::VarType::INT32) {
        ctx->SetOutputDataType("Out", framework::proto::VarType::INT64);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

class ReduceSumOpMaker : public ops::ReduceBaseOpMaker {
 protected:
  virtual std::string GetName() const { return "reduce_sum"; }
  virtual std::string GetOpType() const { return "Reduce reduce_sum"; }
};

DECLARE_INFER_SHAPE_FUNCTOR(reduce_sum,
                            ReduceSumInferShapeFunctor,
                            PD_INFER_META(phi::SumRawInferMeta));

REGISTER_OPERATOR(reduce_sum,
                  ops::ReduceBaseOp,
                  ReduceSumOpMaker,
                  ops::ReduceSumVarTypeInference,
                  ops::ReduceSumOpGradMaker<paddle::framework::OpDesc>,
                  ops::ReduceSumOpGradMaker<paddle::imperative::OpBase>,
                  ops::ReduceSumCompositeGradOpMaker,
                  ReduceSumInferShapeFunctor);
REGISTER_OPERATOR(reduce_sum_grad,
                  ops::ReduceGradOp,
                  ops::ReduceSumDoubleOpGradMaker<paddle::framework::OpDesc>,
                  ops::ReduceSumDoubleOpGradMaker<paddle::imperative::OpBase>,
                  ops::ReduceSumGradNoNeedBufferVarInferer);

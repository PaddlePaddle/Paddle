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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class AsComplexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class AsComplexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of view_as_complex op.");
    AddOutput("Out", "(Tensor), The output tensor of view_as_complex op.");
    AddComment(R"DOC(
As_complex Operator.

This operator is used to return a complex tensor represented
by an old-fashioned real tensor. The size of the last dimension of
the input tensor should be 2, which corresponds to 'real' and
'complex', respectively.

)DOC");
  }
};

template <typename T>
class AsComplexGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("as_real");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput("Out", this->InputGrad("X"));
  }
};

class AsRealOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class AsRealOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of as_real op.");
    AddOutput("Out", "(Tensor), The output tensor of as_real op.");
    AddComment(R"DOC(
AsReal Operator.

This operator is used to return an old-fashioned real tensor from a
complex tensor. The size of the last dimension of the output tensor is 2,
which corresponds to 'real' and 'complex', respectively.

)DOC");
  }
};

template <typename T>
class AsRealGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("as_complex");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput("Out", this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(as_real,
                            AsRealInferShapeFunctor,
                            PD_INFER_META(phi::AsRealInferMeta));

REGISTER_OPERATOR(as_real,
                  ops::AsRealOp,
                  ops::AsRealOpMaker,
                  AsRealInferShapeFunctor,
                  ops::AsRealGradMaker<paddle::framework::OpDesc>,
                  ops::AsRealGradMaker<paddle::imperative::OpBase>);

DECLARE_INFER_SHAPE_FUNCTOR(as_complex,
                            AsComplexInferShapeFunctor,
                            PD_INFER_META(phi::AsComplexInferMeta));

REGISTER_OPERATOR(as_complex,
                  ops::AsComplexOp,
                  ops::AsComplexOpMaker,
                  AsComplexInferShapeFunctor,
                  ops::AsComplexGradMaker<paddle::framework::OpDesc>,
                  ops::AsComplexGradMaker<paddle::imperative::OpBase>);

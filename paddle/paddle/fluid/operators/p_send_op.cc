/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
class PSendOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
};

class PSendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor), input 0 of send op.");
    AddAttr<int>("ring_id", "(int), attribute 0 for send op.").SetDefault(0);
    AddAttr<int>("peer", "(int), attribute 1 for send op.").SetDefault(0);
    AddAttr<bool>("dynamic_shape", "(bool), attribute 2 for send op.")
        .SetDefault(false);
    AddComment(R"DOC(
                    TODO: Documentation of send op.
                    )DOC");
  }
};

class PSendArrayOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
};

class PSendArrayOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor[]), input 0 of p_send_array op.").AsDuplicable();
    AddAttr<int>("ring_id", "(int), attribute 0 for p_send_array op.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int), attribute 1 for p_send_array op.")
        .SetDefault(0);
    AddComment(R"DOC(
    TODO: Documentation of p_send_array op.
    )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(p_send,
                            PSendInferShapeFunctor,
                            PD_INFER_META(phi::PSendInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(p_send_array,
                            PSendArrayInferShapeFunctor,
                            PD_INFER_META(phi::PSendArrayInferMeta));

REGISTER_OPERATOR(
    p_send,
    ops::PSendOp,
    ops::PSendOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    PSendInferShapeFunctor);

REGISTER_OPERATOR(
    p_send_array,
    ops::PSendArrayOp,
    ops::PSendArrayOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    PSendArrayInferShapeFunctor);

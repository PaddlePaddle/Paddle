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
class SendV3Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
};

class SendV3OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of send op.");
    AddAttr<int>("ring_id", "(int), attribute 0 for send op.").SetDefault(0);
    AddAttr<int>("peer", "(int), attribute 1 for send op.").SetDefault(0);
    AddAttr<bool>("dynamic_shape", "(bool), attribute 2 for send op.")
        .SetDefault(false);
    AddComment(R"DOC(
                    TODO: Documentation of send op.
                    )DOC");
  }
};

class SendV3ArrayOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
};

class SendV3ArrayOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor[]), input 0 of send_v3_array op.").AsDuplicable();
    AddOutput("out", "(Tensor), output 0 of send_v3_array op.");
    AddAttr<int>("ring_id", "(int), attribute 0 for send_v3_array op.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int), attribute 1 for send_v3_array op.")
        .SetDefault(0);
    AddComment(R"DOC(
    TODO: Documentation of send_v3_array op.
    )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(send_v3,
                            SendV3InferShapeFunctor,
                            PD_INFER_META(phi::SendV3InferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(send_v3_array,
                            SendV3ArrayInferShapeFunctor,
                            PD_INFER_META(phi::SendV3ArrayInferMeta));

REGISTER_OPERATOR(
    send_v3,
    ops::SendV3Op,
    ops::SendV3OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    SendV3InferShapeFunctor);

REGISTER_OPERATOR(
    send_v3_array,
    ops::SendV3ArrayOp,
    ops::SendV3ArrayOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    SendV3ArrayInferShapeFunctor);

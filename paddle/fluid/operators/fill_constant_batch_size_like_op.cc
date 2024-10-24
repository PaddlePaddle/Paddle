/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using MetaTensor = framework::CompatMetaTensor;

class BatchSizeLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", Type());
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", Type());

    MetaTensor x(ctx->GetInputVarPtrs("Input")[0], ctx->IsRuntime());
    MetaTensor out(ctx->GetOutputVarPtrs("Out")[0], ctx->IsRuntime());
    auto& shape = ctx->Attrs().Get<std::vector<int>>("shape");
    int x_batch_size_dim = ctx->Attrs().Get<int>("input_dim_idx");
    int out_batch_size_dim = ctx->Attrs().Get<int>("output_dim_idx");
    phi::BatchSizeLikeInferMeta(
        x, shape, x_batch_size_dim, out_batch_size_dim, &out);
  }
};

class BatchSizeLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput(
        "Input",
        "Tensor whose input_dim_idx'th dimension specifies the batch_size");
    AddOutput("Out",
              "Tensor of specified shape will be filled "
              "with the specified value");
    AddAttr<std::vector<int>>("shape", "The shape of the output");
    AddAttr<int>("input_dim_idx",
                 "default 0. The index of input's batch size dimension")
        .SetDefault(0);
    AddAttr<int>("output_dim_idx",
                 "default 0. The index of output's batch size dimension")
        .SetDefault(0);
    Apply();
  }

 protected:
  virtual void Apply() = 0;
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(BatchSizeLikeNoNeedBufferVarsInferer,
                                    "Input");

class FillConstantBatchSizeLikeOp : public BatchSizeLikeOp {
 protected:
  using BatchSizeLikeOp::BatchSizeLikeOp;
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    phi::KernelKey kernel_type = phi::KernelKey(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
    if (ctx.Attr<bool>("force_cpu")) {
      kernel_type.set_backend(phi::Backend::CPU);
    }
    return kernel_type;
  }
};

class FillConstantBatchSizeLikeOpMaker : public BatchSizeLikeOpMaker {
 protected:
  void Apply() override {
    AddAttr<int>(
        "dtype",
        "It could be numpy.dtype. Output data type. Default is float32")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<float>("value", "default 0. The value to be filled")
        .SetDefault(0.0f);
    AddAttr<std::string>("str_value", "default empty. The value to be filled")
        .SetDefault("");
    AddAttr<bool>("force_cpu",
                  "(bool, default false) Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device")
        .SetDefault(false);
    AddComment(R"DOC(
This function creates a tensor of specified *shape*, *dtype* and batch size,
and initializes this with a constant supplied in *value*. The batch size is
obtained from the `input` tensor.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(fill_constant_batch_size_like,
                            FillConstantBatchSizeLikeInferShapeFunctor,
                            PD_INFER_META(phi::FullBatchSizeLikeInferMeta));
REGISTER_OPERATOR(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::FillConstantBatchSizeLikeOpMaker,
    ops::BatchSizeLikeNoNeedBufferVarsInferer,
    FillConstantBatchSizeLikeInferShapeFunctor);

/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/embedding_bag_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/platform/bfloat16.h"


namespace paddle {
namespace operators{

class EmbeddingBagOpMaker : public framework::OpProtoAndCheckerMaker  {
    public:
    void Make() override {
        AddInput("input",
                 "(Tensor) The input is a 2-D tensor,"
                 "which represents indecies of bags.");
        AddInput("params",
                 "(Tensor) The params represents embedding tensors"
                 "which is a learnable parameter.");
        AddInput("weight",
                 "(Tensor) When mode is 'sum',"
                 "weights represents summed weights. " );
        AddOutput("out", "The embeddingbag results, which have the same type as params.");
        AddAttr<std::string>("mode",
                             "mode of embeddingbag, containing 'sum' and 'mean'")
                .SetDefault("sum");
        
   
        AddComment(R"DOC(
                   EmbeddingBag Operator.
        )DOC");

    }
};


template <typename T>
class EmbeddingBagGradOpMaker : public framework::SingleGradOpMaker<T> {
    public:
        using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
    
    protected:
        void Apply(GradOpPtr<T> op) const override {
            op -> SetType("embedding_bag_grad");

            op -> SetInput("input", this -> Input("input"));
            op -> SetInput("params", this -> Input("params"));
            op -> SetInput("weight", this -> Input("weight"));
            op -> SetInput(framework::GradVarName("out"), this -> OutputGrad("out"));

            op -> SetOutput(framework::GradVarName("params"), this->InputGrad("params"));
            op -> SetOutput(framework::GradVarName("weight"), this->InputGrad("weight"));

            op -> SetAttrMap(this -> Attrs());
        }
};



class EmbeddingBagOp : public framework::OperatorWithKernel {
    public:
        using framework::OperatorWithKernel::OperatorWithKernel;
    
        void InferShape(framework::InferShapeContext* ctx) const override {
            OP_INOUT_CHECK(ctx->HasInput("input"), "Input", "input", "EmbeddingBag");
            OP_INOUT_CHECK(ctx->HasInput("params"), "Input", "params", "EmbeddingBag");
            OP_INOUT_CHECK(ctx->HasInput("weight"), "Input", "weight", "EmbeddingBag");
            OP_INOUT_CHECK(ctx->HasOutput("out"), "Output", "out", "EmbeddingBag");

            auto table_dims = ctx->GetInputDim("params");
            auto ids_dims = ctx->GetInputDim("input");
            auto weight_dims = ctx->GetInputDim("weight");

            PADDLE_ENFORCE_EQ(
                ids_dims, weight_dims,
                platform::errors::InvalidArgument(
                    "ShapeError: The shapes of the 'input' and 'weight' must be the same."
                    "But received input's shape = [%s],"
                    "weight's shape = [%s].", ids_dims, weight_dims
                )
            );
            PADDLE_ENFORCE_EQ(
                ids_dims.size(), 2,
                platform::errors::InvalidArgument(
                    "ShapeError: The dimensions of the 'input' tensor must be 2."
                    "But received input's dimensions = %d, input's shape = [%s].",
                    ids_dims.size(), ids_dims
                )
            );


            auto output_dims = phi::vectorize(phi::slice_ddim(ids_dims, 0 ,ids_dims.size()-1));
            output_dims.push_back(table_dims[1]);
            ctx -> SetOutputDim("out", phi::make_ddim(output_dims));


        }
        
        protected:
            framework::OpKernelType GetExpectedKernelType(
                const framework::ExecutionContext& ctx) const override {
                    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "params");
                    return framework::OpKernelType(data_type, ctx.device_context());
                }
            

};

class EmbeddingBagOpGrad : public framework::OperatorWithKernel {
    public:
        using framework::OperatorWithKernel::OperatorWithKernel;

    protected:
    void InferShape(framework::InferShapeContext* ctx) const override {
        auto table_dims = ctx->GetInputDim("params");
        auto weight_dims = ctx->GetInputDim("weight");
        ctx -> SetOutputDim(framework::GradVarName("params"), table_dims);
        ctx -> SetOutputDim(framework::GradVarName("weight"), weight_dims);
    }

    protected:
        framework::OpKernelType GetExpectedKernelType(const framework::ExecutionContext& ctx) {
            auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, framework::GradVarName("out"));
            return framework::OpKernelType(data_type, ctx.device_context());
        } 
        
};


}// namespace operators



}// namespace paddle



namespace ops = paddle::operators;

REGISTER_OPERATOR(embedding_bag, ops::EmbeddingBagOp, ops::EmbeddingBagOpMaker,
                  ops::EmbeddingBagGradOpMaker<paddle::framework::OpDesc>,
                  ops::EmbeddingBagGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(embedding_bag_grad, ops::EmbeddingBagOpGrad);

using CPU = phi::CPUContext;

REGISTER_OP_CPU_KERNEL(embedding_bag, ops::EmbeddingBagKernel<float>,
                       ops::EmbeddingBagKernel<double>, ops::EmbeddingBagKernel<int8_t>,
                       ops::EmbeddingBagKernel<int16_t>, ops::EmbeddingBagKernel<paddle::platform::bfloat16>);

REGISTER_OP_CPU_KERNEL(embedding_bag_grad,ops::EmbeddingBagGradKernel<float>,
                       ops::EmbeddingBagGradKernel<double>,
                       ops::EmbeddingBagGradKernel<paddle::platform::bfloat16>);

REGISTER_OP_VERSION(embedding_bag)
    .AddCheckpoint(
        R"ROC(
            Design embeddingbag op 
        )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .BugfixWithBehaviorChanged("upgrade the embeddingbag")
    );
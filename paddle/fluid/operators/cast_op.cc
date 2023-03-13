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

#include <memory>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"

#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"
#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#endif
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

namespace paddle {
namespace operators {

class CastOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of cast op");
    AddOutput("Out", "The output tensor of cast op");
    AddAttr<int>("out_dtype", "output data type");
    AddAttr<int>("in_dtype", "input data type");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
Cast Operator.

This Operator casts the input tensor to another data type and
returns the Output phi::DenseTensor. It's meaningless if the output dtype equals
the input dtype, but it's fine if you do so.

)DOC");
  }
};

template <typename T>
class CastOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType("cast");
    grad->SetInput("X", this->OutputGrad("Out"));
    grad->SetOutput("Out", this->InputGrad("X"));
    grad->SetAttr("out_dtype", this->GetAttr("in_dtype"));
    grad->SetAttr("in_dtype", this->GetAttr("out_dtype"));
    grad->SetAttr("use_mkldnn", this->GetAttr("use_mkldnn"));
  }
};

class CastCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
 public:
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

  void Apply() override {
    paddle::Tensor out_grad = paddle::Tensor(
        std::make_shared<prim::DescTensor>(this->SingleOutputGrad("Out")));
    paddle::Tensor x_grad = paddle::Tensor(
        std::make_shared<prim::DescTensor>(this->SingleInputGrad("X")));
    auto dx_ptr = this->GetOutputPtr(&x_grad);
    std::string dx_name = this->GetOutputName(x_grad);
    auto dtype = phi::TransToPhiDataType((this->Attr<int>("in_dtype")));
    prim::cast_grad<prim::DescTensor>(out_grad, dtype, dx_ptr);
    this->RecoverOutputName(x_grad, dx_name);
  }
};

class CastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // CastOp kernel's device type is decided by input tensor place
    auto *tensor = ctx.Input<phi::DenseTensor>("X");
    PADDLE_ENFORCE_EQ(tensor->IsInitialized(),
                      true,
                      platform::errors::PreconditionNotMet(
                          "The tensor of Input(X) is not initialized."));
    auto &tensor_place = tensor->place();
    // NOTE: cuda pinned tensor need to copy its data to target place
    if (platform::is_cuda_pinned_place(tensor_place)) {
      return phi::KernelKey(framework::TransToProtoVarType(tensor->dtype()),
                            ctx.device_context().GetPlace());
    }

    // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_MKLDNN
    int in_dtype = ctx.Attr<int>("in_dtype");
    int out_dtype = ctx.Attr<int>("out_dtype");

    int dtype_fp32 = static_cast<int>(framework::proto::VarType::FP32);
    int dtype_bf16 = static_cast<int>(framework::proto::VarType::BF16);

    if ((in_dtype != dtype_fp32 && in_dtype != dtype_bf16) ||
        (out_dtype != dtype_fp32 && out_dtype != dtype_bf16)) {
      this->SetDnnFallback(true);
    }
    // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_MKLDNN

#ifdef PADDLE_WITH_MLU
    auto src_type = static_cast<VT::Type>(ctx.Attr<int>("in_dtype"));
    auto dst_type = static_cast<VT::Type>(ctx.Attr<int>("out_dtype"));
    if (src_type == dst_type || MLUSupportsCast(src_type, dst_type)) {
      return phi::KernelKey(framework::TransToProtoVarType(tensor->dtype()),
                            tensor_place);
    } else {
      VLOG(3) << "MLU not support cast type: "
              << framework::DataTypeToString(src_type)
              << " to type: " << framework::DataTypeToString(dst_type)
              << ", fallbacking to CPU one!";
      return phi::KernelKey(framework::TransToProtoVarType(tensor->dtype()),
                            platform::CPUPlace());
    }
#endif
    return phi::KernelKey(framework::TransToProtoVarType(tensor->dtype()),
                          tensor_place);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = phi::CPUContext;

DECLARE_INFER_SHAPE_FUNCTOR(cast,
                            CastInferShapeFunctor,
                            PD_INFER_META(phi::CastInferMeta));

// cast use phi kernel, so no need to REGISTER_OP_CPU_KERNEL here.
REGISTER_OPERATOR(cast,
                  ops::CastOp,
                  ops::CastOpGradMaker<paddle::framework::OpDesc>,
                  ops::CastOpGradMaker<paddle::imperative::OpBase>,
                  ops::CastCompositeGradOpMaker,
                  ops::CastOpProtoMaker,
                  CastInferShapeFunctor);

// [ why register transfer_dtype_op alias with cast_op? ]
// In case of InterpreterCore, if we reuse cast_op, we cannot distinguish
// which cast_op is inserted by new executor when we do profiling.
REGISTER_OPERATOR(transfer_dtype,
                  ops::CastOp,
                  ops::CastOpGradMaker<paddle::framework::OpDesc>,
                  ops::CastOpGradMaker<paddle::imperative::OpBase>,
                  ops::CastOpProtoMaker,
                  CastInferShapeFunctor);

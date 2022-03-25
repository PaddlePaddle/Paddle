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

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/phi_utils.h"

// only can include the headers in paddle/phi/api dirs
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/reshape_grad_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class ReshapeOp : public framework::OperatorWithKernel {
 public:
  ReshapeOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of ReshapeOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of ReshapeOp should not be null."));

    if (ctx->HasInputs("ShapeTensor")) {
      // top prority shape
      auto ShapeTensor = ctx->Inputs("ShapeTensor");
      PADDLE_ENFORCE_GT(
          ShapeTensor.size(), 0,
          platform::errors::InvalidArgument(
              "When `shape` in ReshapeOp is a list or tuple "
              "which contains Tensor, the shape's size can't be zero. "
              "But received shape's size is %d.",
              ShapeTensor.size()));
      auto infer_shape = ctx->Attrs().Get<std::vector<int>>("shape");
      const int64_t copy_dim_val = 0;
      auto in_dims = ctx->GetInputDim("X");
      for (size_t i = 0; i < infer_shape.size(); ++i) {
        if (infer_shape[i] == copy_dim_val) {
          PADDLE_ENFORCE_LT(
              static_cast<int>(i), in_dims.size(),
              platform::errors::InvalidArgument(
                  "The index of 0 in `shape` must be less than "
                  "the input tensor X's dimensions. But received shape[%d] "
                  "= 0, X's dimensions = %d, X's shape = [%s].",
                  i, in_dims.size(), in_dims));
          infer_shape[i] = in_dims[i];
        }
      }
      auto infer_out_dims = phi::make_ddim(infer_shape);
      ctx->SetOutputDim("Out", infer_out_dims);
      return;
    }

    const std::vector<int> &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    if (ctx->HasInput("Shape") && shape.empty()) {
      auto shape_dims = ctx->GetInputDim("Shape");
      int num_ele = 1;
      for (int i = 0; i < shape_dims.size(); ++i) {
        num_ele *= shape_dims[i];
      }
      auto vec_dims = std::vector<int>(num_ele, -1);
      auto out_dims = phi::make_ddim(vec_dims);
      ctx->SetOutputDim("Out", out_dims);
      ctx->ShareLoD("X", /*->*/ "Out");
      return;
    }

    if (ctx->HasInput("Shape") && !shape.empty() && ctx->IsRuntime()) {
      // If true, set the shape of Output(Out) according to Input(Shape) in
      // ReshapeKernel with ExecutionContext. Also check LoD in ReshapeKernel.
      ctx->ShareLoD("X", /*->*/ "Out");
      return;
    }

    PADDLE_ENFORCE_EQ(!shape.empty(), true,
                      platform::errors::InvalidArgument(
                          "The parameter 'shape' in ReshapeOp must be set. "
                          "But received 'shape' is empty."));
    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = ValidateShape(shape, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

  static framework::DDim ValidateShape(const std::vector<int> shape,
                                       const framework::DDim &in_dims) {
    const int64_t in_size = phi::product(in_dims);
    auto in_dims_vec = phi::vectorize(in_dims);
    bool all_positive = std::all_of(in_dims_vec.cbegin(), in_dims_vec.cend(),
                                    [](int64_t i) { return i > 0; });
    // only one dimension can be set to -1, whose size will be automatically
    // infered.
    const int64_t unk_dim_val = -1;
    const int64_t copy_dim_val = 0;

    std::vector<int64_t> output_shape(shape.size(), 0);
    int64_t capacity = 1;
    int unk_dim_idx = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == unk_dim_val) {
        PADDLE_ENFORCE_EQ(
            unk_dim_idx, -1,
            platform::errors::InvalidArgument(
                "Only one dimension value of 'shape' in ReshapeOp can "
                "be -1. But received shape = [%s], shape[%d] is also -1.",
                phi::make_ddim(shape), i));
        unk_dim_idx = i;
      } else if (shape[i] == copy_dim_val) {
        PADDLE_ENFORCE_LT(
            static_cast<int>(i), in_dims.size(),
            platform::errors::InvalidArgument(
                "The index of 0 in `shape` must be less than "
                "the input tensor X's dimensions. "
                "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
                "X's dimensions = %d.",
                phi::make_ddim(shape), i, in_dims, in_dims.size()));
      } else {
        PADDLE_ENFORCE_GT(
            shape[i], 0,
            platform::errors::InvalidArgument(
                "Each dimension value of 'shape' in ReshapeOp must not "
                "be negative except one unknown dimension. "
                "But received  shape = [%s], shape[%d] = %d.",
                phi::make_ddim(shape), i, shape[i]));
      }

      // NOTE all non-zero values will be converted to True (include negative
      // value)
      capacity *= (shape[i] ? shape[i] : in_dims[i]);
      output_shape[i] =
          (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
    }

    if (unk_dim_idx != -1) {
      if (all_positive) {
        // in_size < 0 and is un-determinate in compile time, skip the check,
        // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
        // capacity = -24, in_size = -8, output_shape[0] = 0
        // the following check will fail.
        output_shape[unk_dim_idx] = -in_size / capacity;
        PADDLE_ENFORCE_EQ(
            output_shape[unk_dim_idx] * capacity, -in_size,
            platform::errors::InvalidArgument(
                "The 'shape' attribute in ReshapeOp is invalid. "
                "The input tensor X'size must be divisible by known "
                "capacity of 'shape'. "
                "But received X's shape = [%s], X's size = %d, "
                "'shape' is [%s], known capacity of 'shape' is %d.",
                in_dims, in_size, phi::make_ddim(shape), capacity));
      } else {
        output_shape[unk_dim_idx] = -1;
      }
    } else {
      if (all_positive) {
        PADDLE_ENFORCE_EQ(
            capacity, in_size,
            platform::errors::InvalidArgument(
                "The 'shape' in ReshapeOp is invalid. "
                "The input tensor X'size must be equal to the capacity of "
                "'shape'. "
                "But received X's shape = [%s], X's size = %d, 'shape' is "
                "[%s], the capacity of 'shape' is %d.",
                in_dims, in_size, phi::make_ddim(shape), capacity));
      }
    }

    // support reshape with zero-input(input tensor with product(shape) == 0)
    // by now we require that if the input tensor is zero shape, the target
    // shape of output must be zero
    if (in_size == 0) {
      PADDLE_ENFORCE_LE(
          capacity, in_size,
          platform::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X's shape = [%s], X's capacity = %d."
              "But the target shape of Out is [%s],  the "
              "capacity of 'Out' is %d.",
              in_dims, in_size, phi::make_ddim(shape), capacity));
    }

    return phi::make_ddim(output_shape);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "ShapeTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of reshape operator.");
    AddInput("Shape",
             "(Tensor<int32>, optional). Target shape of reshape operator. "
             "It has a higher priority than Attr(shape) but a lower priority "
             "than Input(ShapeTensor). The Attr(shape) still should be "
             "set correctly to guarantee shape inference in compile time.")
        .AsDispensable();
    AddInput(
        "ShapeTensor",
        "(vector<Tensor<int32>>, optional). Target shape of reshape operator. "
        "It has the highest priority compare with Input(Shape) and "
        "Attr(shape)."
        "The shape of the element in vector must be [1].")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out", "(Tensor). The output tensor of reshape operator.");
    AddAttr<std::vector<int>>(
        "shape",
        "(std::vector<int>) Target shape of reshape operator."
        "It has the lowest priority compare with Input(Shape) and "
        " Input(ShapeTensor).")
        .SetDefault({});
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
Reshape Operator.

Reshape Input(X) into the shape specified by Attr(shape) or Input(Shape). The
data in Input(X) are unchanged.

Examples:

1. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [6, 8], the reshape operator will transform Input(X)
into a 2-D tensor with shape [6, 8] and leaving Input(X)'s data unchanged.

2. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [2, 3, -1, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 3, 4, 2] and leaving Input(X)'s data
unchanged. In this case, one and only dimension of Attr(shape) can be set to -1,
the value of this dimension is inferred from the total element number of
Input(X) and remaining dimensions.

3. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [-1, 0, 3, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 4, 3, 2] and leaving Input(X)'s data
unchanged. In this case, besides -1, 0 means the actual dimension value is going
to be copied from the corresponding dimension of Input(X).

Note:

1. One and only one dimension in Attr(shape) can be set -1. In this case,
the actual dimension value will be infered from the total element number of
Input(X) and remaining dimensions.

2. More than one dimensions in Attr(shape) can be set to 0, which means the real
dimension value will be copied from Input(X) at runtime. Note that the index of
0 can not exceed Rank(X). For example, Input(X) is a 3-D tensor with shape
[2, 3, 4], Attr(shape) = [2, 3, 2, 0] is an invalid input.

3. Input(Shape) has a higher priority than Attr(shape) if it is provided, while
Attr(shape) still should be set correctly to guarantee shape inference in
compile-time.

)DOC");
  }
};

class ReshapeGradOp : public framework::OperatorWithKernel {
 public:
  ReshapeGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) shouldn't be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) shouldn't be null."));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ReshapeKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *out = ctx.Output<framework::LoDTensor>("Out");
    auto *in = ctx.Input<framework::LoDTensor>("X");

    auto list_new_shape_tensor =
        ctx.MultiInput<framework::Tensor>("ShapeTensor");
    auto *shape_tensor = ctx.HasInput("Shape")
                             ? ctx.Input<framework::LoDTensor>("Shape")
                             : nullptr;
    phi::ScalarArray pt_scalar_shape;
    if (list_new_shape_tensor.size() > 0) {
      // have shape tensor
      std::vector<phi::DenseTensor> pt_vec_shape;
      for (auto &tensor : list_new_shape_tensor) {
        if (platform::is_gpu_place(tensor->place()) ||
            platform::is_xpu_place(tensor->place())) {
          framework::Tensor temp;
          paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(),
                                            &temp);
          pt_vec_shape.push_back(std::move(temp));
        } else {
          pt_vec_shape.push_back(*tensor);
        }
      }
      pt_scalar_shape = phi::ScalarArray(pt_vec_shape);
    } else if (shape_tensor) {
      phi::DenseTensor pt_shape;
      if (platform::is_gpu_place(shape_tensor->place()) ||
          platform::is_xpu_place(shape_tensor->place())) {
        framework::Tensor temp;
        paddle::framework::TensorCopySync(*shape_tensor, platform::CPUPlace(),
                                          &temp);
        pt_shape = std::move(temp);
      } else {
        pt_shape = *shape_tensor;
      }
      pt_scalar_shape = phi::ScalarArray(pt_shape);
    } else {
      auto &shape_attr = ctx.Attr<std::vector<int>>("shape");
      pt_scalar_shape = phi::ScalarArray(shape_attr);
    }
    if (platform::is_cpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::CPUDeviceContext>();
      phi::ReshapeKernel(static_cast<const phi::CPUContext &>(dev_ctx), *in,
                         pt_scalar_shape, out);
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::CUDADeviceContext>();
      phi::ReshapeKernel(static_cast<const phi::GPUContext &>(dev_ctx), *in,
                         pt_scalar_shape, out);
    }
#endif
#ifdef PADDLE_WITH_XPU
    if (platform::is_xpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::XPUDeviceContext>();
      phi::ReshapeKernel(static_cast<const phi::XPUContext &>(dev_ctx), *in,
                         pt_scalar_shape, out);
    }
#endif
  }
};

class ReshapeGradKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    d_x->mutable_data(ctx.GetPlace(), d_out->type());

    if (platform::is_cpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::CPUDeviceContext>();
      phi::ReshapeGradKernel(static_cast<const phi::CPUContext &>(dev_ctx),
                             *d_out, d_x);
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::CUDADeviceContext>();
      phi::ReshapeGradKernel(static_cast<const phi::GPUContext &>(dev_ctx),
                             *d_out, d_x);
    }
#endif
#ifdef PADDLE_WITH_XPU
    if (platform::is_xpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::XPUDeviceContext>();
      phi::ReshapeGradKernel(static_cast<const phi::XPUContext &>(dev_ctx),
                             *d_out, d_x);
    }
#endif
  }
};

class ReshapeDoubleGradKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *dd_x = ctx.Input<framework::Tensor>("DDX");
    auto *dd_out = ctx.Output<framework::Tensor>("DDOut");
    dd_out->mutable_data(ctx.GetPlace(), dd_x->type());

    if (platform::is_cpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::CPUDeviceContext>();
      phi::ReshapeDoubleGradKernel(
          static_cast<const phi::CPUContext &>(dev_ctx), *dd_x, dd_out);
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::CUDADeviceContext>();
      phi::ReshapeDoubleGradKernel(
          static_cast<const phi::GPUContext &>(dev_ctx), *dd_x, dd_out);
    }
#endif
#ifdef PADDLE_WITH_XPU
    if (platform::is_xpu_place(ctx.GetPlace())) {
      auto &dev_ctx = ctx.device_context<platform::XPUDeviceContext>();
      phi::ReshapeDoubleGradKernel(
          static_cast<const phi::XPUContext &>(dev_ctx), *dd_x, dd_out);
    }
#endif
  }
};

// FIXME(zcd): reshape2 adds an intermediate output(XShape) based on reshape,
// the XShape is used to carry the shape and lod of X which will be used in
// reshape_grad, in this way, the framework can reuse the memory of X
// immediately the reshape_op is finished.
// Considering compatibility issues, we could not fix reshape_op
class Reshape2Op : public ReshapeOp {
 public:
  Reshape2Op(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : ReshapeOp(type, inputs, outputs, attrs) {}
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasOutput("XShape"), true,
                      platform::errors::InvalidArgument(
                          "Output(XShape) of ReshapeOp should not be null."));
    const auto &x_dims = ctx->GetInputDim("X");
    std::vector<int64_t> xshape_dims(x_dims.size() + 1);
    xshape_dims[0] = 0;
    for (int i = 0; i < x_dims.size(); ++i) {
      xshape_dims[i + 1] = x_dims[i];
    }
    ctx->SetOutputDim("XShape", phi::make_ddim(xshape_dims));
    ctx->ShareLoD("X", /*->*/ "XShape");

    ReshapeOp::InferShape(ctx);
  }
};

class Reshape2OpMaker : public ReshapeOpMaker {
 public:
  void Make() override {
    ReshapeOpMaker::Make();
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in FlattenGradOp.")
        .AsIntermediate();
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"})
        .AsExtra();
  }
};

template <typename T>
class Reshape2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("reshape2_grad");
    grad_op->SetInput("XShape", this->Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class Reshape2DoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("reshape2_grad_grad");
    grad_op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    grad_op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class Reshape2GradOp : public framework::OperatorWithKernel {
 public:
  Reshape2GradOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("XShape"), true,
        platform::errors::InvalidArgument("Input(XShape) shouldn't be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) shouldn't be null."));

    // Construct MetaTensor for InferMeta Func
    using CompatMetaTensor = framework::CompatMetaTensor;
    CompatMetaTensor xshape(ctx->GetInputVarPtrs("XShape")[0],
                            ctx->IsRuntime());
    CompatMetaTensor dx(ctx->GetOutputVarPtrs(framework::GradVarName("X"))[0],
                        ctx->IsRuntime());
    phi::KernelWithXShapeInferMeta(xshape, &dx);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "ShapeTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class Reshape2DoubleGradOp : public framework::OperatorWithKernel {
 public:
  Reshape2DoubleGradOp(const std::string &type,
                       const framework::VariableNameMap &inputs,
                       const framework::VariableNameMap &outputs,
                       const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "DDX"),
        ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "ShapeTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

DECLARE_INPLACE_OP_INFERER(ReshapeOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(ReshapeGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_INPLACE_OP_INFERER(ReshapeDoubleGradInplaceInferer, {"DDX", "DDOut"});
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ReshapeDoubleGradOpNoNeedBufferVarInferer,
                                    "DOut");

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    reshape, ops::ReshapeOp, ops::ReshapeOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>,
    ops::ReshapeOpInplaceInferer);
REGISTER_OPERATOR(reshape_grad, ops::ReshapeGradOp,
                  ops::ReshapeGradInplaceInferer);

REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape, float, ops::ReshapeKernel, double,
                               ops::ReshapeKernel, int16_t, ops::ReshapeKernel,
                               int, ops::ReshapeKernel, int64_t,
                               ops::ReshapeKernel);
REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape_grad, float, ops::ReshapeGradKernel,
                               double, ops::ReshapeGradKernel, int16_t,
                               ops::ReshapeGradKernel, int,
                               ops::ReshapeGradKernel, int64_t,
                               ops::ReshapeGradKernel);

REGISTER_OPERATOR(reshape2, ops::Reshape2Op, ops::Reshape2OpMaker,
                  ops::Reshape2GradMaker<paddle::framework::OpDesc>,
                  ops::Reshape2GradMaker<paddle::imperative::OpBase>,
                  ops::ReshapeOpInplaceInferer);
REGISTER_OPERATOR(reshape2_grad, ops::Reshape2GradOp,
                  ops::Reshape2DoubleGradMaker<paddle::framework::OpDesc>,
                  ops::Reshape2DoubleGradMaker<paddle::imperative::OpBase>,
                  ops::ReshapeGradInplaceInferer);

DECLARE_INFER_SHAPE_FUNCTOR(reshape2_grad_grad,
                            Reshape2DoubleGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralUnaryGradInferMeta));

REGISTER_OPERATOR(reshape2_grad_grad, ops::Reshape2DoubleGradOp,
                  ops::ReshapeDoubleGradInplaceInferer,
                  ops::ReshapeDoubleGradOpNoNeedBufferVarInferer,
                  Reshape2DoubleGradInferShapeFunctor);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape, float, ops::ReshapeKernel, double,
                                ops::ReshapeKernel, int16_t, ops::ReshapeKernel,
                                int, ops::ReshapeKernel, uint8_t,
                                ops::ReshapeKernel, int64_t, ops::ReshapeKernel,
                                plat::float16, ops::ReshapeKernel,
                                plat::bfloat16, ops::ReshapeKernel);
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape_grad, float, ops::ReshapeGradKernel,
                                double, ops::ReshapeGradKernel, int16_t,
                                ops::ReshapeKernel, int, ops::ReshapeGradKernel,
                                int64_t, ops::ReshapeGradKernel, uint8_t,
                                ops::ReshapeGradKernel, plat::float16,
                                ops::ReshapeGradKernel, plat::bfloat16,
                                ops::ReshapeGradKernel);
#endif

//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/matmul_v2_op.h"
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"

namespace paddle {
namespace operators {

static framework::DDim GetDimForInput(const framework::InferShapeContext& ctx,
                                      const std::string input_name) {
  auto shape = ctx.Attrs().Get<std::vector<int>>("fused_reshape_" + input_name);
  auto axis =
      ctx.Attrs().Get<std::vector<int>>("fused_transpose_" + input_name);
  auto dim = ctx.GetInputDim(input_name);

  PADDLE_ENFORCE_GT(dim.size(), 0,
                    platform::errors::InvalidArgument(
                        "The Input(%s) has not been initialized properly. The "
                        "shape of Input(%s) = [%s].",
                        dim));

  // if mkldnn reshape+transpose+matmul fuse activated
  if (!shape.empty() && !axis.empty()) {
    PADDLE_ENFORCE_GE(
        shape.size(), 2,
        platform::errors::InvalidArgument(
            "shape_%s attribute of MatMulOp was implemented for 2, 3 "
            "or 4 dimensions.",
            input_name));
    PADDLE_ENFORCE_LE(
        shape.size(), 4,
        platform::errors::InvalidArgument(
            "shape_%s attribute of MatMulOp was implemented for 2, 3 "
            "or 4 dimensions.",
            input_name));
    PADDLE_ENFORCE_EQ(
        shape.size(), axis.size(),
        platform::errors::InvalidArgument(
            "Ranks of shape_%s and axis_%s attributes of MatMulOp "
            "must be equal.",
            input_name, input_name));

    int num_negative = std::count(shape.begin(), shape.end(), -1);
    PADDLE_ENFORCE_LE(num_negative, 1,
                      platform::errors::InvalidArgument(
                          "The max number of -1 in fused_reshape_%s is 1 "
                          "but received %d.",
                          input_name, num_negative));

    auto it_zero = std::find(shape.begin(), shape.end(), 0);
    if (it_zero != shape.end()) {
      for (uint64_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0) {
          PADDLE_ENFORCE_LT(i, dim.size(),
                            platform::errors::InvalidArgument(
                                "The index of 0 in fused_reshape_%s ",
                                "should be less than output dim size, ",
                                "but the index is %d and output dim size is %d",
                                input_name, i, dim.size()));
          shape[i] = dim.at(i);
        }
      }
    }

    // if "-1" is present then one of reshape dims must be infered
    auto it_negative = std::find(shape.begin(), shape.end(), -1);
    if (it_negative != shape.end()) {
      int64_t dim_product = 1;
      for (int i = 0; i < dim.size(); i++) {
        dim_product *= dim.at(i);
      }

      int64_t shape_product = std::accumulate(shape.begin(), shape.end(), -1,
                                              std::multiplies<int>());
      int index = std::distance(shape.begin(), it_negative);
      shape[index] = dim_product / shape_product;
    }

    dim = dim.reshape(shape).transpose(axis);
  }
  return dim;
}

class MatMulV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "matmul_v2");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "matmul_v2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "matmul_v2");
    bool trans_x = ctx->Attrs().Get<bool>("trans_x");
    bool trans_y = ctx->Attrs().Get<bool>("trans_y");

    std::vector<int64_t> dims_x = phi::vectorize(GetDimForInput(*ctx, "X"));
    std::vector<int64_t> dims_y = phi::vectorize(GetDimForInput(*ctx, "Y"));
    auto ndims_x = dims_x.size();
    auto ndims_y = dims_y.size();
    PADDLE_ENFORCE_GT(ndims_x, 0,
                      platform::errors::InvalidArgument(
                          "The Input(X) dims size must be greater than 0,"
                          " but reviced dims size is 0. "));
    PADDLE_ENFORCE_GT(ndims_y, 0,
                      platform::errors::InvalidArgument(
                          "The Input(Y) dims size must be greater than 0,"
                          " but reviced dims size is 0. "));

    bool x_broadcasted = false, y_broadcasted = false;
    if (ndims_x == 1) {
      dims_x.insert(dims_x.begin(), 1);
      ndims_x = 2;
      x_broadcasted = true;
    }

    if (ndims_y == 1) {
      dims_y.push_back(1);
      ndims_y = 2;
      y_broadcasted = true;
    }

    size_t M, N;
    if (trans_x) {
      M = dims_x[ndims_x - 1];
    } else {
      M = dims_x[ndims_x - 2];
    }
    if (trans_y) {
      N = dims_y[ndims_y - 2];
    } else {
      N = dims_y[ndims_y - 1];
    }

    std::vector<int64_t> new_dims;
    if (ndims_x > ndims_y) {
      new_dims.assign(dims_x.begin(), dims_x.end() - 2);
    } else if (ndims_x < ndims_y) {
      new_dims.assign(dims_y.begin(), dims_y.end() - 2);
    } else {
      new_dims.reserve(ndims_x);
      for (size_t i = 0; i < ndims_x - 2; ++i) {
        new_dims.push_back(std::max(dims_x[i], dims_y[i]));
      }
    }
    if (!x_broadcasted) {
      new_dims.push_back(M);
    }
    if (!y_broadcasted) {
      new_dims.push_back(N);
    }
    if (x_broadcasted && y_broadcasted) {
      new_dims.push_back(1);
    }

    auto ddim_out = phi::make_ddim(new_dims);

#ifdef PADDLE_WITH_MKLDNN
    //  if mkldnn matmul_v2+transpose+reshape fuse activated
    auto reshape_out = ctx->Attrs().Get<std::vector<int>>("fused_reshape_Out");
    auto transpose_out =
        ctx->Attrs().Get<std::vector<int>>("fused_transpose_Out");

    if (!reshape_out.empty() && !transpose_out.empty()) {
      auto reshape_out_size = reshape_out.size();
      auto transpose_out_size = transpose_out.size();
      PADDLE_ENFORCE_EQ(transpose_out_size, 4,
                        platform::errors::InvalidArgument(
                            "transpose_out supported rank is 4, "
                            "received %d",
                            transpose_out_size));
      const std::vector<int> supported_axis{0, 2, 1, 3};
      const bool supported_transpose_axis = std::equal(
          transpose_out.begin(), transpose_out.end(), supported_axis.begin());
      PADDLE_ENFORCE_EQ(
          supported_transpose_axis, true,
          platform::errors::InvalidArgument(
              "supported transpose axis for the fuse are {0, 2, 1, 3}"));
      PADDLE_ENFORCE_EQ(
          reshape_out_size, 3,
          platform::errors::InvalidArgument("reshape_out supported rank is 3, "
                                            "received %d",
                                            reshape_out_size));

      // int num_negative = std::count(reshape_out.begin(), reshape_out.end(),
      // -1);
      // PADDLE_ENFORCE_LE(num_negative, 1,
      //                   platform::errors::InvalidArgument(
      //                       "The max number of -1 in fused_reshape_Out is 1 "
      //                       "but received %d.",
      //                       num_negative));

      // auto it_zero = std::find(reshape_out.begin(), reshape_out.end(), 0);
      // if (it_zero != reshape_out.end()) {
      //   for (uint64_t i = 0; i < reshape_out.size(); i++) {
      //     if (reshape_out[i] == 0) {
      //       PADDLE_ENFORCE_LT(
      //           i, ddim_out.size(),
      //           platform::errors::InvalidArgument(
      //               "The index of 0 in fused_reshape_Out ",
      //               "should be less than output dim size, ",
      //               "but the index is %d and output dim size is %d", i,
      //               ddim_out.size()));
      //       reshape_out[i] = ddim_out.at(i);
      //     }
      //   }
      // }

      // if "-1" is present then one of reshape dims must be infered
      auto it = std::find(reshape_out.begin(), reshape_out.end(), -1);
      if (it != reshape_out.end()) {
        int index = std::distance(reshape_out.begin(), it);

        auto ddim_out_vec = phi::vectorize(ddim_out);

        int ddim_out_product =
            std::accumulate(ddim_out_vec.begin(), ddim_out_vec.end(), 1,
                            std::multiplies<int>());
        int reshape_out_product = std::accumulate(
            reshape_out.begin(), reshape_out.end(), -1, std::multiplies<int>());

        reshape_out[index] = ddim_out_product / reshape_out_product;
      }

      framework::DDim shape_out =
          ddim_out.transpose(transpose_out).reshape(reshape_out);
      ctx->SetOutputDim("Out", shape_out);
    } else {
      ctx->SetOutputDim("Out", ddim_out);
    }
#else
    ctx->SetOutputDim("Out", ddim_out);
#endif

    ctx->ShareLoD("X", /* --> */ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(
          framework::TransToProtoVarType(tensor.dtype()), tensor.place(),
          tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class MatMulV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "tensor of shape (d0, d1 ... M, K)");
    AddInput("Y", "tensor of shape (d0, d1 ... K, N)");
    AddOutput("Out", "tensor of shape (d0, d1 ... M, N)");
    AddAttr<bool>("trans_x",
                  "Set true to transpose the last two dimensions of X before "
                  "doing multiplication")
        .SetDefault(false);
    AddAttr<bool>("trans_y",
                  "Set true to transpose the last two dimensions of Y before "
                  "doing multiplication")
        .SetDefault(false);
    AddAttr<std::vector<int>>(
        "fused_reshape_Out",
        R"DOC(When MKLDNN matmul_v2_transpose_reshape fuse activated, "
              "it's a shape atribute of fused reshape for `Out` output.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>(
        "fused_transpose_Out",
        R"DOC(When MKLDNN matmul_v2_transpose_reshape fuse activated, "
              "it's a axis atribute of fused transpose for `Out` output.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16"})
        .AsExtra();
    AddAttr<std::vector<int>>("fused_reshape_X",
                              R"DOC(Shape of fused reshape of `X` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>("fused_reshape_Y",
                              R"DOC(Shape of fused reshape of `Y` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>("fused_transpose_X",
                              R"DOC(Axis of fused transpose of `X` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>("fused_transpose_Y",
                              R"DOC(Axis of fused transpose of `Y` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddComment(
        R"DOC(Matrix multiplication Out = X * Y. A has shape (d0, d1 ... M, K), 
        B has shape (d0, d1 ... K, N), Out has shape ((d0, d1 ... M, N)). 
        In addition, it also follows the broadcast rule which is similar as
        numpy.matmul.
)DOC");
  }
};

class MatMulV2OpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(
          framework::TransToProtoVarType(tensor.dtype()), tensor.place(),
          tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

template <typename T>
class MatMulV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("matmul_v2_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

class MatMulV2OpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput("DOut"), "Input", "DOut", "matmul");

    if (context->HasOutput("DX") && context->HasInput("DDY")) {
      context->ShareDim("X", "DX");
    }

    if (context->HasOutput("DY") && context->HasInput("DDX")) {
      context->ShareDim("Y", "DY");
    }

    if (context->HasOutput("DDOut") &&
        (context->HasInput("DDY") || context->HasInput("DDX"))) {
      context->ShareDim("DOut", "DDOut");
    }
  }
};

template <typename T>
class MatMulV2OpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("matmul_v2_grad_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    auto ddx = this->OutputGrad(framework::GradVarName("X"));
    auto ddy = this->OutputGrad(framework::GradVarName("Y"));

    if (!ddx.empty() || !ddy.empty()) {
      op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    }
    op->SetOutput("DX",
                  ddy.empty() ? this->EmptyInputGrad() : this->InputGrad("X"));
    op->SetOutput("DY",
                  ddx.empty() ? this->EmptyInputGrad() : this->InputGrad("Y"));

    op->SetAttrMap(this->Attrs());
  }
};
class MatMulV2OpTripleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X",
                   "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y",
                   "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("DOut"), "Input", "DOut",
                   "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("DDX"), "Input", "DDX",
                   "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("DDY"), "Input", "DDY",
                   "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("D_DX"), "Input", "D_DX",
                   "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("D_DY"), "Input", "D_DY",
                   "matmul_v2_triple_grad");
    OP_INOUT_CHECK(context->HasInput("D_DDOut"), "Input", "D_DDOut",
                   "matmul_v2_triple_grad");

    if (context->HasOutput("D_X_out")) {
      context->ShareDim("X", "D_X_out");
    }
    if (context->HasOutput("D_Y_out")) {
      context->ShareDim("Y", "D_Y_out");
    }
    if (context->HasOutput("D_DOut_out")) {
      context->ShareDim("DOut", "D_DOut_out");
    }
    if (context->HasOutput("D_DDX_out")) {
      context->ShareDim("X", "D_DDX_out");
    }
    if (context->HasOutput("D_DDY_out")) {
      context->ShareDim("Y", "D_DDY_out");
    }
  }
};

template <typename T>
class MatMulV2OpTripleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("matmul_v2_triple_grad");

    // get input from double grad
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input("DOut"));
    op->SetInput("DDX", this->Input("DDX"));
    op->SetInput("DDY", this->Input("DDY"));
    op->SetInput("D_DX", this->OutputGrad("DX"));
    op->SetInput("D_DY", this->OutputGrad("DY"));
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));

    // set outputs
    op->SetOutput("D_X_out", this->InputGrad("X"));
    op->SetOutput("D_Y_out", this->InputGrad("Y"));
    op->SetOutput("D_DOut_out", this->InputGrad("DOut"));
    op->SetOutput("D_DDX_out", this->InputGrad("DDX"));
    op->SetOutput("D_DDY_out", this->InputGrad("DDY"));

    op->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(matmul_v2, ops::MatMulV2Op, ops::MatMulV2OpMaker,
                  ops::MatMulV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::MatMulV2GradOpMaker<paddle::imperative::OpBase>);

DECLARE_INFER_SHAPE_FUNCTOR(matmul_v2_grad, MatMulV2GradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralBinaryGradInferMeta));
REGISTER_OPERATOR(matmul_v2_grad, ops::MatMulV2OpGrad,
                  ops::MatMulV2OpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulV2OpDoubleGradMaker<paddle::imperative::OpBase>,
                  MatMulV2GradInferShapeFunctor);

REGISTER_OPERATOR(matmul_v2_grad_grad, ops::MatMulV2OpDoubleGrad,
                  ops::MatMulV2OpTripleGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulV2OpTripleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(matmul_v2_triple_grad, ops::MatMulV2OpTripleGrad);

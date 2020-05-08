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

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op_function.h"

namespace paddle {
namespace operators {

#define HANDLE_DIM(NDIM, RDIM)                                            \
  if (ndim == NDIM && rdim == RDIM) {                                     \
    ReduceFunctor<DeviceContext, OutT, NDIM, RDIM, Functor>(              \
        context.template device_context<DeviceContext>(), *input, output, \
        dims, keep_dim);                                                  \
  }

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T, typename Functor>
struct ReduceKernelFunctor {
  const Tensor* input;
  Tensor* output;
  std::vector<int> dims;
  bool keep_dim;
  bool reduce_all;
  const framework::ExecutionContext& context;
  ReduceKernelFunctor(const Tensor* input, Tensor* output,
                      const std::vector<int>& dims, bool keep_dim,
                      bool reduce_all,
                      const framework::ExecutionContext& context)
      : input(input),
        output(output),
        dims(dims),
        keep_dim(keep_dim),
        reduce_all(reduce_all),
        context(context) {}

  template <typename OutT>
  void apply() const {
    output->mutable_data<OutT>(context.GetPlace());
    if (reduce_all) {
      // Flatten and reduce 1-D tensor
      auto x = EigenVector<OutT>::Flatten(*input);
      auto out = EigenScalar<OutT>::From(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      Functor functor;
      functor(place, &x, &out, reduce_dim);
    } else {
      int ndim = input->dims().size();
      int rdim = dims.size();
      HANDLE_DIM(4, 3);
      HANDLE_DIM(4, 2);
      HANDLE_DIM(4, 1);
      HANDLE_DIM(3, 2);
      HANDLE_DIM(3, 1);
      HANDLE_DIM(2, 1);
      HANDLE_DIM(1, 1);
    }
  }
};
template <typename DeviceContext, typename T, typename Functor>
class ReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* output = context.Output<Tensor>("Out");
    auto dims = context.Attr<std::vector<int>>("dim");
    bool keep_dim = context.Attr<bool>("keep_dim");
    int out_dtype = context.Attr<int>("out_dtype");
    framework::proto::VarType::Type cast_out_dtype;

    if (out_dtype < 0) {
      auto* cast_input = context.Input<Tensor>("X");
      cast_out_dtype =
          static_cast<framework::proto::VarType::Type>(cast_input->type());
      framework::VisitDataType(
          cast_out_dtype,
          ReduceKernelFunctor<DeviceContext, T, Functor>(
              cast_input, output, dims, keep_dim, reduce_all, context));
    } else {
      Tensor tmp_tensor;
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(out_dtype);
      auto* input = context.Input<Tensor>("X");

      tmp_tensor.Resize(input->dims());
      framework::VisitDataType(
          cast_out_dtype,
          CastOpFunctor<DeviceContext, T>(
              input, &tmp_tensor,
              context.template device_context<DeviceContext>()));
      framework::VisitDataType(
          cast_out_dtype,
          ReduceKernelFunctor<DeviceContext, T, Functor>(
              &tmp_tensor, output, dims, keep_dim, reduce_all, context));
    }
  }
};

template <typename DeviceContext, typename OutT, typename Functor>
class BoolReduceKernel : public framework::OpKernel<OutT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<OutT>(context.GetPlace());

    auto dims = context.Attr<std::vector<int>>("dim");
    bool keep_dim = context.Attr<bool>("keep_dim");

    if (reduce_all) {
      // Flatten and reduce 1-D tensor
      auto x = EigenVector<OutT>::Flatten(*input);
      auto out = EigenScalar<OutT>::From(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      Functor functor;
      functor(place, &x, &out, reduce_dim);
    } else {
      int ndim = input->dims().size();
      int rdim = dims.size();
      // comments for accelerating compiling temporarily.
      //      HANDLE_DIM(6, 5);
      //      HANDLE_DIM(6, 4);
      //      HANDLE_DIM(6, 3);
      //      HANDLE_DIM(6, 2);
      //      HANDLE_DIM(6, 1);
      //      HANDLE_DIM(5, 4);
      //      HANDLE_DIM(5, 3);
      //      HANDLE_DIM(5, 2);
      //      HANDLE_DIM(5, 1);
      HANDLE_DIM(4, 3);
      HANDLE_DIM(4, 2);
      HANDLE_DIM(4, 1);
      HANDLE_DIM(3, 2);
      HANDLE_DIM(3, 1);
      HANDLE_DIM(2, 1);
      HANDLE_DIM(1, 1);
    }
  }
};
template <typename DeviceContext, typename T, typename Functor,
          bool kNoNeedBufferX = false, bool kNoNeedBufferY = false>
class ReduceGradKernel : public framework::OpKernel<T> {
 public:
  void ComputeFromInput(const Tensor* input2,
                        const framework::ExecutionContext& context) const {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto dims = context.Attr<std::vector<int>>("dim");
    auto* input0 = context.Input<Tensor>("X");
    auto* input1 = context.Input<Tensor>("Out");

    auto* output = context.Output<Tensor>(framework::GradVarName("X"));
    output->mutable_data<T>(context.GetPlace());

    // NOTE: EigenTensor::From() uses tensor->data()
    // if op has NoNeedBufferVarsInferer, the corresponding kNoNeedBufferX or
    // kNoNeedBufferY should set true
    // and use fake var that has same dims.
    if (kNoNeedBufferX) {
      input0 = output;
    }
    if (kNoNeedBufferY) {
      input1 = input2;
    }

    // NOTE(dengkaipeng): Out is unnecessary in some reduce kernel and
    // not be set as Input in grad Maker, use Out_grad to replace here
    if (!input1) input1 = input2;

    if (reduce_all) {
      auto x = EigenVector<T>::Flatten(*input0);
      auto x_reduce = EigenVector<T>::From(*input1);
      auto x_reduce_grad = EigenVector<T>::From(*input2);
      auto x_grad = EigenVector<T>::Flatten(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto broadcast_dim =
          Eigen::array<int, 1>({{static_cast<int>(input0->numel())}});
      Functor functor;
      functor(place, &x, &x_reduce, &x_grad, &x_reduce_grad, broadcast_dim,
              broadcast_dim[0]);
    } else {
      int rank = input0->dims().size();
      switch (rank) {
        case 1:
          ReduceGradFunctor<DeviceContext, T, 1, Functor>(
              context.template device_context<DeviceContext>(), *input0,
              *input1, *input2, output, dims);
          break;
        case 2:
          ReduceGradFunctor<DeviceContext, T, 2, Functor>(
              context.template device_context<DeviceContext>(), *input0,
              *input1, *input2, output, dims);
          break;
        case 3:
          ReduceGradFunctor<DeviceContext, T, 3, Functor>(
              context.template device_context<DeviceContext>(), *input0,
              *input1, *input2, output, dims);
          break;
        case 4:
          ReduceGradFunctor<DeviceContext, T, 4, Functor>(
              context.template device_context<DeviceContext>(), *input0,
              *input1, *input2, output, dims);
          break;
        case 5:
          ReduceGradFunctor<DeviceContext, T, 5, Functor>(
              context.template device_context<DeviceContext>(), *input0,
              *input1, *input2, output, dims);
          break;
        case 6:
          ReduceGradFunctor<DeviceContext, T, 6, Functor>(
              context.template device_context<DeviceContext>(), *input0,
              *input1, *input2, output, dims);
          break;
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    int in_dtype = context.Attr<int>("in_dtype");
    if (in_dtype >= 0) {
      Tensor tmp_tensor;
      auto* pre_input = context.Input<Tensor>(framework::GradVarName("Out"));
      auto in_kernel_type =
          framework::OpKernelType(pre_input->type(), context.GetPlace());
      auto out_kernel_type = framework::OpKernelType(
          static_cast<framework::proto::VarType::Type>(in_dtype),
          context.GetPlace());
      framework::TransDataType(in_kernel_type, out_kernel_type, *pre_input,
                               &tmp_tensor);
      ComputeFromInput(&tmp_tensor, context);

    } else {
      auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
      ComputeFromInput(input2, context);
    }
  }
};

class ReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ReduceOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ReduceOp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimensions of ReduceOp "
                          "should be less equal than 6. But received X's "
                          "dimensions = %d, X's shape = [%s].",
                          x_rank, x_dims));
    auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
    PADDLE_ENFORCE_GT(dims.size(), 0,
                      platform::errors::InvalidArgument(
                          "The input dim dimensions of ReduceOp "
                          "should be greater than 0. But received the dim "
                          "dimesions of Reduce = %d.",
                          dims.size()));

    for (size_t i = 0; i < dims.size(); ++i) {
      PADDLE_ENFORCE_LT(dims[i], x_rank,
                        platform::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [-dimension(X), dimension(X)] "
                            "which dimesion = %d. But received dim index = %d.",
                            i, x_rank, dims[i]));
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
    }
    sort(dims.begin(), dims.end());
    bool reduce_all = ctx->Attrs().Get<bool>("reduce_all");
    bool keep_dim = ctx->Attrs().Get<bool>("keep_dim");
    if (reduce_all) {
      if (keep_dim)
        ctx->SetOutputDim(
            "Out", framework::make_ddim(std::vector<int64_t>(x_rank, 1)));
      else
        ctx->SetOutputDim("Out", {1});
    } else {
      auto dims_vector = vectorize(x_dims);
      if (keep_dim) {
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = 1;
        }
      } else {
        const int kDelFlag = -2;
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = kDelFlag;
        }
        dims_vector.erase(
            remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
            dims_vector.end());
      }
      if (!keep_dim && dims_vector.size() == 0) {
        dims_vector.push_back(1);
      }
      auto out_dims = framework::make_ddim(dims_vector);
      ctx->SetOutputDim("Out", out_dims);
      if (dims.size() > 0 && dims[0] != 0) {
        // Only pass LoD when not reducing on the first dim.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }
};

class ReduceOpUseInputPlace : public ReduceOp {
 public:
  using ReduceOp::ReduceOp;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

class ReduceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ReduceOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "ReduceOp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6,
                      platform::errors::InvalidArgument(
                          "Tensors with rank at most 6 are supported by "
                          "ReduceOp. Received tensor with rank %d.",
                          x_rank));
    auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
    for (size_t i = 0; i < dims.size(); ++i) {
      PADDLE_ENFORCE_LT(dims[i], x_rank,
                        platform::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [-dimension(X), dimension(X)], "
                            "which dimesion = %d. But received dim index = %d.",
                            i, x_rank, dims[i]));
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
    }
    sort(dims.begin(), dims.end());
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    int in_dtype = ctx.Attr<int>("in_dtype");
    if (in_dtype >= 0) {
      return framework::OpKernelType(
          static_cast<framework::proto::VarType::Type>(in_dtype),
          ctx.GetPlace());
    }
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

class ReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput("X",
             "(Tensor) The input tensor. Tensors with rank at most 6 are "
             "supported.");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddAttr<std::vector<int>>(
        "dim",
        "(list<int>, default {0}) The dimensions to reduce. "
        "Must be in the range [-rank(input), rank(input)). "
        "If `dim[i] < 0`, the dims[i] to reduce is `rank + dims[i]`. "
        "Note that reducing on the first dim will make the LoD info lost.")
        .SetDefault({0});
    AddAttr<bool>("keep_dim",
                  "(bool, default false) "
                  "If true, retain the reduced dimension with length 1.")
        .SetDefault(false);
    AddAttr<bool>("reduce_all",
                  "(bool, default false) "
                  "If true, output a scalar reduced along all dimensions.")
        .SetDefault(false);
    AddAttr<int>("in_dtype",
                 "(int, default -1)"
                 "The dtype of input, default value is -1, the user could not "
                 "set this value.")
        .SetDefault(-1);
    AddAttr<int>(
        "out_dtype",
        "(int, default -1)"
        "The dtype of output, default value is -1, the dtype is same as intput")
        .SetDefault(-1);
    AddComment(string::Sprintf(R"DOC(
%s Operator.

This operator computes the %s of input tensor along the given dimension.
The result tensor has 1 fewer dimension than the input unless keep_dim is true.
If reduce_all is true, just reduce along all dimensions and output a scalar.

)DOC",
                               GetOpType(), GetName()));
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual std::string GetOpType() const = 0;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_REDUCE_OP(op_name)                                           \
  class __##op_name##Maker__ : public ops::ReduceOpMaker {                    \
   protected:                                                                 \
    virtual std::string GetName() const { return #op_name; }                  \
    virtual std::string GetOpType() const { return "Reduce " #op_name; }      \
  };                                                                          \
  REGISTER_OPERATOR(                                                          \
      op_name, ops::ReduceOp, __##op_name##Maker__,                           \
      paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>, \
      paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase,       \
                                            true>);                           \
  REGISTER_OPERATOR(op_name##_grad, ops::ReduceGradOp)

#define REGISTER_REDUCE_OP_WITHOUT_GRAD(op_name, ...)                    \
  class __##op_name##Maker__ : public ops::ReduceOpMaker {               \
   protected:                                                            \
    virtual std::string GetName() const { return #op_name; }             \
    virtual std::string GetOpType() const { return "Reduce " #op_name; } \
  };                                                                     \
  REGISTER_OPERATOR(                                                     \
      op_name, ops::ReduceOp##__VA_ARGS__, __##op_name##Maker__,         \
      paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,    \
      paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

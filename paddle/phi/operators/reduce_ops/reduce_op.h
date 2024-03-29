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
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/operators/reduce_ops/reduce_op_function.h"
// only can include the headers in paddle/phi/api dirs
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/phi/kernels/cpu/reduce.h"

#if defined(__HIPCC__) || defined(__NVCC__) || defined(__xpu__)
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/gpu/reduce_grad.h"
#endif

namespace paddle {
namespace operators {

#define HANDLE_DIM(NDIM, RDIM)                                   \
  if (ndim == NDIM && rdim == RDIM) {                            \
    paddle::operators::                                          \
        ReduceFunctor<DeviceContext, OutT, NDIM, RDIM, Functor>( \
            context.template device_context<DeviceContext>(),    \
            *input,                                              \
            output,                                              \
            dims,                                                \
            keep_dim);                                           \
  }

using DDim = framework::DDim;

inline void GetShuffledDim(const DDim& src_dims,
                           DDim* dst_dims,
                           const std::vector<int>& reduced_dims,
                           std::vector<int>* perm_axis) {
  // check if it's a reduced dim
  std::vector<bool> src_dims_check(src_dims.size(), false);
  size_t src_size = src_dims.size();
  size_t reduce_size = reduced_dims.size();
  for (size_t i = 0; i < reduce_size; ++i) {
    dst_dims->at(src_size - reduce_size + i) = src_dims[reduced_dims[i]];
    (*perm_axis)[src_size - reduce_size + i] = reduced_dims[i];
    src_dims_check[reduced_dims[i]] = true;
  }

  size_t offset = 0;
  for (size_t i = 0; i < src_dims_check.size(); ++i) {
    bool is_reduced = src_dims_check[i];
    if (!is_reduced) {
      (*perm_axis)[offset] = i;
      dst_dims->at(offset++) = src_dims[i];
    }
  }
}

static inline std::vector<int> GetReduceDim(const std::vector<int>& dims,
                                            int dim_size,
                                            bool reduce_all) {
  std::vector<int> reduce_dims;
  if (reduce_all) {
    reduce_dims.resize(dim_size);
    int reduce_size = reduce_dims.size();
    for (int i = 0; i < reduce_size; ++i) {
      reduce_dims[i] = i;
    }
  } else {
    for (auto e : dims) {
      PADDLE_ENFORCE_LT(e,
                        dim_size,
                        paddle::platform::errors::InvalidArgument(
                            "ReduceBaseOp: invalid axis, when x_dims is %d, "
                            "axis[i] should less than x_dims, but got %d.",
                            dim_size,
                            e));
      reduce_dims.push_back(e >= 0 ? e : e + dim_size);
    }
  }
  return reduce_dims;
}
template <typename DeviceContext, typename OutT>
void GetShuffledInput(const framework::ExecutionContext& context,
                      const phi::DenseTensor* input,
                      phi::DenseTensor* shuffled_input,
                      const std::vector<int>& dims) {
  DDim shuffled_dims(input->dims());
  std::vector<int> perm_axis(input->dims().size());
  GetShuffledDim(input->dims(), &shuffled_dims, dims, &perm_axis);

  shuffled_input->Resize(shuffled_dims);
  shuffled_input->mutable_data<OutT>(context.GetPlace());

  phi::funcs::TransposeNormal<DeviceContext, OutT> trans;
  trans(context.template device_context<DeviceContext>(),
        *input,
        shuffled_input,
        perm_axis);
}

inline void GetOriginDimFromShuffled(const DDim& src_dim,
                                     const std::vector<int>& dims,
                                     std::vector<int>* origin_dim) {
  DDim shuffled_dims(src_dim);
  size_t n = src_dim.size();
  std::vector<int> perm_axis(n);
  GetShuffledDim(src_dim, &shuffled_dims, dims, &perm_axis);
  for (size_t i = 0; i < n; ++i) {
    (*origin_dim)[perm_axis[i]] = i;
  }
}

template <typename DeviceContext, typename OutT, typename Functor>
void HandleLargeDim(const framework::ExecutionContext& context,
                    const phi::DenseTensor* input,
                    phi::DenseTensor* output,
                    const std::vector<int>& dims,
                    bool keep_dim) {
  //  shuffle the reduced dim to the end
  phi::DenseTensor shuffled_input;
  GetShuffledInput<DeviceContext, OutT>(context, input, &shuffled_input, dims);

  // transpose to 2D tensor whose shape is {unreduced, reduced}.
  const int64_t unreduced = output->numel();
  const int64_t input_numel = shuffled_input.numel();
  // assume: 0 / 0 == 0, which allow process 0 dim tensor
  const int64_t reduced = (unreduced != 0) ? (input_numel / unreduced) : 0;

  PADDLE_ENFORCE_EQ(
      unreduced * reduced,
      input_numel,
      phi::errors::InvalidArgument(
          "Reducing failed in HandleLargeDim, when try to transpose (%d) "
          "operands into 2D tensor with shape (%d, %d).",
          input_numel,
          unreduced,
          reduced));

  shuffled_input.Resize({unreduced, reduced});

  DDim output_dim = output->dims();
  output->Resize({unreduced});
  paddle::operators::ReduceFunctor<DeviceContext, OutT, 2, 1, Functor>(
      context.template device_context<DeviceContext>(),
      shuffled_input,
      output,
      {1},
      keep_dim);
  output->Resize(output_dim);
}

template <typename DeviceContext, typename T, typename Functor>
void HandleLargeDimGrad(const framework::ExecutionContext& context,
                        const phi::DenseTensor* x,
                        const phi::DenseTensor* out,
                        const phi::DenseTensor* dout,
                        phi::DenseTensor* dx,
                        Functor functor,
                        const std::vector<int>& dims) {
  const int64_t unreduced = out->numel();
  const int64_t x_numel = x->numel();
  // assume: 0 / 0 == 0, which allow process 0 dim tensor
  const int64_t reduced = (unreduced != 0) ? (x_numel / unreduced) : 0;

  PADDLE_ENFORCE_EQ(
      unreduced * reduced,
      x_numel,
      phi::errors::InvalidArgument(
          "Reducing failed in HandleLargeDimGrad, when try to transpose (%d) "
          "operands into 2D tensor with shape (%d, %d).",
          x_numel,
          unreduced,
          reduced));

  DDim out_dim(out->dims());
  DDim x_dim(x->dims());
  // transpose and reshape X
  phi::DenseTensor shuffled_x;
  GetShuffledInput<DeviceContext, T>(context, x, &shuffled_x, dims);
  DDim shuffled_dim = shuffled_x.dims();
  shuffled_x.Resize({unreduced, reduced});
  // reshape dX {unreduced, reduced}
  dx->Resize({unreduced, reduced});
  ReduceGradFunctor<DeviceContext, T, 2, Functor>(
      context.template device_context<DeviceContext>(),
      shuffled_x,
      *out,
      *dout,
      dx,
      functor,
      {1});
  // transpose dX
  std::vector<int> origin_axis(x_dim.size());
  GetOriginDimFromShuffled(x_dim, dims, &origin_axis);
  phi::DenseTensor dx_tmp;
  framework::TensorCopy(*dx, context.GetPlace(), &dx_tmp);
  dx_tmp.Resize(shuffled_dim);
  dx->Resize(x_dim);
  phi::funcs::TransposeNormal<DeviceContext, T> trans;
  trans(context.template device_context<DeviceContext>(),
        dx_tmp,
        dx,
        origin_axis);
}

template <typename DeviceContext, typename T, typename Functor>
struct ReduceKernelFunctor {
  const phi::DenseTensor* input;
  phi::DenseTensor* output;
  std::vector<int> dims;
  bool keep_dim;
  bool reduce_all;
  const framework::ExecutionContext& context;
  ReduceKernelFunctor(const phi::DenseTensor* input,
                      phi::DenseTensor* output,
                      const std::vector<int>& dims,
                      bool keep_dim,
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
      if (ndim > 6) {
        HandleLargeDim<DeviceContext, OutT, Functor>(
            context, input, output, dims, keep_dim);
      } else {
        HANDLE_DIM(6, 5);
        HANDLE_DIM(6, 4);
        HANDLE_DIM(6, 3);
        HANDLE_DIM(6, 2);
        HANDLE_DIM(6, 1);
        HANDLE_DIM(5, 4);
        HANDLE_DIM(5, 3);
        HANDLE_DIM(5, 2);
        HANDLE_DIM(5, 1);
        HANDLE_DIM(4, 3);
        HANDLE_DIM(4, 2);
        HANDLE_DIM(4, 1);
        HANDLE_DIM(3, 2);
        HANDLE_DIM(3, 1);
        HANDLE_DIM(2, 1);
        HANDLE_DIM(1, 1);
      }
    }
  }
};
template <typename DeviceContext, typename T, typename Functor>
class ReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* output = context.Output<phi::DenseTensor>("Out");
    auto dims = context.Attr<std::vector<int>>("dim");
    bool keep_dim = context.Attr<bool>("keep_dim");
    int out_dtype = context.Attr<int>("out_dtype");
    framework::proto::VarType::Type cast_out_dtype;
    auto* input = context.Input<phi::DenseTensor>("X");

    if (out_dtype < 0) {
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(
          framework::TransToProtoVarType(input->dtype()));
    } else {
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(out_dtype);
    }

    auto& dev_ctx = context.device_context<DeviceContext>();
    output->mutable_data(
        dev_ctx.GetPlace(),
        static_cast<framework::proto::VarType::Type>(cast_out_dtype));

    std::vector<int64_t> tmp_dims(dims.begin(), dims.end());

    // call new kernel
    phi::Reduce<typename framework::ConvertToPhiContext<DeviceContext>::TYPE,
                T,
                Functor>(
        static_cast<const typename framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *input,
        reduce_all,
        tmp_dims,
        keep_dim,
        framework::TransToPhiDataType(cast_out_dtype),
        output);
  }
};

template <typename DeviceContext, typename T, typename Functor>
void LaunchReduceGradKernel(const framework::ExecutionContext& context,
                            const phi::DenseTensor* input0,
                            const phi::DenseTensor* input1,
                            const phi::DenseTensor* input2,
                            phi::DenseTensor* output,
                            Functor functor,
                            const std::vector<int>& dims,
                            bool reduce_all = false) {
  if (reduce_all) {
    auto x = EigenVector<T>::Flatten(*input0);
    auto x_reduce = EigenVector<T>::Flatten(*input1);
    auto x_reduce_grad = EigenVector<T>::Flatten(*input2);
    auto x_grad = EigenVector<T>::Flatten(*output);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto broadcast_dim =
        Eigen::array<int, 1>({{static_cast<int>(input0->numel())}});
    functor(place,
            &x,
            &x_reduce,
            &x_grad,
            &x_reduce_grad,
            broadcast_dim,
            broadcast_dim[0]);
  } else {
    int rank = input0->dims().size();
    switch (rank) {
      case 1:
        ReduceGradFunctor<DeviceContext, T, 1, Functor>(
            context.template device_context<DeviceContext>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 2:
        ReduceGradFunctor<DeviceContext, T, 2, Functor>(
            context.template device_context<DeviceContext>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 3:
        ReduceGradFunctor<DeviceContext, T, 3, Functor>(
            context.template device_context<DeviceContext>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 4:
        ReduceGradFunctor<DeviceContext, T, 4, Functor>(
            context.template device_context<DeviceContext>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 5:
        ReduceGradFunctor<DeviceContext, T, 5, Functor>(
            context.template device_context<DeviceContext>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 6:
        ReduceGradFunctor<DeviceContext, T, 6, Functor>(
            context.template device_context<DeviceContext>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      default:
        HandleLargeDimGrad<DeviceContext, T, Functor>(
            context, input0, input1, input2, output, functor, dims);
        break;
    }
  }
}

template <typename DeviceContext,
          typename T,
          typename Functor,
          bool kNoNeedBufferX = false,
          bool kNoNeedBufferY = false>
class ReduceGradKernel : public framework::OpKernel<T> {
 public:
  void ComputeFromInput(const phi::DenseTensor* input2,
                        const framework::ExecutionContext& context) const {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto dims = context.Attr<std::vector<int>>("dim");
    auto* input0 = context.Input<phi::DenseTensor>("X");
    auto* input1 = context.Input<phi::DenseTensor>("Out");

    auto* output =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    output->mutable_data<T>(context.GetPlace());

    // The dims has full dim, set the reduce_all is True
    const auto& input_dim_size =
        context.Input<phi::DenseTensor>("X")->dims().size();
    std::set<int> dims_set(dims.begin(), dims.end());
    bool full_dim = true;
    for (auto i = 0; i < input_dim_size; i++) {
      if (dims_set.find(i) == dims_set.end()) {
        full_dim = false;
        break;
      }
    }
    reduce_all = (reduce_all || full_dim);
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

    const std::vector<int> const_dims = dims;

    // NOTE(dengkaipeng): Out is unnecessary in some reduce kernel and
    // not be set as Input in grad Maker, use Out_grad to replace here
    if (!input1) input1 = input2;
    Functor functor;
    LaunchReduceGradKernel<DeviceContext, T, Functor>(context,
                                                      input0,
                                                      input1,
                                                      input2,
                                                      output,
                                                      functor,
                                                      const_dims,
                                                      reduce_all);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    int in_dtype = context.Attr<int>("in_dtype");
    if (in_dtype >= 0) {
      phi::DenseTensor tmp_tensor;
      auto* pre_input =
          context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
      auto in_kernel_type =
          phi::KernelKey(framework::TransToProtoVarType(pre_input->dtype()),
                         context.GetPlace());
      auto out_kernel_type =
          phi::KernelKey(static_cast<framework::proto::VarType::Type>(in_dtype),
                         context.GetPlace());
      framework::TransDataType(
          in_kernel_type, out_kernel_type, *pre_input, &tmp_tensor);
      ComputeFromInput(&tmp_tensor, context);

    } else {
      auto* input2 =
          context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
      ComputeFromInput(input2, context);
    }
  }
};

class ReduceBaseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ReduceBaseOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ReduceBaseOp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
    PADDLE_ENFORCE_GT(dims.size(),
                      0,
                      platform::errors::InvalidArgument(
                          "The input dim dimensions of ReduceBaseOp "
                          "should be greater than 0. But received the dim "
                          "dimensions of Reduce = %d.",
                          dims.size()));

    for (size_t i = 0; i < dims.size(); ++i) {
      PADDLE_ENFORCE_LT(
          dims[i],
          x_rank,
          platform::errors::InvalidArgument(
              "The reduce dim index %d should be in the "
              "range [-dimension(X), dimension(X)] "
              "which dimension = %d. But received dim index = %d.",
              i,
              x_rank,
              dims[i]));
      PADDLE_ENFORCE_GE(
          dims[i],
          -x_rank,
          platform::errors::InvalidArgument(
              "The reduce dim index %d should be in the "
              "range [-dimension(X), dimension(X)] "
              "which dimension = %d. But received dim index = %d.",
              i,
              x_rank,
              dims[i]));
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
    }
    sort(dims.begin(), dims.end());
    bool reduce_all = ctx->Attrs().Get<bool>("reduce_all");
    bool keep_dim = ctx->Attrs().Get<bool>("keep_dim");
    if (reduce_all) {
      if (keep_dim)
        ctx->SetOutputDim("Out",
                          common::make_ddim(std::vector<int64_t>(x_rank, 1)));
      else
        ctx->SetOutputDim("Out", {1});
    } else {
      auto dims_vector = common::vectorize(x_dims);
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
      auto out_dims = common::make_ddim(dims_vector);
      ctx->SetOutputDim("Out", out_dims);
      if (dims.size() > 0 && dims[0] != 0) {
        // Only pass LoD when not reducing on the first dim.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }

  // oneDNN's reduction kernel is optimized only for reducing throughout the
  // most outer dims, so in case of another type of reduction, it would be
  // better to fallback to native implementation
  static bool HasOptimizedOneDNNKernel(const framework::ExecutionContext& ctx) {
    // native reduce kernels don't support bf16
    // so oneDNN kernel is enforced in that case
    if (ctx.Input<phi::DenseTensor>("X")->dtype() == phi::DataType::BFLOAT16)
      return true;

    if (!ctx.HasAttr("dim") || !ctx.HasAttr("reduce_all")) {
      return false;
    }

    auto reduce_dims = ctx.Attr<std::vector<int>>("dim");
    const bool reduce_all = ctx.Attr<bool>("reduce_all");
    int ndims = ctx.Input<phi::DenseTensor>("X")->dims().size();

    if (reduce_all) {
      return true;
    }

    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      if (reduce_dims[i] < 0) reduce_dims[i] = ndims + reduce_dims[i];
    }
    sort(reduce_dims.begin(), reduce_dims.end());
    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      if (reduce_dims[reduce_dims.size() - i - 1] !=
          static_cast<int>(ndims - i - 1)) {
        return false;
      }
    }

    return true;
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // choose cudnn kernel if the runtime supported.
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

    // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
    if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5 ||
        !HasOptimizedOneDNNKernel(ctx)) {
      this->SetDnnFallback(true);
    }
    // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_DNNL

    if (input_data_type == framework::proto::VarType::FP16) {
      PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()) ||
                            platform::is_xpu_place(ctx.GetPlace()) ||
                            platform::is_custom_place(ctx.GetPlace()),
                        true,
                        platform::errors::InvalidArgument(
                            "float16 can only be used on GPU or XPU place"));
    }
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class ReduceOpUseInputPlace : public ReduceBaseOp {
 public:
  using ReduceBaseOp::ReduceBaseOp;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    phi::KernelKey kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    kt.set_backend(
        phi::TransToPhiBackend(ctx.Input<phi::DenseTensor>("X")->place()));
    return kt;
  }
};

class ReduceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ReduceBaseOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "ReduceBaseOp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    // TODO(dev): We should delete Infershape and migrate it into
    // UnchangeInferMeta.In case of 'dim' is Variable, it will
    // not exist in Attrs but in Inputs.
    if (ctx->HasAttr("dim")) {
      auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
      for (size_t i = 0; i < dims.size(); ++i) {
        PADDLE_ENFORCE_LT(
            dims[i],
            x_rank,
            platform::errors::InvalidArgument(
                "The reduce dim index %d should be in the "
                "range [-dimension(X), dimension(X)], "
                "which dimension = %d. But received dim index = %d.",
                i,
                x_rank,
                dims[i]));
        if (dims[i] < 0) dims[i] = x_rank + dims[i];
      }
    }

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    int out_dtype = ctx.Attr<int>("out_dtype");
    auto input_data_type =
        (out_dtype >= 0)
            ? static_cast<framework::proto::VarType::Type>(out_dtype)
            : OperatorWithKernel::IndicateVarDataType(
                  ctx, framework::GradVarName("Out"));

    // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
    // max 5D tensor is supported
    if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5) {
      dnn_fallback_ = true;
    }
    // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_DNNL

    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class ReduceBaseOpMaker : public framework::OpProtoAndCheckerMaker {
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
        .SetDefault({0})
        .SupportTensor();
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
                               GetOpType(),
                               GetName()));
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual std::string GetOpType() const = 0;
};

#if defined(__HIPCC__) || defined(__NVCC__) || defined(__xpu__)
template <typename T,
          template <typename>
          class ReduceBaseOp,
          template <typename, typename>
          class TransformOp>
class ReduceCudaKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    const phi::DenseTensor* input = context.Input<phi::DenseTensor>("X");
    phi::DenseTensor* output = context.Output<phi::DenseTensor>("Out");
    auto out_dtype = context.Attr<int>("out_dtype");
    auto pt_out_dtype = paddle::framework::TransToPhiDataType(
        static_cast<framework::proto::VarType::Type>(out_dtype));
    std::vector<int> dims = context.Attr<std::vector<int>>("dim");
#ifdef PADDLE_WITH_XPU_KP
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
#else
    auto& dev_ctx = context.cuda_device_context();
#endif
    if (out_dtype >= 0) {
      output->mutable_data(dev_ctx.GetPlace(), pt_out_dtype);
    } else {
      output->mutable_data(dev_ctx.GetPlace(), input->dtype());
    }

    std::vector<int64_t> dims_int64{dims.begin(), dims.end()};

    phi::Reduce<T, ReduceBaseOp, TransformOp>(
        dev_ctx, *input, reduce_all, dims_int64, false, pt_out_dtype, output);
  }
};

#ifndef PADDLE_WITH_XPU_KP
template <typename T, template <typename, typename> class TransformOp>
class ReduceCudaGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    std::vector<int> dims = context.Attr<std::vector<int>>("dim");
    auto* in_x = context.Input<phi::DenseTensor>("X");

    auto* d_out =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* d_x = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto out_dtype = context.Attr<int>("in_dtype");
    auto pt_out_dtype = framework::TransToPhiDataType(
        static_cast<framework::proto::VarType::Type>(out_dtype));
    // get reduce_dim and reduce_num for reduce_mean_grad
    int dim_size = in_x->dims().size();
    std::vector<int> reduce_dims = GetReduceDim(dims, dim_size, reduce_all);
    auto update_dims = common::vectorize(d_x->dims());
    int reduce_num = 1;
    for (auto i : reduce_dims) {
      reduce_num *= (in_x->dims())[i];
      update_dims[i] = 1;
    }
    // make new tensor
    phi::DenseTensor new_d_out(d_out->type());
    new_d_out.ShareDataWith(*d_out);
    new_d_out.Resize(common::make_ddim(update_dims));
    auto& dev_ctx = context.cuda_device_context();
    if (out_dtype > 0) {
      d_x->mutable_data(dev_ctx.GetPlace(), pt_out_dtype);
    } else {
      d_x->mutable_data(dev_ctx.GetPlace(), d_out->dtype());
    }
    auto pt_d_out = std::make_unique<phi::DenseTensor>(new_d_out);
    auto pt_d_x = std::make_unique<phi::DenseTensor>(*d_x);
    if (out_dtype <= 0) {
      pt_out_dtype = d_out->dtype();
    }

    using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
    phi::ReduceGrad<TransformOp<T, MPType>>(dev_ctx,
                                            pt_d_out.get(),
                                            pt_d_x.get(),
                                            pt_out_dtype,
                                            TransformOp<T, MPType>(reduce_num));
  }
};

template <typename T>
struct EqualFunctor {
  inline T initial() { return static_cast<T>(0.0f); }

  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return static_cast<T>(a == b);
  }
};

template <typename T, typename Enable = void>
struct DivideFunctor {
  inline T initial() { return static_cast<T>(1.0f); }

  inline HOSTDEVICE T operator()(const T a, const T b) const { return a / b; }
};
#endif
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_REDUCE_OP(op_name)                                           \
  class __##op_name##Maker__ : public ops::ReduceBaseOpMaker {                \
   protected:                                                                 \
    virtual std::string GetName() const { return #op_name; }                  \
    virtual std::string GetOpType() const { return "Reduce " #op_name; }      \
  };                                                                          \
  REGISTER_OPERATOR(                                                          \
      op_name,                                                                \
      ops::ReduceBaseOp,                                                      \
      __##op_name##Maker__,                                                   \
      paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>, \
      paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase,       \
                                            true>);                           \
  REGISTER_OPERATOR(op_name##_grad, ops::ReduceGradOp)

#define REGISTER_REDUCE_OP_WITHOUT_GRAD(op_name, ...)                    \
  class __##op_name##Maker__ : public ops::ReduceBaseOpMaker {           \
   protected:                                                            \
    virtual std::string GetName() const { return #op_name; }             \
    virtual std::string GetOpType() const { return "Reduce " #op_name; } \
  };                                                                     \
  REGISTER_OPERATOR(                                                     \
      op_name,                                                           \
      ops::ReduceBaseOp##__VA_ARGS__,                                    \
      __##op_name##Maker__,                                              \
      paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,    \
      paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

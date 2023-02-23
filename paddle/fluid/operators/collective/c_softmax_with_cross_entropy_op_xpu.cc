/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_softmax_with_cross_entropy_op.h"

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"
#include "paddle/phi/kernels/xpu/reduce.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CSoftmaxWithCrossEntropyOp : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      CSoftmaxWithCrossEntropyProcessGroupFunctor<DeviceContext, T> functor_;
      functor_(ctx);
    } else {
      CSoftmaxWithCrossEntropyFunctor<DeviceContext, T> functor_;
      functor_(ctx);
    }
  }
};

template <typename T>
struct CSoftmaxWithCrossEntropyProcessGroupFunctor<phi::XPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
    const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
    phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
    phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    distributed::ProcessGroup* pg = map->get(rid);
    distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::MAX;

    // allocate memory on device.
    dev_ctx.template Alloc(softmax, logits->dtype());
    dev_ctx.template Alloc(loss, logits->dtype());

    const auto& logits_dims = logits->dims();

    const int axis = logits_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, logits_dims);
    const int D = phi::funcs::SizeFromAxis(axis, logits_dims);

    phi::DenseTensor logits_2d, softmax_2d;
    framework::TensorCopy(
        *logits, ctx.GetPlace(), ctx.device_context(), &logits_2d);
    framework::TensorCopy(
        *softmax, ctx.GetPlace(), ctx.device_context(), &softmax_2d);
    logits_2d.Resize({N, D});
    softmax_2d.Resize({N, D});

    int ret = -1;
    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max = ctx.AllocateTmpTensor<T, phi::XPUContext>({N, 1}, dev_ctx);
    {
      // reduce last dim
      int dims[1] = {1};
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  XPUType* y,
                  const std::vector<int>& xdims,
                  const std::vector<int>& reduce_dims) {
        return xpu::reduce_max<XPUType>(ctx, x, y, xdims, reduce_dims);
      };
      ret = phi::XPUReduce<phi::XPUContext, XPUType>(
          dev_ctx,
          logits_2d,
          std::vector<int64_t>(dims, dims + 1),
          false,
          false,
          &logits_max,
          f);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_max");
    }

    std::vector<phi::DenseTensor> in_out;
    in_out.push_back(logits_max);
    pg->AllReduce(in_out, in_out, opts)->Synchronize();

    // step 2, obtain logit - logit_max
    {
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  const XPUType* y,
                  XPUType* z,
                  const std::vector<int>& xshape,
                  const std::vector<int>& yshape) {
        return xpu::broadcast_sub<XPUType>(ctx, x, y, z, xshape, yshape);
      };
      phi::XPUElementwise<T, XPUType>(
          dev_ctx, logits_2d, logits_max, axis, &softmax_2d, f);
    }

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, phi::XPUContext>({N, 1}, dev_ctx);
    const int start_index = rank * D;
    const int end_index = start_index + D;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());
    if (label_type == framework::proto::VarType::INT32) {
      ret = xpu::mask_label_by_index<XPUType, int32_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          labels->data<int32_t>(),
          reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
          start_index,
          end_index,
          N,
          D,
          nranks);
    } else if (label_type == framework::proto::VarType::INT64) {
      ret = xpu::mask_label_by_index<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          labels->data<int64_t>(),
          reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
          start_index,
          end_index,
          N,
          D,
          nranks);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "mask_label_by_index");

    in_out.clear();
    in_out.push_back(predicted_logits);
    opts.reduce_op = distributed::ReduceOp::SUM;
    pg->AllReduce(in_out, in_out, opts)->Synchronize();

    // step 4, obtain exp(logit)
    ret = xpu::exp<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
        reinterpret_cast<XPUType*>(softmax_2d.data<T>()),
        N * D);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "exp");

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits = ctx.AllocateTmpTensor<T, phi::XPUContext>({N, 1}, dev_ctx);
    {
      int dims[1] = {1};
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  XPUType* y,
                  const std::vector<int>& xdims,
                  const std::vector<int>& reduce_dims) {
        return xpu::reduce_sum<XPUType>(ctx, x, y, xdims, reduce_dims);
      };
      ret = phi::XPUReduce<phi::XPUContext, XPUType>(
          dev_ctx,
          softmax_2d,
          std::vector<int64_t>(dims, dims + 1),
          false,
          false,
          &sum_exp_logits,
          f);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_max");
    }

    in_out.clear();
    in_out.push_back(sum_exp_logits);
    opts.reduce_op = distributed::ReduceOp::SUM;
    pg->AllReduce(in_out, in_out, opts)->Synchronize();

    int dims[4] = {N, D, N, 1};
    ret = xpu::broadcast_div<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
        reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
        reinterpret_cast<XPUType*>(softmax_2d.data<T>()),
        std::vector<int64_t>(dims, dims + 2),
        std::vector<int64_t>(dims + 2, dims + 4));
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_div");

    ret = xpu::log<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
        reinterpret_cast<XPUType*>(sum_exp_logits.data<T>()),
        N * 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "log");
    ret = xpu::sub<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
        reinterpret_cast<const XPUType*>(predicted_logits.data<T>()),
        reinterpret_cast<XPUType*>(loss->data<T>()),
        N * 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "sub");

    framework::TensorCopy(
        softmax_2d, ctx.GetPlace(), ctx.device_context(), softmax);
  }
};

template <typename T>
struct CSoftmaxWithCrossEntropyFunctor<phi::XPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
    const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
    phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
    phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const auto& place = ctx.GetPlace();
    const auto& comm = platform::BKCLCommContext::Instance().Get(rid, place);
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

    // use global calculate stream
    const auto stream = static_cast<phi::XPUContext*>(
                            platform::DeviceContextPool::Instance().Get(place))
                            ->stream();

    // allocate memory on device.
    dev_ctx.template Alloc(softmax, logits->dtype());
    dev_ctx.template Alloc(loss, logits->dtype());

    const auto& logits_dims = logits->dims();

    const int axis = logits_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, logits_dims);
    const int D = phi::funcs::SizeFromAxis(axis, logits_dims);

    phi::DenseTensor logits_2d, softmax_2d;
    framework::TensorCopy(
        *logits, ctx.GetPlace(), ctx.device_context(), &logits_2d);
    framework::TensorCopy(
        *softmax, ctx.GetPlace(), ctx.device_context(), &softmax_2d);
    logits_2d.Resize({N, D});
    softmax_2d.Resize({N, D});

    int ret = -1;
    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max = ctx.AllocateTmpTensor<T, phi::XPUContext>({N, 1}, dev_ctx);
    void* logits_max_buff = logits_max.data<T>();
    {
      int dims[1] = {1};
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  XPUType* y,
                  const std::vector<int>& xdims,
                  const std::vector<int>& reduce_dims) {
        return xpu::reduce_max<XPUType>(ctx, x, y, xdims, reduce_dims);
      };
      ret = phi::XPUReduce<phi::XPUContext, XPUType>(
          dev_ctx,
          logits_2d,
          std::vector<int64_t>(dims, dims + 1),
          false,
          false,
          &logits_max,
          f);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_max");
    }
    PADDLE_ENFORCE_XPU_SUCCESS(
        bkcl_all_reduce(comm->comm(),
                        logits_max_buff,
                        logits_max_buff,
                        logits_max.numel(),
                        platform::ToBKCLDataType(
                            framework::TransToProtoVarType(logits_max.dtype())),
                        BKCL_MAX,
                        stream));
    xpu_wait(stream);

    // step 2, obtain logit - logit_max
    {
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  const XPUType* y,
                  XPUType* z,
                  const std::vector<int>& xshape,
                  const std::vector<int>& yshape) {
        return xpu::broadcast_sub<XPUType>(ctx, x, y, z, xshape, yshape);
      };
      phi::XPUElementwise<T, XPUType>(
          dev_ctx, logits_2d, logits_max, axis, &softmax_2d, f);
    }

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, phi::XPUContext>({N, 1}, dev_ctx);
    void* predict_logits_buff = predicted_logits.data<T>();
    ret = xpu::constant<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
        N,
        0.0);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
    const int start_index = rank * D;
    const int end_index = start_index + D;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());
    if (label_type == framework::proto::VarType::INT32) {
      ret = xpu::mask_label_by_index<XPUType, int32_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          labels->data<int32_t>(),
          reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
          start_index,
          end_index,
          N,
          D,
          nranks);
    } else if (label_type == framework::proto::VarType::INT64) {
      ret = xpu::mask_label_by_index<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          labels->data<int64_t>(),
          reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
          start_index,
          end_index,
          N,
          D,
          nranks);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "mask_label_by_index");

    PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_reduce(
        comm->comm(),
        predict_logits_buff,
        predict_logits_buff,
        predicted_logits.numel(),
        platform::ToBKCLDataType(
            framework::TransToProtoVarType(predicted_logits.dtype())),
        BKCL_ADD,
        stream));
    xpu_wait(stream);

    // step 4, obtain exp(logit)
    ret = xpu::exp<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
        reinterpret_cast<XPUType*>(softmax_2d.data<T>()),
        N * D);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "exp");

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits = ctx.AllocateTmpTensor<T, phi::XPUContext>({N, 1}, dev_ctx);
    {
      int dims[1] = {1};
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  XPUType* y,
                  const std::vector<int>& xdims,
                  const std::vector<int>& reduce_dims) {
        return xpu::reduce_sum<XPUType>(ctx, x, y, xdims, reduce_dims);
      };
      ret = phi::XPUReduce<phi::XPUContext, XPUType>(
          dev_ctx,
          softmax_2d,
          std::vector<int64_t>(dims, dims + 1),
          false,
          false,
          &sum_exp_logits,
          f);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_max");
    }

    void* sum_exp_logits_buff = sum_exp_logits.data<T>();
    PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_reduce(
        comm->comm(),
        sum_exp_logits_buff,
        sum_exp_logits_buff,
        sum_exp_logits.numel(),
        platform::ToBKCLDataType(
            framework::TransToProtoVarType(sum_exp_logits.dtype())),
        BKCL_ADD,
        stream));
    xpu_wait(stream);

    {
      int dims[4] = {N, D, N, 1};
      ret = xpu::broadcast_div<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
          reinterpret_cast<XPUType*>(softmax_2d.data<T>()),
          std::vector<int64_t>(dims, dims + 2),
          std::vector<int64_t>(dims + 2, dims + 4));
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_div");
    }

    ret = xpu::log<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
        reinterpret_cast<XPUType*>(sum_exp_logits.data<T>()),
        N * 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "log");
    ret = xpu::sub<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
        reinterpret_cast<const XPUType*>(predicted_logits.data<T>()),
        reinterpret_cast<XPUType*>(loss->data<T>()),
        N * 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "sub");

    framework::TensorCopy(
        softmax_2d, ctx.GetPlace(), ctx.device_context(), softmax);
  }
};

template <typename DeviceContext, typename T>
class CSoftmaxWithCrossEntropyGrad : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using XPUType = typename XPUTypeTrait<T>::Type;
    const phi::DenseTensor* labels = context.Input<phi::DenseTensor>("Label");
    const phi::DenseTensor* loss_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    phi::DenseTensor* logit_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Logits"));
    const phi::DenseTensor* softmax =
        context.Input<phi::DenseTensor>("Softmax");
    const int rank = context.Attr<int>("rank");
    auto& dev_ctx = context.template device_context<DeviceContext>();

    if (logit_grad != softmax) {
      framework::TensorCopy(
          *softmax, context.GetPlace(), context.device_context(), logit_grad);
    }
    const auto softmax_dims = softmax->dims();
    const int axis = softmax_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, softmax_dims);
    const int D = phi::funcs::SizeFromAxis(axis, softmax_dims);

    const int start_index = rank * D;
    const int end_index = start_index + D;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());

    int ret = 0;
    if (label_type == framework::proto::VarType::INT32) {
      ret = xpu::mask_label_by_index_grad<XPUType, int32_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(loss_grad->data<T>()),
          labels->data<int32_t>(),
          reinterpret_cast<XPUType*>(logit_grad->data<T>()),
          start_index,
          end_index,
          N,
          D);
    } else if (label_type == framework::proto::VarType::INT64) {
      ret = xpu::mask_label_by_index_grad<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(loss_grad->data<T>()),
          labels->data<int64_t>(),
          reinterpret_cast<XPUType*>(logit_grad->data<T>()),
          start_index,
          end_index,
          N,
          D);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "mask_label_by_index_grad");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(c_softmax_with_cross_entropy,
                       ops::CSoftmaxWithCrossEntropyOp<phi::XPUContext, float>);

REGISTER_OP_XPU_KERNEL(
    c_softmax_with_cross_entropy_grad,
    ops::CSoftmaxWithCrossEntropyGrad<phi::XPUContext, float>);

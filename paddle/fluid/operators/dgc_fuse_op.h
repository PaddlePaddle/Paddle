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
#include <vector>

#include "dgc/dgc.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

inline float get_period_sparcity(const std::vector<float>& sparsity,
                                 float cur_step, float rampup_steps) {
  PADDLE_ENFORCE_GE(static_cast<int>(cur_step), 0,
                    platform::errors::InvalidArgument(
                        "DGC current step=%d, but it must >= 0, "
                        "please submit issue in github",
                        static_cast<int>(cur_step)));

  size_t idx = static_cast<int>(cur_step * sparsity.size() / rampup_steps);
  if (idx >= sparsity.size()) {
    idx = sparsity.size() - 1;
  }

  PADDLE_ENFORCE_LT(
      idx, sparsity.size(),
      platform::errors::OutOfRange(
          "sparsity index out of bounds. idx=%d >= sparsity.size=%d", idx,
          sparsity.size()));
  return sparsity[idx];
}

template <typename DeviceContext, typename T>
class DGCFuseOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto u = ctx.Input<framework::Tensor>("U");
    auto v = ctx.Input<framework::Tensor>("V");
    auto g = ctx.Input<framework::Tensor>("Grad");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::CUDADeviceContext>();
    framework::Tensor grad_out =
        ctx.AllocateTmpTensor<T, DeviceContext>(g->dims(), dev_ctx);

    // attrs
    float m = ctx.Attr<float>("m");
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    auto sparsity = ctx.Attr<std::vector<float>>("sparsity");
    auto rampup_begin_step = ctx.Attr<float>("rampup_begin_step");
    auto rampup_step = ctx.Attr<float>("rampup_step");

    // nranks
    auto nranks_tensor = ctx.Input<framework::Tensor>("nranks");
    const int nranks = static_cast<const int>(*nranks_tensor->data<float>());
    PADDLE_ENFORCE_GT(nranks, 1,
                      platform::errors::PreconditionNotMet(
                          "DGC is not useful when num_trainers <= 1. Please "
                          "use multi card or multi machine GPU"));

    // regularization
    auto p = ctx.Input<framework::Tensor>("Param");
    float regular_coeff = ctx.Attr<float>("regular_coeff");
    int regular_type = ctx.Attr<int>("regular_type");

    auto p_e = framework::EigenVector<T>::Flatten(*p);
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto grad_out_e = framework::EigenVector<T>::Flatten(grad_out);

    auto place = ctx.GetPlace();
    auto out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(g->dims(), ctx.GetPlace());

    // auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto& eigen_ctx = *dev_ctx.eigen_device();

    // NOTE. In paddle, loss has divided by nranks. Because dgc_op is before
    // allreduce, so local regular_coeff need div nranks too. But now we
    // multi grad with nranks in dgc_op, in that case regular_coeff don't
    // need to /nranks, can prevent precision loss. For coeff often equal
    // with 1e-4, if nranks=32, coeff/nranks will be 3.125e-6, the numerical
    // accuracy of coeff/nranks will be too low.
    PADDLE_ENFORCE_EQ(regular_type >= 0 && regular_type <= 2, true,
                      platform::errors::InvalidArgument(
                          "DGC only support one of None|L1Decay|L2Decay "
                          "Regularization for now."));
    if (regular_type == 0) {
      grad_out_e.device(eigen_ctx) = (1.0 * nranks) * g_e;
    } else if (regular_type == 1) {
      // L1Decay. grad = grad + coeff * sign(param)
      grad_out_e.device(eigen_ctx) =
          (1.0 * nranks) * g_e + regular_coeff * p_e.sign();
    } else if (regular_type == 2) {
      // L2Decay. grad = grad + coeff * param
      grad_out_e.device(eigen_ctx) = (1.0 * nranks) * g_e + regular_coeff * p_e;
    }

    // current step
    auto current_step_tensor = ctx.Input<framework::Tensor>("current_step");
    const float* current_step = current_step_tensor->data<float>();

    const int ring_id = ctx.Attr<int>("ring_id");
    auto comm =
        platform::NCCLCommContext::Instance().Get(ring_id, ctx.GetPlace());
    auto stream = comm->stream();

    if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
      VLOG(10) << "current_step:" << *current_step
               << " < rampup_begin_step:" << rampup_begin_step
               << " so does't use dgc";
      framework::TensorCopySync(*g, place, out);
      // do allreduce
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      auto compute_stream =
          static_cast<platform::CUDADeviceContext*>(
              platform::DeviceContextPool::Instance().Get(place))
              ->stream();
      auto comm_stream = stream;
      auto event = platform::NCCLCommContext::Instance()
                       .Get(ring_id, ctx.GetPlace())
                       ->compute_event();

// compute_stream-->event-->comm_stream
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event, compute_stream));
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(comm_stream, event, 0));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, compute_stream));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(comm_stream, event, 0));
#endif
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU."));
#endif

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
          grad_out.data<T>(), out->data<T>(), grad_out.numel(),
          platform::ToNCCLDataType(
              framework::TransToProtoVarType(grad_out.dtype())),
          ncclSum, comm->comm(), stream));

      // memory::Copy(ctx.GetPlace(), raw_g->data<T>(),
      //              ctx.GetPlace(),  out->data<T>(),
      //              raw_g->numel(), stream);

      return;
    }

    float ratio =
        1 - get_period_sparcity(
                sparsity, static_cast<float>(*current_step - rampup_begin_step),
                rampup_step);
    PADDLE_ENFORCE_GE(
        ratio, 0.0,
        platform::errors::InvalidArgument("DGC sparsity ratio must >= 0"));
    PADDLE_ENFORCE_LT(
        ratio, 1.0,
        platform::errors::InvalidArgument("DGC sparsity ratio must < 1"));
    int k = static_cast<int>(g->numel() * ratio);

    VLOG(10) << "m:" << m << ", use_nesterov:" << use_nesterov
             << ", rampup_begin_step:" << rampup_begin_step
             << ", rampup_step:" << rampup_step
             << ",  current_step:" << *current_step << ", ratio:" << ratio
             << ", k:" << k << ", nranks:" << nranks;

    auto u_out = ctx.Output<framework::Tensor>("U_out");
    u_out->mutable_data<T>(u->dims(), ctx.GetPlace());
    u_out->ShareDataWith(*u);
    auto v_out = ctx.Output<framework::Tensor>("V_out");
    v_out->mutable_data<T>(v->dims(), ctx.GetPlace());
    v_out->ShareDataWith(*v);

    // FIXME(gongwb): use cublas.
    auto u_out_e = framework::EigenVector<T>::Flatten(*u_out);
    auto u_e = framework::EigenVector<T>::Flatten(*u);

    // calc local momentum from global momentum
    // NOTE. If grad not multi nranks, need add below code.
    // if (static_cast<int>(*current_step) ==
    //     static_cast<int>(rampup_begin_step)) {
    //   u_out_e.device(eigen_ctx) = (1.0 / nranks) * u_e;
    // }

    if (use_nesterov) {
      // u = m * (u + g)
      u_out_e.device(eigen_ctx) = m * (u_e + grad_out_e);

      // v = u + v + g
      ElementwiseComputeEx<phi::funcs::AddFunctor<T>, DeviceContext, T>(
          ctx, u, v, 0, phi::funcs::AddFunctor<T>(), v_out);

      ElementwiseComputeEx<phi::funcs::AddFunctor<T>, DeviceContext, T>(
          ctx, g, v, 0, phi::funcs::AddFunctor<T>(), v_out);
    } else {
      // u = m * u + g
      u_out_e.device(eigen_ctx) = m * u_e + grad_out_e;

      // v = u + v
      ElementwiseComputeEx<phi::funcs::AddFunctor<T>, DeviceContext, T>(
          ctx, u, v, 0, phi::funcs::AddFunctor<T>(), v_out);
    }

    framework::Tensor encode_grad_out =
        ctx.AllocateTmpTensor<T, DeviceContext>({2 * k}, dev_ctx);
    framework::Tensor gather_buff =
        ctx.AllocateTmpTensor<T, DeviceContext>({2 * k * nranks}, dev_ctx);

    T* v_out_data = v_out->mutable_data<T>(ctx.GetPlace());
    T* u_out_data = u_out->mutable_data<T>(ctx.GetPlace());
    T* encode_grad_out_data = encode_grad_out.data<T>();

    int buf_size = paddle::communication::dgc::get_buffer_size(k);
    auto tmp_ious_data = memory::Alloc(dev_ctx, buf_size);
    void* buf = reinterpret_cast<void*>(tmp_ious_data->ptr());

    if (!paddle::communication::dgc::k_select(
            static_cast<void*>(encode_grad_out_data), k, v_out_data,
            static_cast<int>(v_out->numel()), buf, dev_ctx.stream(),
            u_out_data)) {
      // TODO(weihang): owner should polish this error message
      PADDLE_THROW(platform::errors::InvalidArgument(
          "V_out numel error, V_out numel is %d.", v_out->numel()));
    }

    phi::funcs::SetConstant<DeviceContext, T> tset;
    tset(dev_ctx, &grad_out, static_cast<T>(0));

    // do dgc comm
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto compute_stream =
        static_cast<platform::CUDADeviceContext*>(
            platform::DeviceContextPool::Instance().Get(place))
            ->stream();
    auto comm_stream = stream;
    auto event = platform::NCCLCommContext::Instance()
                     .Get(ring_id, ctx.GetPlace())
                     ->compute_event();

// compute_stream-->event-->comm_stream
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event, compute_stream));
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(comm_stream, event, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, compute_stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(comm_stream, event, 0));
#endif
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif

    framework::TensorCopySync(*g, place, out);

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        encode_grad_out.data<T>(), gather_buff.data<T>(),
        static_cast<int64_t>(2 * k),
        platform::ToNCCLDataType(
            framework::TransToProtoVarType(grad_out.dtype())),
        comm->comm(), stream));

    PADDLE_ENFORCE_EQ(
        paddle::communication::dgc::sparseReduce(
            reinterpret_cast<void*> gather_buff.data<T>(), k, out->data<T>(),
            out->numel(), nranks, stream),
        true, platform::errors::Unavailable("Calling sparseReduce() failed."));

    // memory::Copy(ctx.GetPlace(), raw_g->data<T>(),
    //              ctx.GetPlace(),  out->data<T>(),
    //              grad_out.numel(), stream);
  }
};
}  // namespace operators
}  // namespace paddle

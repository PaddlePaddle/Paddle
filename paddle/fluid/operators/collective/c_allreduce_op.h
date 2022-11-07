/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/api/include/tensor.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) ||          \
    defined(PADDLE_WITH_ASCEND_CL) || defined(PADDLE_WITH_XPU_BKCL) || \
    defined(PADDLE_WITH_CNCL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include <gloo/allreduce.h>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/device/npu/hccl_helper.h"
#endif

#if defined(PADDLE_WITH_CNCL)
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
DECLARE_bool(hccl_check_nan);
#endif

namespace paddle {
namespace operators {

enum ReduceType { kRedSum, kRedMax, kRedMin, kRedProd };

class CAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (var_name == "Cond") {
      return expected_kernel_type;
    } else {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();
    int64_t send_numel = in->numel();
    const T* send_buff = in->data<T>();
    T* recv_buff = out->mutable_data<T>(in->dims(), place);
    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(),
        true,
        platform::errors::PreconditionNotMet(
            "You must initialize the gloo environment first to use it."));
    gloo::AllreduceOptions opts(gloo->GetContext());
    opts.setInput(const_cast<T*>(send_buff), send_numel);
    opts.setOutput(recv_buff, send_numel);
    switch (red_type) {
      case kRedSum:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::sum<T>));
        break;
      case kRedMax:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::max<T>));
        break;
      case kRedMin:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::min<T>));
        break;
      case kRedProd:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::product<T>));
        break;
      default:
        PADDLE_ENFORCE_EQ(true,
                          false,
                          platform::errors::InvalidArgument(
                              "Invalid reduce type: %d.", red_type));
    }
    gloo::allreduce(opts);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

#if defined(PADDLE_WITH_ASCEND_CL)
// return true if found_nan or return false;
inline bool ContainsNan(const paddle::platform::NPUDeviceContext& dev_ctx,
                        aclrtStream stream,
                        const phi::DenseTensor* in) {
  using Tensor = phi::DenseTensor;
  Tensor out(in->type());

  Tensor mean(in->type());
  mean.Resize({1});
  mean.mutable_data<float>(dev_ctx.GetPlace());
  std::vector<int> axes;
  for (int i = 0; i < in->dims().size(); ++i) {
    axes.push_back(i);
  }

  std::vector<float> vec;
  try {
    const auto& runner_mean = paddle::operators::NpuOpRunner(
        "ReduceMeanD", {*in}, {mean}, {{"axes", axes}, {"keep_dims", false}});
    paddle::framework::TensorToVector(mean, dev_ctx, &vec);
  } catch (...) {
    LOG(WARNING) << "ContainsNan catch exception";
    return true;
  }

  VLOG(4) << "reducemeand result:" << vec[0];
  if (std::isnan(static_cast<float>(vec[0]))) {
    LOG(WARNING) << "ContainsNan detects nan";
    return true;
  }

  if (std::isinf(static_cast<float>(vec[0]))) {
    LOG(WARNING) << "ContainsNan detects inf";
  }

  return false;
}

#endif

template <ReduceType red_type, typename T>
class CAllReduceOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    if (ctx.HasInput("Cond")) {
      auto cond = ctx.Input<phi::DenseTensor>("Cond");
      auto place = cond->place();
      PADDLE_ENFORCE_EQ(platform::is_cpu_place(place),
                        true,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` tensor should be on cpu place"));
      PADDLE_ENFORCE_EQ(cond->numel(),
                        1,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` should be shape [1]"));
      if (!cond->data<bool>()[0]) {
        VLOG(4) << "Skip all reduce Op since cond is 0";
        return;
      }
    }

    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();
    HcclDataType dtype =
        platform::ToHCCLDataType(framework::TransToProtoVarType(in->dtype()));
    int64_t numel = in->numel();

    void* sendbuff = reinterpret_cast<void*>(const_cast<T*>(in->data<T>()));
    out->mutable_data<T>(in->dims(), ctx.GetPlace());
    void* recvbuff = reinterpret_cast<void*>(out->data<T>());

    int ring_id = ctx.Attr<int>("ring_id");
    std::string group =
        std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    auto comm =
        paddle::platform::HCCLCommContext::Instance().Get(ring_id, place);

    aclrtStream stream = nullptr;
    auto dev_ctx = static_cast<platform::NPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = dev_ctx->stream();
    } else {
      stream = comm->stream();
    }

    HcclReduceOp hccl_red_type = HCCL_REDUCE_SUM;
    switch (red_type) {
      case kRedSum:
        hccl_red_type = HCCL_REDUCE_SUM;
        break;

      case kRedMax:
        hccl_red_type = HCCL_REDUCE_MAX;
        break;

      case kRedMin:
        hccl_red_type = HCCL_REDUCE_MIN;
        break;

      case kRedProd:
        hccl_red_type = HCCL_REDUCE_PROD;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }

    VLOG(3) << "hccl allreduce, parameter is: "
            << "input num: " << in->dims() << "dtype: " << dtype
            << "hccl_red_type: " << hccl_red_type << ", group is: " << group
            << ", sendbuff:" << sendbuff << ", recvbuff:" << recvbuff
            << ", out_size:" << out->memory_size()
            << ", use_calc_stream:" << ctx.Attr<bool>("use_calc_stream")
            << ", stream:" << stream;

    phi::DenseTensor tmp;
    tmp.mutable_data<float>({8}, ctx.GetPlace());

    bool found_nan = false;

    auto d_type = framework::TransToProtoVarType(in->dtype());
    switch (d_type) {
      case framework::proto::VarType::FP16: {
        break;
      }
      case framework::proto::VarType::FP32: {
        if (FLAGS_hccl_check_nan) {
          VLOG(3) << "prepare to FoundNanInf";
          // NOTE: performance relating, DO NOT REMOVE!
          ContainsNan(*dev_ctx, dev_ctx->stream(), in);
        }
        break;
      }
      default:
        break;
    }

    if (found_nan) {
      T inf = static_cast<T>(std::numeric_limits<float>::infinity());
      VLOG(4) << "fill input data constant inf";
      auto dims = in->dims();
      auto mutable_in = const_cast<phi::DenseTensor*>(in);
      FillNpuTensorWithConstant<T>(mutable_in, inf);
      mutable_in->Resize(dims);
    }

    VLOG(3) << "hccl allreduce, parameter is: "
            << "input num: " << numel << "dtype: " << dtype
            << "hccl_red_type: " << hccl_red_type << ", group is: " << group
            << ", sendbuff:" << sendbuff << ", recvbuff:" << recvbuff
            << ", out_size:" << out->memory_size();

    PADDLE_ENFORCE_NPU_SUCCESS(
        platform::dynload::HcclAllReduce(sendbuff,
                                         recvbuff,
                                         numel,
                                         dtype,
                                         hccl_red_type,
                                         comm->comm(),
                                         reinterpret_cast<void*>(stream)));

    out->Resize(in->dims());
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_XPU_BKCL)
    if (ctx.HasInput("Cond")) {
      auto cond = ctx.Input<phi::DenseTensor>("Cond");
      auto place = cond->place();
      PADDLE_ENFORCE_EQ(platform::is_cpu_place(place),
                        true,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` tensor should be on cpu place"));
      PADDLE_ENFORCE_EQ(cond->numel(),
                        1,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` should be shape [1]"));
      if (!cond->data<bool>()[0]) {
        VLOG(4) << "Skip all reduce Op since cond is 0";
        return;
      }
    }

    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();
    BKCLDataType dtype =
        platform::ToBKCLDataType(framework::TransToProtoVarType(in->dtype()));
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::BKCLCommContext::Instance().Get(rid, place);

    XPUStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                   ->x_context()
                   ->xpu_stream;
    } else {
      stream = comm->stream();
    }

    BKCLOp bkcl_red_type = BKCL_ADD;
    switch (red_type) {
      case kRedSum:
        bkcl_red_type = BKCL_ADD;
        break;

      case kRedMax:
        bkcl_red_type = BKCL_MAX;
        break;

      case kRedMin:
        bkcl_red_type = BKCL_MIN;
        break;

      case kRedProd:
        bkcl_red_type = BKCL_PRODUCT;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }

    PADDLE_ENFORCE_EQ(
        bkcl_all_reduce(comm->comm(),
                        sendbuff,
                        recvbuff,
                        numel,
                        dtype,
                        bkcl_red_type,
                        stream),
        BKCL_SUCCESS,
        platform::errors::PreconditionNotMet("BKCL all reduce failed"));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should be compiled with XPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    if (ctx.HasInput("Cond")) {
      auto cond = ctx.Input<phi::DenseTensor>("Cond");
      auto place = cond->place();
      PADDLE_ENFORCE_EQ(platform::is_cpu_place(place),
                        true,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` tensor should be on cpu place"));
      PADDLE_ENFORCE_EQ(cond->numel(),
                        1,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` should be shape [1]"));
      if (!cond->data<bool>()[0]) {
        VLOG(4) << "Skip all reduce Op since cond is 0";
        return;
      }
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int rid = ctx.Attr<int>("ring_id");

    auto place = ctx.GetPlace();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*in);
      out_tensor.push_back(*out);

      distributed::AllreduceOptions opts;
      switch (red_type) {
        case kRedSum:
          opts.reduce_op = distributed::ReduceOp::SUM;
          break;

        case kRedMax:
          opts.reduce_op = distributed::ReduceOp::MAX;
          break;

        case kRedMin:
          opts.reduce_op = distributed::ReduceOp::MIN;
          break;

        case kRedProd:
          opts.reduce_op = distributed::ReduceOp::PRODUCT;
          break;

        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Invalid reduce type: %d", red_type));
      }

      auto task = pg->AllReduce(in_tensor, out_tensor, opts);
      task->Wait();
      return;
    }

    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);

    gpuStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::GPUContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    VLOG(10) << "all reduce buffer:" << sendbuff << ", numel:" << numel
             << ", redtype:" << static_cast<int>(red_type)
             << ", dtype:" << dtype << ", comm:" << comm
             << ", stream:" << stream;

    ncclRedOp_t nccl_red_type = ncclSum;
    switch (red_type) {
      case kRedSum:
        nccl_red_type = ncclSum;
        break;

      case kRedMax:
        nccl_red_type = ncclMax;
        break;

      case kRedMin:
        nccl_red_type = ncclMin;
        break;

      case kRedProd:
        nccl_red_type = ncclProd;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, nccl_red_type, comm->comm(), stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_CNCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    if (ctx.HasInput("Cond")) {
      auto cond = ctx.Input<phi::DenseTensor>("Cond");
      auto place = cond->place();
      PADDLE_ENFORCE_EQ(platform::is_cpu_place(place),
                        true,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` tensor should be on cpu place"));
      PADDLE_ENFORCE_EQ(cond->numel(),
                        1,
                        platform::errors::PreconditionNotMet(
                            "The input `cond` should be shape [1]"));
      if (!cond->data<bool>()[0]) {
        VLOG(4) << "Skip all reduce Op since cond is 0";
        return;
      }
    }

    auto place = ctx.GetPlace();
    cnclDataType_t dtype =
        platform::ToCNCLDataType(framework::TransToProtoVarType(in->dtype()));
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::CNCLCommContext::Instance().Get(rid, place);

    mluStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::MLUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    cnclReduceOp_t cncl_red_type = cnclSum;
    switch (red_type) {
      case kRedSum:
        cncl_red_type = cnclSum;
        break;

      case kRedMax:
        cncl_red_type = cnclMax;
        break;

      case kRedMin:
        cncl_red_type = cnclMin;
        break;

      case kRedProd:
        cncl_red_type = cnclProd;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }

    PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(
        sendbuff, recvbuff, numel, dtype, cncl_red_type, comm->comm(), stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with MLU."));
#endif
  }
};

class CAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced.");
    AddOutput("Out", "(Tensor) the allreduced result.");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
#if defined(PADDLE_WITH_ASCEND_CL)
    AddAttr<std::string>("tag", "(string default tag) tag for all reduce.")
        .SetDefault("tag");
#endif
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<bool>(
        "use_model_parallel",
        "(bool default false) use this op with model parallel mode. In model "
        "parallel mode, the backward is c_identity which returns itself for "
        "c_allreduce_sum.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
CAllReduce %s Operator

Call collective AllReduce with reduce type %s. If input and output are
the same variable, in-place allreduce will be used.
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce
)DOC",
                               GetName(),
                               GetName()));
    ExtraMake();
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual void ExtraMake() {}
};

}  // namespace operators
}  // namespace paddle

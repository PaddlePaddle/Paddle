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
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#elif defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_XPU)
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif
#include "paddle/fluid/platform/collective_helper.h"
namespace paddle {
namespace operators {

class CAllReduceXOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {}
 protected:
  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name, const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (var_name == "Cond") {
	  return phi::KernelKey(phi::Backend::ALL_BACKEND,
						    expected_kernel_type.layout(),
						    expected_kernel_type.dtype());
    } else {
	  return phi::KernelKey(
		tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};
template <typename T>
class CAllReduceXOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_tensors = ctx.MultiInput<phi::DenseTensor>("X");
    auto out_tensors = ctx.MultiOutput<phi::DenseTensor>("Out");

    PADDLE_ENFORCE_EQ(in_tensors.size(),
        out_tensors.size(),
        platform::errors::InvalidArgument(
            "The number of CReduceX operator's input and "
            "output is not match, "
            "input number is %u, output number is %u.",
            in_tensors.size(),
            out_tensors.size()));

    int ring_id = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);

#if defined(PADDLE_WITH_NCCL)
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = dynamic_cast<phi::GPUContext *>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
#elif defined(PADDLE_WITH_XPU_BKCL)
    auto comm = platform::BKCLCommContext::Instance().Get(ring_id, place);
    XPUStream stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                       ->x_context()
                       ->xpu_stream;
#else
    PADDLE_THROW("PaddlePaddle should compile with NCCL OR XPU.");
#endif

    // Init the output as input
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      auto &out_tensor = out_tensors[i];
      if (out_tensor->IsInitialized()) {
        PADDLE_ENFORCE_EQ(out_tensor->numel(),
            in_tensors[i]->numel(),
            platform::errors::InvalidArgument(
            "The number of CReduceX operator's X[%u] and "
            "Out[%u] is not match, "
            "input numel is %u, output numel is %u.",
            i,
            i,
            out_tensor->numel(),
            in_tensors[i]->numel()));
      } else {
        out_tensor->Resize(in_tensors[i]->dims());
        out_tensor->mutable_data<T>(place);
      }
    }
#if defined(PADDLE_WITH_NCCL)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
#endif
    // allreduce sum data
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      auto &in_tensor = in_tensors[i];
      auto &out_tensor = out_tensors[i];
      int64_t numel = in_tensor->numel();
      const T *sendbuff = in_tensor->data<T>();
      T *recvbuff = out_tensor->mutable_data<T>(place);
#if defined(PADDLE_WITH_NCCL)
      ncclDataType_t nccl_dtype =
                  platform::ToNCCLDataType(framework::TransToProtoVarType(in_tensor->dtype()));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::ncclAllReduce(
              sendbuff,
              recvbuff,
              numel,
              nccl_dtype,
              ncclSum,
              comm->comm(),
              stream));
#elif defined(PADDLE_WITH_XPU_BKCL)
      BKCLDataType bkcl_dtype =
                  platform::ToBKCLDataType(framework::TransToProtoVarType(in_tensor->dtype()));
      PADDLE_ENFORCE_EQ(
          bkcl_all_reduce(comm->comm(),
              sendbuff,
              recvbuff,
              numel,
              bkcl_dtype,
              BKCL_ADD,
              stream),
              BKCL_SUCCESS,
              platform::errors::PreconditionNotMet("BKCL all reduce failed"));
#endif
    }
#if defined(PADDLE_WITH_NCCL)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
#endif
  }
};
class CAllReduceXOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X",
             "(vector<LoDTensor>) The input tensors of callreduce_x_tensor "
             "operator.")
        .AsDuplicable();
    AddOutput("Out",
              "(LoDTensor) The output tensor ")
        .AsDuplicable();
    AddAttr<int>("ring_id", "(int default -1) nccl ring id num.")
        .SetDefault(-1);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
CAllReduceX %s Operator

Call collective ReduceX with reduce type %s. If input and output are
the same variable, in-place allreduce will be used.
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce
)DOC",
                               GetName(), GetName()));
  }
 protected:
  virtual std::string GetName() { return "ReduceX"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_allreduce_xsum, ops::CAllReduceXOp,
                  ops::CAllReduceXOpMaker);
REGISTER_OP_CPU_KERNEL(c_allreduce_xsum, ops::CAllReduceXOpKernel<float>,
                       ops::CAllReduceXOpKernel<double>,
                       ops::CAllReduceXOpKernel<int>,
                       ops::CAllReduceXOpKernel<int64_t>,
                       ops::CAllReduceXOpKernel<plat::float16>);
#if defined(PADDLE_WITH_NCCL)
REGISTER_OP_CUDA_KERNEL(c_allreduce_xsum, ops::CAllReduceXOpKernel<float>,
                        ops::CAllReduceXOpKernel<double>,
                        ops::CAllReduceXOpKernel<int>,
                        ops::CAllReduceXOpKernel<int64_t>,
                        ops::CAllReduceXOpKernel<plat::float16>);
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
REGISTER_OP_XPU_KERNEL(c_allreduce_xsum, ops::CAllReduceXOpKernel<float>,
                        ops::CAllReduceXOpKernel<double>,
                        ops::CAllReduceXOpKernel<int>,
                        ops::CAllReduceXOpKernel<int64_t>,
                        ops::CAllReduceXOpKernel<plat::float16>);
#endif

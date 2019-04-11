/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenseshashernless required by applicable law or agreed
to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <functional>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using platform::Communicator;
using framework::LoDTensor;

template <typename Type>
class NCCLTypeWrapper;

template <>
class NCCLTypeWrapper<float> {
 public:
  static const ncclDataType_t type = ncclFloat;
};

template <>
class NCCLTypeWrapper<double> {
 public:
  static const ncclDataType_t type = ncclDouble;
};

template <typename T>
class NCCLAllReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    auto* comm = ctx.Input<Communicator>("Communicator");
    std::string reduction = ctx.Attr<std::string>("reduction");

    ncclRedOp_t reduction_op_ = ncclSum;
    if (reduction == "ncclMin") {
      reduction_op_ = ncclMin;
    } else if (reduction == "ncclMax") {
      reduction_op_ = ncclMax;
    } else if (reduction == "ncclSum") {
      reduction_op_ = ncclSum;
    } else if (reduction == "ncclProd") {
      reduction_op_ = ncclProd;
    } else {
      PADDLE_THROW("Invalid reduction. default ncclSum.");
    }
    // device id
    int gpu_id = boost::get<platform::CUDAPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(gpu_id);
    VLOG(3) << "gpu : "
            << " invoke allreduce. send " << x->numel() << " recv "
            << out->numel();
    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        x->data<T>(), out->mutable_data<T>(ctx.GetPlace()), out->numel(),
        NCCLTypeWrapper<T>::type, reduction_op_, comm->comms().at(idx),
        ctx.cuda_device_context().stream()));
    VLOG(3) << "gpu : "
            << " finished allreduce. send " << x->numel() << " recv "
            << out->numel();
  }
};

template <typename T>
class NCCLReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto x = ctx.Input<LoDTensor>("X");  // x0, x1, x2
    auto out = ctx.Output<LoDTensor>("Out");
    auto* comm = ctx.Input<Communicator>("Communicator");
    int root = ctx.Attr<int>("root");
    std::string reduction = ctx.Attr<std::string>("reduction");

    ncclRedOp_t reduction_op_ = ncclSum;
    if (reduction == "ncclMin") {
      reduction_op_ = ncclMin;
    } else if (reduction == "ncclMax") {
      reduction_op_ = ncclMax;
    } else if (reduction == "ncclSum") {
      reduction_op_ = ncclSum;
    } else if (reduction == "ncclProd") {
      reduction_op_ = ncclProd;
    } else {
      PADDLE_THROW("Invalid reduction. default ncclSum.");
    }
    // device id
    int gpu_id = boost::get<platform::CUDAPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(gpu_id);
    T* recvbuffer = nullptr;
    if (root == gpu_id) {
      recvbuffer = out->mutable_data<T>(ctx.GetPlace());
    } else {
      out->Resize(framework::make_ddim({0}));
    }
    VLOG(3) << "gpu : " << gpu_id << " invoke reduce. send " << x->numel()
            << " recv " << out->numel();
    PADDLE_ENFORCE(platform::dynload::ncclReduce(
        x->data<T>(), recvbuffer, x->numel(), NCCLTypeWrapper<T>::type,
        reduction_op_, root, comm->comms().at(idx),
        ctx.cuda_device_context().stream()));
    VLOG(3) << "gpu : " << gpu_id << " finished reduce. send " << x->numel()
            << " recv " << out->numel();
  }
};

template <typename T>
class NCCLBcastKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    int root = ctx.Attr<int>("root");
    auto* comm = ctx.Input<Communicator>("Communicator");
    // device id
    int gpu_id = boost::get<platform::CUDAPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(gpu_id);
    if (idx == root) {
      auto* x = ctx.Input<LoDTensor>("X");
      VLOG(3) << "gpu : " << gpu_id << " invoke Bcast. send " << x->numel();
      PADDLE_ENFORCE(platform::dynload::ncclBcast(
          reinterpret_cast<void*>(const_cast<T*>(x->data<T>())), x->numel(),
          NCCLTypeWrapper<T>::type, root, comm->comms().at(idx),
          ctx.cuda_device_context().stream()));
      VLOG(3) << "gpu : " << gpu_id << " finished Bcast.";
    } else {
      auto* out = ctx.Output<LoDTensor>("Out");
      VLOG(3) << "gpu : " << gpu_id << " invoke Bcast. recv buffer "
              << framework::product(out->dims());
      PADDLE_ENFORCE(platform::dynload::ncclBcast(
          out->mutable_data<T>(ctx.GetPlace()), out->numel(),
          NCCLTypeWrapper<T>::type, root, comm->comms().at(idx),
          ctx.cuda_device_context().stream()));
      VLOG(3) << "gpu : " << gpu_id << " finished Bcast. recv " << out->numel();
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(ncclAllReduce, ops::NCCLAllReduceKernel<float>);
REGISTER_OP_CUDA_KERNEL(ncclBcast, ops::NCCLBcastKernel<float>);
REGISTER_OP_CUDA_KERNEL(ncclReduce, ops::NCCLReduceKernel<float>);

/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenseshashernless required by applicable law or agreed
to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU
#include <functional>

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/nccl/nccl_gpu_common.h"

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

    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto outs = ctx.MultiOutput<LoDTensor>("Out");

    auto* comm = ctx.Input<Communicator>("Communicator");

    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();
    // device id
    int device_id =
        boost::get<platform::GPUPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(device_id);

    for (size_t i = 0; i < ins.size(); ++i) {
      VLOG(1) << " invoke allreduce. send " << ins[i]->numel() << " recv "
              << outs[i]->numel();

      PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
          ins[i]->data<T>(), outs[i]->mutable_data<T>(ctx.GetPlace()),
          outs[i]->numel(), NCCLTypeWrapper<T>::type, ncclSum,
          comm->comms_[idx], stream));
      PADDLE_ENFORCE(cudaStreamSynchronize(stream));

      VLOG(1) << " finished allreduce. send " << ins[i]->numel() << " recv "
              << outs[i]->numel();
    }
  }
};

template <typename T>
class NCCLReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    auto ins = ctx.MultiInput<LoDTensor>("X");  // x0, x1, x2
    auto outs = ctx.MultiOutput<LoDTensor>("Out");
    int root = ctx.Attr<int>("root");

    auto* comm = ctx.Input<Communicator>("Communicator");

    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();
    // device id
    int device_id =
        boost::get<platform::GPUPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(device_id);

    auto ins_names = ctx.Inputs("X");
    std::hash<std::string> hasher;
    for (size_t i = 0; i < ins.size(); ++i) {
      if (root == -1) {
        root = hasher(ins_names[i]) % comm->comms_.size();
      }
      T* recvbuffer = nullptr;
      if (root == device_id) {
        recvbuffer = outs[i]->mutable_data<T>(ctx.GetPlace());
      }

      VLOG(1) << " invoke reduce. send " << ins[i]->numel() << " recv "
              << outs[i]->numel();

      PADDLE_ENFORCE(platform::dynload::ncclReduce(
          ins[i]->data<T>(), recvbuffer, ins[i]->numel(),
          NCCLTypeWrapper<T>::type, ncclSum, root, comm->comms_[idx], stream));
      PADDLE_ENFORCE(cudaStreamSynchronize(stream));

      VLOG(1) << " finished reduce. send " << ins[i]->numel() << " recv "
              << outs[i]->numel();
    }
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

    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();
    // device id
    int device_id =
        boost::get<platform::GPUPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(device_id);

    if (idx == root) {
      auto ins = ctx.MultiInput<LoDTensor>("X");
      for (size_t i = 0; i < ins.size(); ++i) {
        VLOG(1) << " invoke Bcast. send " << ins[i]->numel();

        PADDLE_ENFORCE(platform::dynload::ncclBcast(
            (void*)ins[i]->data<T>(), ins[i]->numel(), NCCLTypeWrapper<T>::type,
            root, comm->comms_[idx], stream));
        PADDLE_ENFORCE(cudaStreamSynchronize(stream));

        VLOG(1) << " finished Bcast.";
      }
    } else {
      auto outs = ctx.MultiOutput<LoDTensor>("Out");
      for (size_t i = 0; i < outs.size(); ++i) {
        VLOG(1) << " invoke Bcast. recv. ";

        PADDLE_ENFORCE(platform::dynload::ncclBcast(
            outs[i]->mutable_data<T>(ctx.GetPlace()), outs[i]->numel(),
            NCCLTypeWrapper<T>::type, root, comm->comms_[idx], stream));
        PADDLE_ENFORCE(cudaStreamSynchronize(stream));

        VLOG(1) << " finished Bcast. recv " << outs[i]->numel();
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(ncclAllReduce, ops::NCCLAllReduceKernel<float>);
REGISTER_OP_GPU_KERNEL(ncclBcast, ops::NCCLBcastKernel<float>);
REGISTER_OP_GPU_KERNEL(ncclReduce, ops::NCCLReduceKernel<float>);

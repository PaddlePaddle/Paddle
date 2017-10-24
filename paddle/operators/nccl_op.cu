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

#include "paddle/operators/nccl_op.h"

namespace paddle {
namespace operators {

template <typename T>
class NCCLAllReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    auto ins = ctx.MultiInput<Tensor>("X");
    auto outs = ctx.MultiOutput<Tensor>("Out");

    auto* comm = ctx.Input<Communicator>("Communicator");

    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();
    // device id
    int device_id =
        boost::get<platform::GPUPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(device_id);

    for (size_t i = 0; i < ins.size(); ++i) {
      PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
          ins[i]->data<T>(), outs[i]->mutable_data<T>(ctx.GetPlace()),
          outs[i]->numel() * sizeof(T), NCCLTypeWrapper<T>::type, ncclSum,
          comm->comms_[idx], stream));
      PADDLE_ENFORCE(cudaStreamSynchronize(stream));
    }
  }
};

template <typename T>
class NCCLReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    auto ins = ctx.MultiInput<Tensor>("X");  // x0, x1, x2
    auto outs = ctx.MultiOutput<Tensor>("Out");

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
      int root = hasher(ins_names[i]) % comm->comms_.size();
      T* recvbuffer = nullptr;
      if (root == device_id) {
        recvbuffer = outs[i]->mutable_data<T>(ctx.GetPlace());
      }
      PADDLE_ENFORCE(platform::dynload::ncclReduce(
          ins[i]->data<T>(), recvbuffer, ins[i]->numel(),
          NCCLTypeWrapper<T>::type, ncclSum, root, comm->comms_[idx], stream));
      PADDLE_ENFORCE(cudaStreamSynchronize(stream));
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
      auto ins = ctx.MultiInput<Tensor>("X");
      for (size_t i = 0; i < ins.size(); ++i) {
        PADDLE_ENFORCE(platform::dynload::ncclBcast(
            (void*)ins[i]->data<T>(), ins[i]->numel(), NCCLTypeWrapper<T>::type,
            root, comm->comms_[idx], stream));
        PADDLE_ENFORCE(cudaStreamSynchronize(stream));
      }
    } else {
      auto outs = ctx.MultiOutput<Tensor>("Out");
      for (size_t i = 0; i < outs.size(); ++i) {
        PADDLE_ENFORCE(platform::dynload::ncclBcast(
            outs[i]->mutable_data<T>(ctx.GetPlace()), outs[i]->numel(),
            NCCLTypeWrapper<T>::type, root, comm->comms_[idx], stream));
        PADDLE_ENFORCE(cudaStreamSynchronize(stream));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(ncclAllReduce, ops::NCCLAllReduceKernel<float>);
REGISTER_OP_GPU_KERNEL(ncclBcastSend, ops::NCCLBcastKernel<float>);
REGISTER_OP_GPU_KERNEL(ncclReduce, ops::NCCLReduceKernel<float>);
REGISTER_OP_GPU_KERNEL(ncclBcastRecv, ops::NCCLBcastKernel<float>);

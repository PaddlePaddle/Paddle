/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

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
class AllGatherCUDAKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    int device_count = platform::GetCUDADeviceCount();
    auto x_dim = in->dims();
    x_dim[0] *= device_count;
    T* out_data = out->mutable_data<T>(x_dim, in->place());
    const T* in_data = in->data<T>();

    std::vector<int> gpus(device_count);
    for (int i = 0; i < device_count; ++i) {
      gpus[i] = i;
    }
    platform::Communicator comm;
    comm.InitAll(gpus);
    int gpu_id =
        boost::get<platform::CUDAPlace>(context.GetPlace()).GetDeviceId();
    int idx = comm.GetCommId(gpu_id);

    PADDLE_ENFORCE(platform::dynload::ncclAllGather(
        in_data, out_data, in->numel(), NCCLTypeWrapper<T>::type,
        comm.comms().at(idx), context.cuda_device_context().stream()));
  }
};

template <typename T>
class AllGatherGradCUDAKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    platform::Communicator comm;
    int gpu_id =
        boost::get<platform::CUDAPlace>(context.GetPlace()).GetDeviceId();
    int idx = comm.GetCommId(gpu_id);
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    in_grad->mutable_data<T>(context.GetPlace());
    auto out_grad_dims = out_grad->dims();
    auto in_grad_dims = in_grad->dims();
    framework::Tensor sub_tensor =
        out_grad->Slice(idx * in_grad_dims[0], (idx + 1) * in_grad_dims[0]);
    framework::TensorCopy(sub_tensor, context.GetPlace(), in_grad);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(all_gather, ops::AllGatherCUDAKernel<float>);
REGISTER_OP_CUDA_KERNEL(all_gather_grad, ops::AllGatherGradCUDAKernel<float>);

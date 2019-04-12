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

#pragma once
#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AllReduceOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place),
                   "AllReduce op can run on gpu place only for now.");
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    int dtype = platform::ToNCCLDataType(in->type());
    int64_t numel = in->numel();
    auto* sendbuff = in->data<void>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    auto* comm = dev_ctx.nccl_comm();
    // FIXME(typhoonzero): should use nccl stream here.
    auto stream = dev_ctx.stream();
    PADDLE_ENFORCE_NOT_NULL(stream, "Should initialize NCCL firstly.");

    int reduce_type = ctx.Attr<int>("reduce_type");
    ncclRedOp_t red_type = ncclSum;
    switch (reduce_type) {
      case 0:
        red_type = ncclSum;
        break;
      case 1:
        red_type = ncclProd;
        break;
      case 2:
        red_type = ncclMax;
        break;
      case 3:
        red_type = ncclMin;
        break;
    }

    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, static_cast<ncclDataType_t>(dtype), red_type,
        comm, stream));
  }
};

}  // namespace operators
}  // namespace paddle

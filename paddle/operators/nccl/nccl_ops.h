/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/nccl/nccl_gpu_common.h"

#include <string.h>

namespace paddle {
namespace operators {

using framework::Tensor;

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

class NCCLInitOp : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto gpus = ctx.Input<std::vector<int>>("gpus");
    auto* comm = ctx.Output<Communicator>("Communicator");
    comm->mutable_data<Communicator>(CPUPlace());
    comm = NCCLManager::GetCommunicator(gpus);
  }
};

template <typename T>
class NCCLAllReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<Tensor>("X");
    auto outs = ctx.MultiOutput<Tensor>("Out");
    std::string reduction = ctx.Attr<std::string>("reduction");
    std::vector<int> gpus = ctx.Attr<std::vector<int>>("gpus");
    ncclRedOp_t op_type;
    if (reduction == "ncclSum") {
      op_type = ncclSum;
    } else if (reduction == "ncclProd") {
      op_type = ncclProd;
    } else if (reduction == "ncclMin") {
      op_type = ncclMin;
    } else if (reduction == "ncclMax") {
      op_type = ncclMax;
    }

    auto* comm = ctx.Input<Communicator>("Communicator");

    auto dev_ctx =
        static_cast<const platform::CUDADeviceContext>(ctx.device_context());

    // platform::NCCLManager* m = platform::NCCLManager::Get();

    // auto* comm = m->GetCommunicator(gpus);
    // comm->wg_.Add(1);

    auto stream = dev_ctx.stream();

    // device id
    int gid = static_cast<platform::GPUPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = gid % gpus.size();
    comm->streams_[idx] = stream;

    for (size_t i = 0; i < ins.size(); ++i) {
      PADDLE_ENFORCE(
          ncclAllReduce(ins[i]->data<T>(), outs[i]->mutable_data<T>(),
                        outs[i]->numel() * sizeof(T), NCCLTypeWrapper<T>::type,
                        op_type, comm->comms_[idx], comm->streams_[idx]));
      PADDLE_ENFORCE(cudaEventRecord(comm->events_[idx], comm->streams_[idx]));

      // // wait finish
      // PADDLE_ENFORCE(
      //     cudaStreamWaitEvent(comm->streams_[idx], comm->events_[idx], 0));
    }

    // comm->wg_.Done();

    // comm->wg_.Wait();
  }
};

}  // namespace operators
}  // namespace paddle

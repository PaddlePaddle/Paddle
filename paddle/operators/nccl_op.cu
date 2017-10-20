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

#define EIGEN_USE_GPU
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
    std::string reduction = ctx.Attr<std::string>("reduction");
    ncclRedOp_t op_type;
    if (reduction == "ncclSum") {
      op_type = ncclSum;
    } else if (reduction == "ncclProd") {
      op_type = ncclProd;
    } else if (reduction == "ncclMin") {
      op_type = ncclMin;
    } else if (reduction == "ncclMax") {
      op_type = ncclMax;
    } else {
      PADDLE_ENFORCE(false, "reduction error.");
    }

    auto* comm = ctx.Input<Communicator>("Communicator");

    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();

    // device id
    int device_id =
        boost::get<platform::GPUPlace>(ctx.GetPlace()).GetDeviceId();
    int idx = comm->GetCommId(device_id);

    for (size_t i = 0; i < ins.size(); ++i) {
      PADDLE_ENFORCE(ncclAllReduce(
          ins[i]->data<T>(), outs[i]->mutable_data<T>(ctx.GetPlace()),
          outs[i]->numel() * sizeof(T), NCCLTypeWrapper<T>::type, op_type,
          comm->comms_[idx], stream));
      PADDLE_ENFORCE(cudaStreamSynchronize(stream));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(ncclAllReduce, ops::NCCLAllReduceKernel<float>);
